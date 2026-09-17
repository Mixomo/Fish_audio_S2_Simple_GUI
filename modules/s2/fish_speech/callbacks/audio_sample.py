from pathlib import Path
import json
import os
from typing import Optional

import torch
import torchaudio
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from loguru import logger


class AudioSampleCallback(Callback):
    """
    Generates a sample wav file whenever a checkpoint is saved.
    Reuses the training model (no second copy loaded). Before inference,
    optimizer states are offloaded to CPU and gradients are freed to make
    room for KV caches and the codec model.
    """

    def __init__(
        self,
        text: str,
        codec_checkpoint_path: str,
        output_dir: Optional[str] = None,
        audio_format: str = "wav",
    ):
        super().__init__()
        self.text = text
        self.codec_checkpoint_path = codec_checkpoint_path
        self.output_dir = Path(output_dir) if output_dir else None
        self.audio_format = audio_format
        self.enabled = False
        self.every_n_steps = 0
        self.seed = 1234
        self.reference_audio_path = None
        self.reference_text = None
        self.reference_source = "none"
        self._prompt_tokens_cpu = None
        config_path = os.environ.get("FISH_AUDIO_SAMPLE_CONFIG", "").strip()
        if config_path and Path(config_path).is_file():
            try:
                payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
                self.text = str(payload.get("text") or self.text)
                self.every_n_steps = max(0, int(payload.get("every_n_steps", 0) or 0))
                if payload.get("output_dir"):
                    self.output_dir = Path(payload["output_dir"])
                self.audio_format = str(payload.get("audio_format") or self.audio_format)
                self.seed = int(payload.get("seed", self.seed) or self.seed)
                self.reference_audio_path = payload.get("reference_audio_path") or None
                self.reference_text = payload.get("reference_text") or None
                self.reference_source = str(payload.get("reference_source") or "none")
                self.enabled = self.every_n_steps > 0
                logger.info(f"AudioSampleCallback: Easy GUI evaluation contract loaded from {config_path}")
            except Exception as exc:
                logger.warning(f"AudioSampleCallback: could not read Easy GUI evaluation contract: {exc}")
        self._codec = None
        self._last_sampled_step = -1

    # ------------------------------------------------------------------
    # Memory management: free training-only GPU allocations for inference
    # ------------------------------------------------------------------

    @staticmethod
    def _offload_optimizer(trainer):
        """Move only CUDA optimizer tensors to CPU and remember exactly what moved."""
        moved = []
        for optimizer in trainer.optimizers:
            optimizer.zero_grad(set_to_none=True)
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        moved.append((state, key, value.device))
                        state[key] = value.cpu()
        return moved

    @staticmethod
    def _restore_optimizer(moved):
        """Restore only tensors offloaded by _offload_optimizer."""
        for state, key, device in moved:
            value = state.get(key)
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                state[key] = value.to(device)

    # ------------------------------------------------------------------
    # Codec
    # ------------------------------------------------------------------

    def _ensure_codec(self, device):
        if self._codec is not None:
            return
        from fish_speech.models.text2semantic.inference import load_codec_model

        logger.info("AudioSampleCallback: loading codec model...")
        self._codec = load_codec_model(self.codec_checkpoint_path, device)
        logger.info("AudioSampleCallback: codec loaded.")

    def _free_codec(self):
        del self._codec
        self._codec = None


    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_sample(self, trainer: Trainer, pl_module: LightningModule):
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_ar,
            decode_to_audio,
            generate_long,
        )

        device = pl_module.device
        step = trainer.global_step
        print(f"[UI-PROGRESS] Evaluation sample · step {step} · starting", flush=True)

        # Free training memory to make room for KV caches + codec
        moved_optimizer_state = self._offload_optimizer(trainer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = pl_module.model
        was_training = model.training
        model.eval()

        try:
            self._ensure_codec(device)

            # Keep every checkpoint comparison deterministic. Fish S2's default
            # finetuning workflow remains prompt-conditioned for stable speaker
            # identity, so use one fixed dataset reference whenever available.
            try:
                from fish_speech.utils import set_seed
                set_seed(int(self.seed))
            except Exception:
                torch.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))

            prompt_tokens = None
            prompt_text = None
            if self.reference_audio_path and self.reference_text and Path(self.reference_audio_path).is_file():
                if self._prompt_tokens_cpu is None:
                    wav, source_sr = torchaudio.load(str(self.reference_audio_path))
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    target_sr = int(self._codec.sample_rate)
                    if int(source_sr) != target_sr:
                        wav = torchaudio.functional.resample(wav, int(source_sr), target_sr)
                    wav = wav[0].to(device)
                    audios = wav[None, None, :].to(
                        dtype=next(self._codec.parameters()).dtype
                    )
                    audio_lengths = torch.tensor(
                        [wav.shape[0]], device=device, dtype=torch.long
                    )
                    indices, feature_lengths = self._codec.encode(audios, audio_lengths)
                    self._prompt_tokens_cpu = indices[
                        0, :, : feature_lengths[0]
                    ].detach().cpu()
                prompt_tokens = [self._prompt_tokens_cpu]
                prompt_text = [str(self.reference_text)]
                print(
                    f"[UI-PROGRESS] Evaluation sample · step {step} · prompted "
                    f"({self.reference_source}) · seed {self.seed}",
                    flush=True,
                )
            else:
                print(
                    f"[UI-PROGRESS] Evaluation sample · step {step} · no prompt fallback · seed {self.seed}",
                    flush=True,
                )

            results = list(
                generate_long(
                    model=model,
                    device=device,
                    decode_one_token=decode_one_token_ar,
                    text=self.text,
                    num_samples=1,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                )
            )

            # Collect codes from all "sample" responses and concatenate
            code_chunks = [r.codes for r in results if r.codes is not None]
            if not code_chunks:
                logger.warning(
                    "AudioSampleCallback: generate_long returned no samples."
                )
                return

            codes = (
                torch.cat(code_chunks, dim=1)
                if len(code_chunks) > 1
                else code_chunks[0]
            )

            # Decode codes to audio with the codec
            codes = codes.to(device=device)
            wav = decode_to_audio(codes, self._codec)
            # TensorBoard SummaryWriter.add_audio and torchaudio.save both want
            # channels-first audio. Keep a canonical mono [1, T] tensor.
            wav = wav.detach().float().cpu().reshape(1, -1)
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
            wav = wav.clamp(-1.0, 1.0)

            # Determine output path
            if self.output_dir:
                out_dir = self.output_dir
            else:
                out_dir = Path(trainer.log_dir) / "audio_samples"
            out_dir.mkdir(parents=True, exist_ok=True)

            sample_rate = int(self._codec.sample_rate)
            mode = "prompted" if prompt_tokens is not None else "no_prompt"
            out_path = out_dir / f"step_{step:09d}_{mode}.{self.audio_format}"
            torchaudio.save(str(out_path), wav, sample_rate, format=self.audio_format)
            logger.info(f"AudioSampleCallback: saved sample to {out_path}")
            print(f"[UI-PROGRESS] Evaluation sample · step {step} · saved {out_path.name}", flush=True)

            # Log the exact same waveform to every TensorBoard SummaryWriter.
            # Logging failures are isolated so a good WAV never invalidates the
            # evaluation sample or training run.
            tb_logged = 0
            for lg in trainer.loggers:
                experiment = getattr(lg, "experiment", None)
                if experiment is None or not hasattr(experiment, "add_audio"):
                    continue
                try:
                    experiment.add_audio(
                        "evaluation/prompted_audio" if prompt_tokens is not None else "evaluation/no_prompt_audio",
                        wav,
                        global_step=int(step),
                        sample_rate=sample_rate,
                    )
                    if hasattr(experiment, "add_text"):
                        experiment.add_text(
                            "evaluation/generated_text",
                            str(self.text),
                            global_step=int(step),
                        )
                        experiment.add_text(
                            "evaluation/reference",
                            (
                                f"mode={'prompted' if prompt_tokens is not None else 'no-prompt'}; "
                                f"source={self.reference_source}; seed={self.seed}; "
                                f"reference_text={self.reference_text or ''}"
                            ),
                            global_step=int(step),
                        )
                    if hasattr(experiment, "flush"):
                        experiment.flush()
                    tb_logged += 1
                except Exception as tb_exc:
                    logger.warning(
                        f"AudioSampleCallback: TensorBoard audio logging failed "
                        f"for {type(lg).__name__}: {tb_exc}"
                    )
            if tb_logged:
                logger.info(
                    f"AudioSampleCallback: TensorBoard evaluation/audio logged "
                    f"at step {step} ({tb_logged} logger(s))."
                )
                print(
                    f"[UI-PROGRESS] Evaluation sample · step {step} · TensorBoard audio logged",
                    flush=True,
                )
            else:
                logger.warning(
                    "AudioSampleCallback: no TensorBoard SummaryWriter with add_audio was available."
                )

        except Exception as e:
            logger.exception(
                f"AudioSampleCallback: failed to generate sample: {e}"
            )
        finally:
            self._teardown_caches(model)
            self._free_codec()
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._restore_optimizer(moved_optimizer_state)

    # ------------------------------------------------------------------
    # Cache teardown
    # ------------------------------------------------------------------

    @staticmethod
    def _teardown_caches(model):
        """Reset KV caches that were created during inference back to None."""
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "attention"):
                layer.attention.kv_cache = None

        for layer in getattr(model, "fast_layers", []):
            if hasattr(layer, "attention"):
                layer.attention.kv_cache = None

        if hasattr(model, "_cache_setup_done"):
            model._cache_setup_done = False
        if hasattr(model, "max_seq_len"):
            model.max_seq_len = -1
        if hasattr(model, "max_batch_size"):
            model.max_batch_size = -1

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _checkpoint_interval(self, trainer) -> int:
        """Return every_n_train_steps from ModelCheckpoint, or 0 if not found."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb._every_n_train_steps:
                return cb._every_n_train_steps
        return 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0 or not self.enabled:
            return
        step = trainer.global_step
        if step <= 0 or step == self._last_sampled_step:
            return
        interval = self.every_n_steps
        if interval <= 0 or step % interval != 0:
            return
        self._last_sampled_step = step
        self._generate_sample(trainer, pl_module)
