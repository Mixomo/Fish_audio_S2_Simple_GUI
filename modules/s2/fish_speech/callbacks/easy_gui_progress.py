from __future__ import annotations

import json
import os
import time
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class EasyGUIProgressCallback(Callback):
    """Small sidecar callback used by the Windows Easy GUI.

    It does not affect optimization.  It writes an atomic JSON snapshot and a
    carriage-return progress line so the parent CMD and embedded Live Console
    show the same training status without parsing Lightning's Rich renderer.
    """

    def __init__(self):
        super().__init__()
        self.path = Path(os.environ.get("FISH_EASY_PROGRESS_FILE", "")) if os.environ.get("FISH_EASY_PROGRESS_FILE") else None
        self.total_steps = max(1, int(os.environ.get("FISH_EASY_TOTAL_STEPS", "1") or 1))
        self.steps_per_epoch = max(1, int(os.environ.get("FISH_EASY_STEPS_PER_EPOCH", "1") or 1))
        self.save_every = max(1, int(os.environ.get("FISH_EASY_SAVE_EVERY_STEPS", "1") or 1))
        self.eval_every = max(1, int(os.environ.get("FISH_EASY_EVAL_EVERY_STEPS", "1") or 1))
        self.started = time.time()
        self._last_step = -1

    def _write(self, trainer, state="running"):
        if trainer.global_rank != 0:
            return
        step = int(trainer.global_step)
        metrics = trainer.callback_metrics
        loss = metrics.get("train/loss")
        try:
            loss = float(loss.detach().cpu()) if loss is not None else None
        except Exception:
            loss = None
        payload = {
            "state": state,
            "step": step,
            "total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "epoch_equivalent": step / self.steps_per_epoch,
            "loss": loss,
            "save_every_steps": self.save_every,
            "eval_every_steps": self.eval_every,
            "next_save_step": min(self.total_steps, ((step // self.save_every) + 1) * self.save_every) if step < self.total_steps else self.total_steps,
            "next_eval_step": min(self.total_steps, ((step // self.eval_every) + 1) * self.eval_every) if step < self.total_steps else self.total_steps,
            "elapsed": max(0.0, time.time() - self.started),
            "timestamp": time.time(),
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path)
        pct = min(100.0, step / self.total_steps * 100.0)
        width = 28
        filled = int(round(width * pct / 100.0))
        bar = "█" * filled + "░" * (width - filled)
        loss_text = "--" if loss is None else f"{loss:.5f}"
        elapsed = max(0.0, time.time() - self.started)
        eta = (elapsed / step * (self.total_steps - step)) if step > 0 and self.total_steps > step else 0.0
        def _clock(value):
            value = max(0, int(value or 0)); h, rem = divmod(value, 3600); m, sec = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
        next_save = min(self.total_steps, ((step // self.save_every) + 1) * self.save_every)
        next_eval = min(self.total_steps, ((step // self.eval_every) + 1) * self.eval_every)
        print(
            f"[UI-PROGRESS] Training           [{bar}] {pct:6.2f}% "
            f"Step {step}/{self.total_steps} · Epoch≈{step/self.steps_per_epoch:.2f} · Loss {loss_text} · "
            f"Elapsed {_clock(elapsed)} · ETA {_clock(eta)} · Next save {next_save} · Next eval {next_eval}",
            end="\r" if state == "running" else "\n",
            flush=True,
        )

    def on_train_start(self, trainer, pl_module):
        self._write(trainer, "running")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step != self._last_step:
            self._last_step = step
            self._write(trainer, "running")

    def on_train_end(self, trainer, pl_module):
        self._write(trainer, "complete")
