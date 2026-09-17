from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path

import gradio as gr

import fish_backend as B
from fish_easy.console import html_view, log
from fish_easy.projects import (
    PROJECTS, clone_project, delete_surface, list_projects, load_project,
    save_project, update_surface,
)

APP_TITLE = "Fish Audio S2 Easy GUI"
NONE = "None"
MAX_DIALOGUE_SEGMENTS = 20

# Copied structurally from the newest CosyVoice3 Easy GUI. Native Gradio theme
# owns colors/fonts; this CSS only enforces family hierarchy, spacing and cards.
CSS = """
html, body, #root { width:100%; min-width:0; }
.gradio-container { max-width:none !important; width:100% !important; margin:0 auto !important; padding:18px 28px 28px !important; }
.tabs, .tabitem, .gradio-container .block, .gradio-container .form { width:100% !important; max-width:none !important; }
.title-section { border-bottom:1px solid var(--border-color-primary); margin-bottom:8px; padding-bottom:8px; align-items:center !important; }
.title-section .prose, .title-section h1 { margin:0 !important; padding:0 !important; }
.title-section button { min-height:36px !important; white-space:nowrap; }
.tab-subtitle { opacity:.78; margin:0 0 12px 0 !important; padding:0 !important; }
.section-heading { margin:10px 0 2px !important; }
.global-toolbar, .project-strip { padding:4px 2px !important; border:0 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important; }
.project-strip { padding-bottom:12px !important; margin-bottom:10px !important; border-bottom:1px solid var(--border-color-primary) !important; }
.global-toolbar { margin:4px 0 10px !important; }
.global-toolbar button { min-height:38px !important; }
.runtime-row, .workflow-actions, .dialogue-toolbar { gap:10px !important; align-items:end !important; }
.compact-button { min-width:52px !important; max-width:110px !important; }
.medium-control { max-width:420px !important; }
.small-control { max-width:260px !important; }
.dialogue-turn-card { padding:14px 14px 12px !important; margin:6px 0 12px !important; border:1px solid var(--border-color-primary) !important; border-radius:10px !important; background:transparent !important; box-shadow:none !important; }
.dialogue-turn-title { margin:0 0 6px 0 !important; opacity:.86; }
.dialogue-actions { gap:7px !important; align-items:center !important; }
.dialogue-actions button { min-width:88px !important; padding-left:10px !important; padding-right:10px !important; white-space:nowrap !important; }
.audio-safe-space { overflow:visible !important; padding-bottom:18px !important; border:0 !important; box-shadow:none !important; }
.output-clean, .output-clean > div, .output-clean .wrap { border:0 !important; box-shadow:none !important; }
.compact-status { margin:0 0 8px 0 !important; padding:0 !important; min-height:0 !important; }
.console-accordion, .console-accordion > div { border-radius:8px !important; }
.training-progress-card { padding:10px 12px; border:1px solid var(--border-color-primary); border-radius:9px; background:transparent; }
.training-progress-head { display:flex; justify-content:space-between; gap:16px; margin-bottom:8px; }
.training-progress-track { width:100%; height:10px; border-radius:999px; background:var(--background-fill-secondary); overflow:hidden; }
.training-progress-fill { height:100%; border-radius:999px; background:var(--button-primary-background-fill); transition:width .25s ease; }
.training-progress-meta { margin-top:8px; opacity:.82; font-size:.92em; }
.voice-library-row { align-items:end !important; gap:8px !important; }
.voice-library-row button { min-width:48px !important; max-width:52px !important; }
footer { display:none !important; }
@media(max-width:900px){ .gradio-container{padding:10px!important}.title-section button{min-width:100% !important;} .medium-control,.small-control{max-width:none!important;} }
"""


def _whisper_model(display): return B.WHISPER_MODELS.get(display, display)

def _voice_choices(): return [NONE, *B.get_sample_choices()]

def _adapter_choices():
    # Always rebuild from results/<project>/checkpoints recursively so every
    # checkpoint (including nested Lightning checkpoint folders) is visible.
    return B.get_lora_adapter_choices()

def _sample_payload(name):
    if not name or name == NONE:
        return None, "", "No saved voice selected."
    audio, text = B.load_sample(name)
    return audio, text, f"Loaded '{name}'."

def _refresh_voice(current=NONE):
    choices = _voice_choices()
    value = current if current in choices else NONE
    return gr.update(choices=choices, value=value)


def _delete_voice(name):
    if not name or name == NONE:
        return gr.update(choices=_voice_choices(), value=NONE), None, "", "Select a saved voice to delete."
    msg, _ = B.delete_sample(name)
    return gr.update(choices=_voice_choices(), value=NONE), None, "", msg

def _save_voice(audio, name, text):
    status, update = B.save_prep_sample(audio, name, text)
    selected = None
    try:
        selected = update.get("value") if isinstance(update, dict) else None
    except Exception:
        selected = None
    choices = _voice_choices()
    if selected not in choices:
        selected = choices[-1] if len(choices) > 1 else NONE
    return status, gr.update(choices=choices, value=selected)

_adapter_inventory_cache = None

def _adapter_inventory():
    choices = _adapter_choices()
    values = tuple(v for _, v in choices)
    return choices, values

def _refresh_adapters(current=NONE):
    global _adapter_inventory_cache
    choices, values = _adapter_inventory()
    _adapter_inventory_cache = values
    selected = current if current in values else NONE
    return gr.update(choices=choices, value=selected)


def _engine_visibility(engine):
    B.switch_inference_engine(engine)
    return gr.update(visible=(engine == B.ENGINE_CPP))


def _transcribe(audio, model_display, batch_size=4):
    # Fish synthesis is language-free and the Easy GUI keeps ASR equally simple:
    # Faster-Whisper auto-detects instead of exposing another language dropdown.
    return B.transcribe_only(audio, _whisper_model(model_display), "Auto-detect", batch_size)


def _browse_folder(current=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        folder=filedialog.askdirectory(initialdir=current or str(Path.cwd()))
        root.destroy(); return folder or current or ""
    except Exception as exc:
        return current or f"Folder picker unavailable: {exc}"


def _clear_vram():
    return B.unload_all_models(
        reason="manual Unload All Models / Free RAM - VRAM",
        reset_compiler=True,
        trim_ram=True,
    )


def _clear_outputs():
    root=Path(B.OUTPUTS_DIR); root.mkdir(parents=True, exist_ok=True); removed=0
    for item in root.iterdir():
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink(); removed += 1
        except Exception: pass
    return f"Removed {removed} generated output item(s)."


def _inference_ui_start():
    B.begin_inference()
    return gr.update(interactive=False), gr.update(interactive=True), "⏳ Inference running…"


def _inference_ui_finish(status):
    return gr.update(interactive=True), gr.update(interactive=False), status


def _stop_inference_ui():
    status = B.stop_inference()
    return gr.update(interactive=True), gr.update(interactive=False), status


def _resolve_inference_seed(seed, fixed_seed):
    # Unlocked = a fresh random seed for every Generate click, regardless of
    # the value left visible in the Seed box by the previous generation.
    # Locked = preserve/use the visible manual or previously generated seed.
    if bool(fixed_seed):
        chosen = B.normalize_seed(seed)
        return chosen if chosen is not None else B.random_seed_value()
    return B.random_seed_value()


def _run_single(engine, cpp_model, cpp_backend, cpp_device, codec_policy, adapter, text, ref_audio, ref_text,
                top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, seed, fixed_seed):
    chosen=_resolve_inference_seed(seed, fixed_seed)
    wav,status,_=B.clone_voice(
        engine, cpp_model, cpp_backend, cpp_device, codec_policy, adapter, text, ref_audio, ref_text,
        top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, B.INTERNAL_TORCH_COMPILE_MODE, chosen
    )
    return wav,status,chosen,chosen

def _run_dialogue(engine, cpp_model, cpp_backend, cpp_device, codec_policy, adapter,
                  top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, seed, fixed_seed, row_count, silence, *args):
    chosen=_resolve_inference_seed(seed, fixed_seed)
    wav,status,_=B.generate_dialogue(
        engine, cpp_model, cpp_backend, cpp_device, codec_policy, adapter,
        top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, B.INTERNAL_TORCH_COMPILE_MODE, chosen, row_count, silence, *args
    )
    return wav,status,chosen,chosen

def _dialogue_refresh(*current):
    choices=_voice_choices()
    return [gr.update(choices=choices, value=(v if v in choices else NONE)) for v in current]


def _dialogue_mutate(action, index, count, *args):
    samples=list(args[:MAX_DIALOGUE_SEGMENTS]); texts=list(args[MAX_DIALOGUE_SEGMENTS:]); count=int(count)
    if action == "insert" and count < MAX_DIALOGUE_SEGMENTS:
        samples.insert(index+1, samples[index]); texts.insert(index+1, ""); samples.pop(); texts.pop(); count += 1
    elif action == "clone" and count < MAX_DIALOGUE_SEGMENTS:
        samples.insert(index+1, samples[index]); texts.insert(index+1, texts[index]); samples.pop(); texts.pop(); count += 1
    elif action == "remove" and count > 1:
        samples.pop(index); texts.pop(index); samples.append(NONE); texts.append(""); count -= 1
    return [count] + [gr.update(value=samples[i], visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)] + [gr.update(value=texts[i], visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)] + [gr.update(visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)]


def _project_update(surface, value=None, preserve=None):
    projects=list_projects(surface)
    choices=[NONE, *projects]
    candidate=preserve if preserve not in (None, "", NONE) else value
    selected=candidate if candidate in projects else NONE
    return gr.update(choices=choices, value=selected)


def _create_project(name, dataset_name, folder, wm, batch, current_train):
    try:
        project=save_project(name, {"dataset":{}, "training":{}})
        update_surface(project, "dataset", {"dataset_name":dataset_name or project, "source_folder":folder or "", "whisper_model":wm, "batch_size":int(batch)})
        return _project_update("dataset", project), _project_update("training", project, current_train), project, f"Project '{project}' created."
    except Exception as exc:
        return _project_update("dataset"), _project_update("training"), dataset_name, f"Create failed: {exc}"


def _save_dataset_project(project, dataset_name, folder, wm, batch, current_train):
    if project in (None, "", NONE): return _project_update("training", preserve=current_train), "Select a Dataset project first."
    update_surface(project, "dataset", {"dataset_name":dataset_name or project, "source_folder":folder or "", "whisper_model":wm, "batch_size":int(batch)})
    return _project_update("training", preserve=current_train), f"Saved Dataset state for '{project}'. Training state was preserved."


def _load_dataset_project(project):
    if project in (None, "", NONE):
        return "", "", "large-v3 (~10 GB VRAM)", 8
    d=load_project(project).get("dataset", {})
    return d.get("dataset_name", project), d.get("source_folder", ""), d.get("whisper_model", "large-v3 (~10 GB VRAM)"), d.get("batch_size", 8)


def _clone_project_ui(project, current_other=None, source_surface="dataset"):
    try:
        cloned=clone_project(project)
        return _project_update("dataset", cloned if source_surface=="dataset" else None, current_other if source_surface=="training" else None), _project_update("training", cloned if source_surface=="training" else None, current_other if source_surface=="dataset" else None), cloned, f"Cloned '{project}' → '{cloned}'."
    except Exception as exc:
        return _project_update("dataset", project), _project_update("training"), project, f"Clone failed: {exc}"


def _clone_training_project_ui(project, current_dataset):
    try:
        cloned=clone_project(project)
        return _project_update("dataset", preserve=current_dataset), _project_update("training", cloned), f"Cloned '{project}' → '{cloned}'."
    except Exception as exc:
        return _project_update("dataset", preserve=current_dataset), _project_update("training", project), f"Clone failed: {exc}"


def _delete_dataset_project(project, current_train):
    if project in (None, "", NONE): return _project_update("dataset"), _project_update("training", preserve=current_train), "Select a Dataset project first."
    msg=delete_surface(project, "dataset")
    return _project_update("dataset"), _project_update("training", preserve=current_train), msg


def _delete_training_project(project, current_dataset):
    if project in (None, "", NONE): return _project_update("training"), _project_update("dataset", preserve=current_dataset), "Select a Training project first."
    B.prepare_tensorboard_project(None)
    msg=delete_surface(project, "training")
    return _project_update("training"), _project_update("dataset", preserve=current_dataset), msg


def _save_training_project(project, dataset, preset, steps,
                           lr, rank, alpha, dropout, training_seed, save_steps, eval_steps, resume,
                           eval_enabled, eval_text, weight_decay, max_length, causal, interactive_prob,
                           gradient_clip, scheduler,
                           current_dataset_project):
    if project in (None, "", NONE):
        return _project_update("dataset", preserve=current_dataset_project), "Select a Training project first."
    update_surface(project, "training", {
        "dataset":dataset, "vram_preset":preset,
        "max_steps":int(steps),
        "learning_rate":float(lr), "rank":int(rank), "alpha":int(alpha),
        "dropout":float(dropout), "training_seed":int(training_seed),
        "save_every_steps":int(save_steps), "eval_every_steps":int(eval_steps),
        "resume":resume, "weight_decay":float(weight_decay), "max_length":int(max_length),
        "causal":bool(causal), "interactive_prob":float(interactive_prob), "gradient_clip":float(gradient_clip),
        # Persist the single supported schedule even when loading a project
        # created by an older GUI revision.
        "scheduler":"Cosine 10% floor",
        "eval":{"enabled":bool(eval_enabled), "text":eval_text or ""},
    })
    return _project_update("dataset", preserve=current_dataset_project), f"Saved Training + Evaluation state for '{project}'. Dataset state was preserved."


def _load_training_project(project):
    datasets=B.get_dataset_choices()
    if project in (None, "", NONE):
        return (
            gr.update(choices=datasets, value=None),
            "24GB VRAM",500,1e-5,32,16,0.10,1234,
            50,50,gr.update(choices=["Fresh / None"],value="Fresh / None"),
            False,"This is a fixed evaluation sample generated during training.",
            0.01,4096,False,1.0,1.0,"Cosine 10% floor"
        )
    data=load_project(project); t=data.get("training",{}); d=data.get("dataset",{}); ev=t.get("eval",{})
    dataset=t.get("dataset") or d.get("dataset_name")
    dataset_choices=list(datasets)
    if dataset and dataset not in dataset_choices: dataset_choices.append(dataset)
    if not dataset: dataset=None
    resume_saved=t.get("resume") or "Fresh / None"
    resume_choices=["Fresh / None", *B.handle_lora_list_checkpoints(project)]
    if resume_saved not in resume_choices: resume_choices.append(resume_saved)
    return (
        gr.update(choices=dataset_choices,value=dataset),
        t.get("vram_preset","24GB VRAM"),
        t.get("max_steps",250),t.get("learning_rate",1e-5),t.get("rank",32),t.get("alpha",16),t.get("dropout",0.10),
        t.get("training_seed",1234),t.get("save_every_steps",50),t.get("eval_every_steps",50),
        gr.update(choices=resume_choices,value=resume_saved),
        bool(ev.get("enabled", False)), ev.get("text","This is a fixed evaluation sample generated during training."),
        t.get("weight_decay",0.01),t.get("max_length",4096),t.get("causal",False),
        t.get("interactive_prob",1.0),t.get("gradient_clip",1.0),
        "Cosine 10% floor"
    )


def _checkpoint_update(project, current="Fresh / None"):
    ckpts=B.handle_lora_list_checkpoints(project) if project not in (None, "", NONE) else []
    choices=["Fresh / None", *ckpts]
    return gr.update(choices=choices, value=current if current in choices else "Fresh / None")


def _prepare_training_project(project, current_resume):
    B.prepare_tensorboard_project(project)
    return _checkpoint_update(project, current_resume)


def _stop_training():
    return B.stop_training()


def _batch_dataset(folder, dataset, wm, batch): return B.handle_full_batch_process(folder, dataset, _whisper_model(wm), "Auto-detect", batch)

def _new_seed_pair():
    value=B.random_seed_value(); return value,value


def _autotune_ui(dataset, preset):
    tuned=B.autotune_training(dataset, preset)
    return (
        gr.update(value=tuned["steps"]), gr.update(value=tuned["save_steps"]),
        gr.update(value=tuned["eval_steps"]), gr.update(value=tuned["rank"]),
        gr.update(value=tuned["alpha"]), gr.update(value=tuned["dropout"]),
        gr.update(value=tuned["lr"]), gr.update(value=tuned["weight_decay"]),
        gr.update(value=tuned["max_length"]), gr.update(value=tuned["causal"]),
        gr.update(value=tuned["interactive_prob"]), gr.update(value=tuned["gradient_clip"]),
        gr.update(value=tuned["scheduler"]), gr.update(value=tuned["training_seed"]), tuned["report"],
    )


def _fmt_duration(seconds):
    if seconds is None: return "--"
    seconds=max(0,int(seconds)); h,rem=divmod(seconds,3600); m,sec=divmod(rem,60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def training_progress_html(snapshot=None):
    snap=snapshot or B.training_progress_snapshot()
    pct=min(100.0,max(0.0,float(snap.get("pct",0.0) or 0.0)))
    running=bool(snap.get("running")); code=snap.get("exit_code")
    state="Running" if running else ("Complete" if code == 0 else ("Stopped / Failed" if code is not None else "Idle"))
    loss=snap.get("loss"); loss_text="--" if loss is None else f"{float(loss):.5f}"
    completed=int(snap.get("step", 0) or 0)
    total=int(snap.get("total_steps", 0) or 0)
    progress_text=f"Step {completed}/{total or '--'}"
    next_bits=[]
    if int(snap.get("next_save_step",0) or 0): next_bits.append(f"Next save {int(snap['next_save_step'])}")
    if int(snap.get("next_eval_step",0) or 0): next_bits.append(f"Next eval {int(snap['next_eval_step'])}")
    cadence=(" · " + " · ".join(next_bits)) if next_bits else ""
    meta=(f"{progress_text} · "
          f"Loss {loss_text} · Elapsed {_fmt_duration(snap.get('elapsed'))} · ETA {_fmt_duration(snap.get('eta'))}{cadence}")
    return (f'<div class="training-progress-card"><div class="training-progress-head"><b>{html.escape(state)}</b>'
            f'<span>{pct:.1f}%</span></div><div class="training-progress-track"><div class="training-progress-fill" '
            f'style="width:{pct:.2f}%"></div></div><div class="training-progress-meta">{html.escape(meta)}</div></div>')


def training_poll_ui(current_adapter):
    global _adapter_inventory_cache
    snap=B.training_progress_snapshot(); running=bool(snap.get("running"))
    choices, values = _adapter_inventory()

    # Do not rewrite the inference dropdown every timer tick. Gradio can briefly
    # receive a stale None value during a choices update, which used to erase the
    # checkpoint list/selection. Only update when files were actually added,
    # removed or renamed.
    if _adapter_inventory_cache != values:
        _adapter_inventory_cache = values
        adapter_update = gr.update(
            choices=choices,
            value=current_adapter if current_adapter in values else NONE,
        )
    else:
        adapter_update = gr.update()

    return B.training_status(), training_progress_html(snap), gr.update(interactive=not running), gr.update(interactive=running), adapter_update


_voice_init=_voice_choices()
_dataset_projects=[NONE, *list_projects("dataset")]
_training_projects=[NONE, *list_projects("training")]
_datasets=B.get_dataset_choices()

with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Default()) as app:
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=7, min_width=420):
            gr.Markdown("# 🎙️ Fish Audio S2 Easy GUI")
            gr.Markdown(
                "Fish Audio S2 Pro · PyTorch + universal s2.cpp · reusable Voice Library · dataset preparation · Fast-AR LoRA training",
                elem_classes="tab-subtitle",
            )
        with gr.Column(scale=3, min_width=480):
            with gr.Row():
                unload_btn=gr.Button("🧹 Unload All Models / Free RAM - VRAM", variant="secondary", size="sm")
                clear_outputs_btn=gr.Button("🗑️ Clear Outputs", variant="secondary", size="sm")
    hidden_status=gr.State("")
    unload_btn.click(_clear_vram, outputs=hidden_status, queue=False)
    clear_outputs_btn.click(_clear_outputs, outputs=hidden_status, queue=False)

    with gr.Tabs(elem_classes="tabs"):
        # ------------------------------------------------------------------
        # PREP SAMPLES
        # ------------------------------------------------------------------
        with gr.Tab("🎙️ Prep Samples"):
            gr.Markdown("*Create reusable reference voices for Single Inference, Dialogue Builder and training evaluation.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""### Workflow
1. **Load or record a reference** — use a clean, dry voice clip; roughly 10–30 seconds is a practical starting point.
2. **Reference Transcript** — enter exactly what is spoken. Fish S2 derives language automatically; no synthesis-language selector is required.
3. **Faster-Whisper (optional)** — use it only to bootstrap the transcript. Language detection is always automatic; review the transcript before saving.
4. **Voice Name → Save Voice** — stores audio + transcript in the shared Voice Library. Choosing **None** clears the saved selection without deleting anything.
5. **↻ / 🗑️** — refresh the library after external file changes or delete the currently selected saved voice.

### What happens internally
Fish handles the audio preprocessing required by inference/training. The old manual **Normalize** and **Convert to Mono** actions are intentionally removed. Saving a voice does not train or embed anything; it simply creates a reusable reference pair.""")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row(elem_classes="voice-library-row"):
                        prep_voice=gr.Dropdown(_voice_init, value=NONE, label="Voice Library", scale=8,
                                               info="Load a saved reference pair. Select None to clear the current library selection.")
                        prep_refresh=gr.Button("↻", size="sm", variant="secondary")
                        prep_delete=gr.Button("🗑️", size="sm", variant="stop")
                    prep_audio=gr.Audio(type="filepath", label="Reference Audio", sources=["upload","microphone"], elem_classes="audio-safe-space")
                    prep_text=gr.Textbox(label="Reference Transcript", lines=5,
                                         info="Exact transcript of the reference clip; used for voice cloning/prompt conditioning.")
                    with gr.Accordion("🛰️ Faster-Whisper Transcription", open=False):
                        prep_wm=gr.Dropdown(list(B.WHISPER_MODELS), value="large-v3 (~10 GB VRAM)", label="Whisper Model",
                                            info="ASR model used only to transcribe the reference; larger models are more accurate and use more VRAM.")
                        prep_wb=gr.Slider(1,16,4,1,label="ASR Batch Size",
                                          info="Faster-Whisper batch size. Increase for speed when VRAM allows; lower it if transcription runs out of memory.")
                        with gr.Row():
                            prep_transcribe=gr.Button("Transcribe Reference", variant="secondary", scale=5)
                            prep_transcribe_stop=gr.Button("🛑 Stop", variant="stop", scale=1)
                with gr.Column(scale=1):
                    prep_name=gr.Textbox(label="Voice Name", placeholder="voice_name",
                                         info="Library name used by inference and Dialogue Builder.")
                    prep_save=gr.Button("Save Voice", variant="primary", size="lg")
                    prep_status=gr.Markdown("No saved voice selected.", elem_classes="compact-status")
            prep_voice.change(_sample_payload, prep_voice, [prep_audio,prep_text,prep_status], queue=False)
            prep_refresh.click(_refresh_voice, prep_voice, prep_voice, queue=False)
            prep_delete_event=prep_delete.click(_delete_voice, prep_voice, [prep_voice,prep_audio,prep_text,prep_status], queue=False)
            prep_transcribe_event=prep_transcribe.click(_transcribe, [prep_audio,prep_wm,prep_wb], prep_text)
            prep_transcribe_stop.click(B.stop_aux_job, outputs=hidden_status, cancels=[prep_transcribe_event], queue=False)
            prep_save_event=prep_save.click(_save_voice, [prep_audio,prep_name,prep_text], [prep_status,prep_voice], queue=False)

        # ------------------------------------------------------------------
        # INFERENCE
        # ------------------------------------------------------------------
        with gr.Tab("🔊 Inference"):
            gr.Markdown("*Choose the runtime once, then use the same Voice Library and LoRA adapters in Single Inference or Dialogue Builder.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""### Runtime controls
- **Inference Engine** — `s2.cpp` uses GGUF with the universal CPU/CUDA/Vulkan executable; PyTorch uses the native S2 Pro weights. Switching engines is a hard memory boundary: the previous runtime is unloaded and RAM/VRAM caches are released automatically.
- **Unload All Models / Free RAM - VRAM** — aggressively releases persistent PyTorch model/codec state, compiled wrappers, the managed `s2.cpp` process, CUDA caches, Python garbage, and trims the Windows working set. Active training is never killed by this button.
- **LoRA Adapter** — dynamically mounts a training checkpoint. `None` means the untouched base model. No merged model is created.
- **PyTorch acceleration** — managed automatically. The GUI pins `torch.compile` to **max-autotune-no-cudagraphs** (the stable fast path for this Windows build) and automatically falls back to Default, then eager execution if compilation or a first-run kernel fails. No manual compile selector is exposed.
- **GGUF / Compute Backend / GPU Device / Codec Execution** — C++ only. Auto prefers CUDA on NVIDIA and otherwise Vulkan. `Full GPU` keeps both AR inference and the audio codec on the selected GPU; `Hybrid` keeps AR on GPU but runs the codec on CPU. CPU forces the full pipeline to CPU.

### Generation controls
`Top-P`, `Top-K` and `Temperature` shape sampling diversity; `Repetition Penalty` discourages loops. With **Enable Fixed Seed** OFF, every Generate click receives a new random seed and the used value is shown in **Seed**. Turn **Enable Fixed Seed** ON to keep that visible seed unchanged across generations. If the field is `0` when locking is enabled, the first generation creates one random seed and subsequent generations reuse it; a manually entered seed is likewise preserved.

### Long-form workflow
**Chunking Rule** defaults to `None` for a single uninterrupted synthesis. Enable `Paragraph/Sentence Auto` only for long-form text that benefits from segmentation. **Chunk Gap** inserts configurable silence between generated chunks; it is separate from **Dialogue Silence**, which separates speaker turns.

### Voice conditioning
Reference audio is optional. Leave **Voice Library** and **Reference Audio** empty for Fish S2's native no-reference generation; the timbre is then selected by the model and is not guaranteed to stay voice-identical across different seeds. When cloning a specific voice, use a clean reference and an exact transcript. Fish S2 handles language automatically.""")
            gr.Markdown("### ⚙️ Runtime", elem_classes="section-heading")
            with gr.Row(elem_classes="runtime-row"):
                engine=gr.Dropdown([B.ENGINE_CPP,B.ENGINE_PYTORCH], value=B.ENGINE_CPP, label="Inference Engine", scale=2,
                                   info="C++ = GGUF CPU/CUDA/Vulkan; PyTorch = native S2 Pro inference. Switching engines automatically unloads the previous runtime and frees RAM/VRAM.")
                adapter=gr.Dropdown(_adapter_choices(), value=NONE, label="LoRA Adapter", scale=3,
                                    info="Dynamic checkpoint adapter. None restores the base model without merging weights.")
                adapter_refresh=gr.Button("↻", size="sm", variant="secondary", elem_classes="compact-button")
            with gr.Row(visible=True) as cpp_runtime:
                cpp_model=gr.Dropdown(list(B.GGUF_MODELS), value=list(B.GGUF_MODELS)[1], label="GGUF Model", scale=3,
                                      info="Quantization/precision used by s2.cpp. Auto selects the safest backend chain per quantization; Q2_K remains the smallest/most experimental option.")
                cpp_backend=gr.Dropdown(B.CPP_BACKENDS, value=B.CPP_BACKENDS[0], label="Compute Backend", scale=2,
                                        info="Auto policy: F16/Q8 = CUDA → Vulkan → CPU. K-quants Q2/Q3/Q4/Q5/Q6 = Vulkan → CPU (CUDA is excluded for K-quants in this bundled runtime). If a native backend crashes, Auto remembers it for this app session and advances to the next safe backend. Explicit CUDA/Vulkan/CPU selections are never overridden.")
                cpp_device=gr.Number(0, precision=0, minimum=0, label="GPU Device", scale=1,
                                     info="Zero-based GPU index used by CUDA/Vulkan.")
                codec_policy=gr.Dropdown(["Full GPU (Recommended)", "Hybrid (GPU model + CPU codec)"], value="Full GPU (Recommended)", label="Codec Execution", scale=2,
                                         info="Full GPU keeps reference encoder and waveform decoder on CUDA/Vulkan. Hybrid keeps the AR model on GPU but runs the audio codec on CPU for compatibility.")
            engine.change(_engine_visibility, engine, cpp_runtime, queue=False)
            adapter_refresh.click(_refresh_adapters, adapter, adapter, queue=False)

            with gr.Accordion("🎛️ Generation Settings", open=True):
                with gr.Row():
                    top_p=gr.Slider(.1,1,.7,.05,label="Top-P",info="Nucleus sampling threshold; lower values are more conservative.")
                    top_k=gr.Slider(1,100,30,1,label="Top-K",info="Limits sampling to the K most likely candidates.")
                    temperature=gr.Slider(.1,2,.7,.05,label="Temperature",info="Sampling randomness; higher values increase variation.")
                    repetition=gr.Slider(1,2,1.2,.05,label="Repetition Penalty",info="Penalizes repeated semantic tokens and looping patterns.")
                with gr.Row():
                    seed=gr.Number(0,precision=0,minimum=0,maximum=(1<<31)-1,label="Seed",info="Shows the seed used by the last generation. It is only kept for the next generation when Enable Fixed Seed is ON.")
                    fixed_seed=gr.Checkbox(False,label="Enable Fixed Seed",info="OFF = new random seed on every Generate. ON = preserve and reuse the visible Seed; manual seeds remain untouched.")
                    last_seed=gr.State(0)
                    reuse_seed=gr.Button("Reuse Last Seed",size="sm")
                    random_seed=gr.Button("Random Seed",size="sm")
                    chunk_mode=gr.Dropdown(B.CHUNK_CHOICES, value="None", label="Split / Chunking Rule",
                                           info="Long-form segmentation shared with the newer Easy GUI family workflows.")
                    chunk_gap=gr.Slider(0,2,.35,.05,label="Chunk Gap (s)",info="Silence inserted between long-form chunks; 0 concatenates directly.")
                    dialogue_silence=gr.Slider(0,5,.5,.1,label="Dialogue Silence (s)",info="Silence inserted between Dialogue Builder turns only.")
                reuse_seed.click(lambda s:s,last_seed,seed,queue=False)
                random_seed.click(_new_seed_pair,outputs=[seed,last_seed],queue=False)
            with gr.Accordion("❓ Pronunciation / Control Markup Guide", open=False):
                gr.Markdown("S2 Pro uses open-ended natural-language control rather than a small closed tag vocabulary. Useful forms include `[pause]`, `[emphasis]`, `[laughing]`, `[inhale]`, `[singing]`, `[excited]`, `[angry]`, `[whisper]`, `[screaming]`, `[short pause]`, `[volume up]`, `[volume down]`, `[sigh]` and free descriptions such as `[professional broadcast tone]`. Treat these as learned controls rather than guaranteed deterministic commands.")
            with gr.Accordion("🌐 Supported Languages / ISO Codes", open=False):
                gr.Markdown(B.S2_LANGUAGE_MARKDOWN + "\n\n**No language selector is required in this GUI:** S2 Pro infers language from the input/reference context automatically.")

            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row(elem_classes="voice-library-row"):
                                infer_voice=gr.Dropdown(_voice_init,value=NONE,label="Voice Library",scale=8,
                                                        info="Select a reusable reference voice, or None. With None and no uploaded reference, Fish S2 can synthesize without prompt conditioning.")
                                infer_refresh=gr.Button("↻",size="sm",variant="secondary")
                            infer_audio=gr.Audio(type="filepath",label="Reference Audio (Optional)",sources=["upload"],elem_classes="audio-safe-space")
                            infer_ref_text=gr.Textbox(label="Reference Transcript (Required only with reference audio)",lines=4,
                                                      info="Exact transcript of the reference audio. Filled automatically when a saved voice is loaded.")
                            infer_voice_status=gr.Markdown("No saved voice selected.",elem_classes="compact-status")
                            with gr.Accordion("🛰️ Transcribe Uploaded Reference",open=False):
                                infer_wm=gr.Dropdown(list(B.WHISPER_MODELS),value="large-v3 (~10 GB VRAM)",label="Whisper Model",
                                                     info="Optional Faster-Whisper model used only for reference transcription.")
                                infer_wb=gr.Slider(1,16,4,1,label="ASR Batch Size",
                                                  info="Faster-Whisper batch size used only for uploaded-reference transcription.")
                                with gr.Row():
                                    infer_transcribe=gr.Button("Transcribe",variant="secondary",scale=5)
                                    infer_transcribe_stop=gr.Button("🛑 Stop",variant="stop",scale=1)
                        with gr.Column(scale=2):
                            target_text=gr.Textbox(label="Text to Generate",lines=8,
                                                   info="Target text. Language is inferred automatically by Fish S2.")
                            with gr.Row():
                                generate=gr.Button("Generate Audio 🚀",variant="primary",size="lg",scale=5)
                                stop_generate=gr.Button("🛑 Stop",variant="stop",size="lg",interactive=False,scale=1)
                            output_audio=gr.Audio(label="Generated Audio",type="filepath",elem_classes="output-clean")
                            infer_status=gr.Markdown("")
                    infer_voice.change(_sample_payload,infer_voice,[infer_audio,infer_ref_text,infer_voice_status],queue=False)
                    infer_refresh.click(_refresh_voice,infer_voice,infer_voice,queue=False)
                    infer_transcribe_event=infer_transcribe.click(_transcribe,[infer_audio,infer_wm,infer_wb],infer_ref_text)
                    infer_transcribe_stop.click(B.stop_aux_job,outputs=hidden_status,cancels=[infer_transcribe_event],queue=False)
                    _single_start = generate.click(
                        _inference_ui_start, None, [generate, stop_generate, infer_status], queue=False
                    )
                    _single_run = _single_start.then(
                        _run_single,
                        [engine,cpp_model,cpp_backend,cpp_device,codec_policy,adapter,target_text,infer_audio,infer_ref_text,top_p,top_k,temperature,repetition,chunk_mode,chunk_gap,seed,fixed_seed],
                        [output_audio,infer_status,seed,last_seed],
                    )
                    _single_run.then(
                        _inference_ui_finish, infer_status, [generate, stop_generate, infer_status], queue=False
                    )
                    stop_generate.click(
                        _stop_inference_ui, None, [generate, stop_generate, infer_status],
                        cancels=[_single_run], queue=False
                    )

                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("### 💬 Multi-Speaker Dialogue Builder",elem_classes="section-heading")
                    gr.Markdown("Each turn selects a Voice Library reference and its own text. Insert/Clone/Remove follows the same row workflow as the sister GUIs; global generation settings above apply to every turn.", elem_classes="tab-subtitle")
                    with gr.Row(elem_classes="dialogue-toolbar"):
                        dialogue_voice_refresh=gr.Button("↻ Refresh Voice Library",variant="secondary")
                    dialogue_count=gr.State(2); d_groups=[]; d_voices=[]; d_texts=[]; d_insert=[]; d_clone=[]; d_remove=[]
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        with gr.Group(visible=i<2,elem_classes="dialogue-turn-card") as group:
                            gr.Markdown(f"**Turn {i+1}**",elem_classes="dialogue-turn-title")
                            with gr.Row():
                                voice=gr.Dropdown(_voice_init,value=NONE,label="Voice",scale=2,
                                                  info="Saved reference voice for this dialogue turn.")
                                text=gr.Textbox(label="Text",lines=3,scale=6,
                                                info="Text spoken by this turn; Fish detects language automatically.")
                            with gr.Row(elem_classes="dialogue-actions"):
                                ins=gr.Button("＋ Insert",size="sm"); clone=gr.Button("⧉ Clone",size="sm"); rem=gr.Button("− Remove",size="sm")
                        d_groups.append(group); d_voices.append(voice); d_texts.append(text); d_insert.append(ins); d_clone.append(clone); d_remove.append(rem)
                    inputs=[dialogue_count,*d_voices,*d_texts]; outputs=[dialogue_count,*d_voices,*d_texts,*d_groups]
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        d_insert[i].click(lambda count,*vals,_i=i: _dialogue_mutate("insert",_i,count,*vals), inputs, outputs, queue=False)
                        d_clone[i].click(lambda count,*vals,_i=i: _dialogue_mutate("clone",_i,count,*vals), inputs, outputs, queue=False)
                        d_remove[i].click(lambda count,*vals,_i=i: _dialogue_mutate("remove",_i,count,*vals), inputs, outputs, queue=False)
                    dialogue_voice_refresh.click(_dialogue_refresh,d_voices,d_voices,queue=False)
                    with gr.Row():
                        dialogue_generate=gr.Button("Generate Dialogue 🚀",variant="primary",size="lg",scale=5)
                        dialogue_stop=gr.Button("🛑 Stop",variant="stop",size="lg",scale=1)
                    dialogue_output=gr.Audio(label="Dialogue Output",type="filepath",elem_classes="output-clean")
                    dialogue_status=gr.Markdown("")
                    _dialogue_start=dialogue_generate.click(_inference_ui_start,None,[dialogue_generate,dialogue_stop,dialogue_status],queue=False)
                    _dialogue_run=_dialogue_start.then(_run_dialogue,[engine,cpp_model,cpp_backend,cpp_device,codec_policy,adapter,top_p,top_k,temperature,repetition,chunk_mode,chunk_gap,seed,fixed_seed,dialogue_count,dialogue_silence,*d_voices,*d_texts],[dialogue_output,dialogue_status,seed,last_seed])
                    _dialogue_run.then(_inference_ui_finish,dialogue_status,[dialogue_generate,dialogue_stop,dialogue_status],queue=False)
                    dialogue_stop.click(_stop_inference_ui,None,[dialogue_generate,dialogue_stop,dialogue_status],cancels=[_dialogue_run],queue=False)

        # ------------------------------------------------------------------
        # DATASET PREPARATION
        # ------------------------------------------------------------------
        with gr.Tab("📂 Dataset Preparation"):
            gr.Markdown("*Prepare the project-local Fish training dataset while keeping Dataset and Training UI state independently editable.*",elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide",open=False):
                gr.Markdown("""### Project workflow
1. **Create Project** once. The same project name immediately becomes available in Training, but selecting/editing it here does not force Training to switch projects.
2. **Dataset Name** is the folder created under `datasets/`. **Source Audio Folder** is your raw input folder. If matching `name.wav` + `name.txt`/`name.lab` pairs already exist, their transcripts are reused directly and Faster-Whisper is skipped for those files.
3. **Whisper Model / ASR Batch Size** control transcription only. Faster-Whisper language detection is automatic.
4. **Build / Transcribe Dataset** converts supported audio to Fish's training preparation format and writes transcripts/metadata.
5. **Save Project** persists only Dataset settings. Training hyperparameters/evaluation reference for the same logical project are preserved independently.
6. **Clone Project** creates the next numeric name automatically (`project-2`, `project-3`, ...). **🗑️** removes only the Dataset surface unless its Training surface was already removed.

After a successful build, the Training **Dataset Folder** dropdown is refreshed automatically. Training itself smart-skips VQ/protobuf preprocessing when those artifacts are already newer than the source WAV/LAB files.""")
            with gr.Row(elem_classes="project-strip"):
                dataset_project=gr.Dropdown(_dataset_projects,value=NONE,label="Project",scale=3,
                                            info="Logical project used by Dataset Preparation. Dataset and Training keep separate UI state blocks.")
                dataset_new=gr.Textbox(label="New Project Name",placeholder="my_voice",scale=2,
                                        info="Name for a new logical project. Clones are numbered automatically.")
                dataset_create=gr.Button("Create Project")
                dataset_save=gr.Button("💾 Save Project")
                dataset_clone=gr.Button("Clone Project",variant="secondary")
                dataset_refresh=gr.Button("↻",size="sm",variant="secondary",elem_classes="compact-button")
                dataset_delete=gr.Button("🗑️",size="sm",variant="stop",elem_classes="compact-button")
            dataset_project_status=gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=1):
                    dataset_name=gr.Textbox(label="Dataset Name",placeholder="my_voice",
                                            info="Target folder name created under datasets/. Usually keep it equal to the project name.")
                    with gr.Row():
                        source_folder=gr.Textbox(label="Source Audio Folder",scale=8,
                                                 info="Folder containing raw WAV/MP3/FLAC/M4A/OGG files.")
                        browse=gr.Button("Browse…",scale=1)
                    dataset_wm=gr.Dropdown(list(B.WHISPER_MODELS),value="large-v3 (~10 GB VRAM)",label="Whisper Model",
                                           info="Faster-Whisper model used to create training transcripts.")
                    dataset_batch=gr.Slider(1,32,8,1,label="Transcription Batch Size",
                                            info="Number of audio items Faster-Whisper batches together. Reduce if ASR runs out of VRAM.")
                with gr.Column(scale=1):
                    with gr.Row():
                        dataset_build=gr.Button("Build / Transcribe Dataset",variant="primary",size="lg",scale=5)
                        dataset_stop=gr.Button("🛑 Stop",variant="stop",size="lg",scale=1)
                    dataset_status=gr.Textbox(label="Dataset Status",interactive=False,lines=10,
                                          info="Preparation/transcription summary. Detailed progress is mirrored to CMD and the shared Live Console.")
            dataset_project.change(_load_dataset_project,dataset_project,[dataset_name,source_folder,dataset_wm,dataset_batch],queue=False)
            dataset_refresh.click(lambda:_project_update("dataset"),outputs=dataset_project,queue=False)
            browse.click(_browse_folder,source_folder,source_folder,queue=False)
            dataset_build_event=dataset_build.click(_batch_dataset,[source_folder,dataset_name,dataset_wm,dataset_batch],dataset_status)
            dataset_stop.click(B.stop_aux_job,outputs=dataset_status,cancels=[dataset_build_event],queue=False)

        # ------------------------------------------------------------------
        # LORA TRAINING
        # ------------------------------------------------------------------
        with gr.Tab("🚀 LoRA Training"):
            gr.Markdown("*Project-aware S2-Pro Fast-AR LoRA training with optimizer-step control, AutoTune, true-holdout validation, prompted evaluation, resume and isolated TensorBoard.*",elem_classes="tab-subtitle")
            gr.Markdown("""### ⚠️ Experimental Training
**LoRA training in this Easy GUI is experimental.** Fish Audio S2 Pro already performs very strong zero-shot voice cloning from a short reference audio + exact transcript, so fine-tuning can take substantial time without producing an important improvement over the untouched base model. Small or imperfect datasets can also overfit or degrade pronunciation, prosody, accent, stability or generalization.

Use training only when you have a specific adaptation goal and compare intermediate checkpoints against **Base Model + the same reference prompt**. More steps are not automatically better.

**Hardware requirement:** training is intended only for NVIDIA GPUs with **24 GB VRAM or more**. The 24 GB preset is the minimum supported training profile in this GUI; 32 GB+ and 48 GB+ presets provide additional headroom.""", elem_classes="compact-status")
            with gr.Accordion("📖 Quick Guide",open=False):
                gr.Markdown("""### 1 · Select project + dataset
Choose the same logical project prepared in Dataset. The Dataset folder is synchronized automatically when available, while all Training controls remain a separate saved state.

### 2 · AutoTune or tune manually
**AutoTune** reads clip count/duration and the selected hardware profile. Fish S2's native fine-tune dataset is an *iterable augmented stream* and training is controlled directly by `trainer.max_steps`. **Optimizer steps are the only training-length unit in this GUI.**

- **Training Steps** — exact optimizer-update target.
- **Save Every (Steps)** — checkpoint cadence in optimizer steps.
- **Evaluation Every** — optimizer-step cadence for quantitative validation on the true holdout. When Evaluation Zone is enabled, the prompted monitoring audio uses the same cadence.
- **Rank / Alpha / Dropout** — Fast-AR LoRA capacity, scaling and regularization.
- **Learning Rate / Scheduler** — AutoTune applies the validated Fast-AR values: learning rate `1e-5` with cosine decay to a 10% LR floor. This is the single supported scheduler for reproducible Fast-AR training.
- **Training Seed** — records a reproducible random seed for Fish/Lightning while keeping strict deterministic CUDA mode off for normal performance.
- **Resume Training From** — continue from a project checkpoint, or `Fresh / None` for a new optimizer run.

### 3 · Prompted Holdout Evaluation
Set one fixed evaluation sentence before training. Easy GUI deterministically selects one fixed reference utterance from the validation holdout (or, only when no holdout exists, a deterministic dataset fallback) and reuses that **same audio + exact transcript + seed** at every evaluation step. This matches Fish S2's normal prompt-conditioned voice-cloning workflow and makes checkpoint-to-checkpoint timbre/style evolution directly comparable.

### 4 · Monitor / stop
The modern progress card shows percentage, optimizer steps, loss, elapsed time and ETA. **Stop Training** sends a graceful interrupt so existing checkpoints remain. TensorBoard is project-isolated; switching Training projects detaches any viewer from the old logdir.

### 5 · Inference
Created checkpoints appear directly in **Inference → LoRA Adapter**. Fast-AR-only checkpoints can be used by both PyTorch and the native C++ `.s2lora` path.""")
            with gr.Row(elem_classes="project-strip"):
                train_project=gr.Dropdown(_training_projects,value=NONE,label="Project",scale=3,
                                          info="Training-side project selection. Its saved state is independent from Dataset-side controls.")
                train_save=gr.Button("💾 Save Project")
                train_clone=gr.Button("Clone Project",variant="secondary")
                train_refresh=gr.Button("↻",size="sm",variant="secondary",elem_classes="compact-button")
                train_delete=gr.Button("🗑️",size="sm",variant="stop",elem_classes="compact-button")
            train_project_status=gr.Markdown("")


            with gr.Row(equal_height=False):
                training_dataset=gr.Dropdown(_datasets,value=(_datasets[0] if _datasets and _datasets[0] != "(No datasets)" else None),label="Dataset Folder",scale=4,
                                             info="Prepared dataset under datasets/. The selected project's Dataset state is used when available; changing this does not modify Dataset-tab controls.")
                vram=gr.Dropdown(["24GB VRAM","32GB+ VRAM","48GB+ VRAM"],value="24GB VRAM",label="Hardware Preset",scale=2,
                                 info="Sets the Fish micro-batch, gradient accumulation and worker profile used to calculate effective batch and AutoTune's optimizer-step budget.")
                autotune=gr.Button("⚙ AutoTune",variant="secondary",scale=1)
                datasets_refresh=gr.Button("↻",variant="secondary",size="sm",scale=0,min_width=52,elem_classes="compact-button")
            dataset_info=gr.Markdown("Select **Hardware Preset**, then press **AutoTune** to analyze the dataset and calculate the Fast-AR optimizer-step target, checkpoint cadence and hyperparameters. Holdout validation uses the evaluation cadence; Evaluation Zone optionally adds prompted monitoring audio at that same cadence.",elem_classes="compact-status")

            gr.Markdown("### ↩️ Training Resume", elem_classes="section-heading")
            with gr.Row():
                resume=gr.Dropdown(["Fresh / None"],value="Fresh / None",label="Resume Checkpoint",scale=4,
                                   info="Fresh / None starts a clean optimizer run. Select a checkpoint from this project to continue model and optimizer state.")
                resume_refresh=gr.Button("↻",size="sm",variant="secondary",scale=0,min_width=52,elem_classes="compact-button")
            resume_status=gr.Markdown("Fresh / None = start a new training run.",elem_classes="compact-status")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1,min_width=420):
                    gr.Markdown("### 📦 Training Setup",elem_classes="section-heading")
                    steps=gr.Slider(50,20000,250,50,label="Training Steps",
                                    info="Exact optimizer updates after gradient accumulation. This is the native training-length unit.")
                    save_steps=gr.Slider(10,5000,50,10,label="Save Every (Steps)",
                                         info="Checkpoint cadence in optimizer steps; clamped to the final training target.")
                    eval_steps=gr.Slider(10,5000,50,10,label="Evaluate Every (Steps)",
                                         info="Run true-holdout validation every N optimizer steps; when Evaluation Zone is enabled, also generate the fixed prompted Eval utterance.")
                    training_seed=gr.Number(1234,precision=0,label="Training Seed",
                                            info="Seed passed to Fish/Lightning for reproducibility. Strict deterministic CUDA mode remains off for normal speed.")
                    gr.Markdown("**Managed internally:** single-GPU S2-Pro Fast-AR LoRA · step-based checkpoint cadence · step-based evaluation cadence · project-local logs/TensorBoard.",elem_classes="compact-status")

                with gr.Column(scale=1,min_width=420):
                    gr.Markdown("### ⚙️ LoRA Hyperparameters",elem_classes="section-heading")
                    with gr.Row():
                        rank=gr.Slider(4,64,32,4,label="LoRA Rank",info="Fast-AR adapter capacity. Higher values increase trainable parameters, VRAM use and overfitting risk.")
                        alpha=gr.Slider(4,128,16,4,label="LoRA Alpha",info="LoRA scaling numerator; effective adapter scale is alpha divided by rank.")
                    with gr.Row():
                        dropout=gr.Slider(0.0,0.30,0.10,0.01,label="LoRA Dropout",info="Regularization inside Fast-AR LoRA layers. Small datasets usually benefit from more dropout.")
                        lr=gr.Number(1e-5,label="Learning Rate",info="AdamW learning rate; validated Fast-AR recipe uses 1e-5.")
                    with gr.Accordion("🧪 Advanced Training Hyperparameters",open=False):
                        gr.Markdown("Real Fish/Lightning controls. **AutoTune overwrites these with the validated Fast-AR values.**",elem_classes="compact-status")
                        with gr.Row():
                            scheduler=gr.Dropdown(["Cosine 10% floor"],value="Cosine 10% floor",label="LR Scheduler",
                                                   info="Fixed validated Fast-AR schedule: cosine decay with a 10% learning-rate floor.")
                        with gr.Row():
                            weight_decay=gr.Number(0.01,label="Weight Decay")
                            gradient_clip=gr.Number(1.0,label="Gradient Clip Norm")
                        with gr.Row():
                            max_length=gr.Slider(512,4096,4096,256,label="Max Sequence Length")
                            interactive_prob=gr.Slider(0.0,1.0,1.0,0.05,label="Interactive / Prompt Probability")
                        causal=gr.Checkbox(False,label="Causal Dataset Sampling",
                                           info="OFF is recommended for speaker datasets for random mixing; ON may over-represent contiguous recording sessions.")


            eval_enabled=gr.Checkbox(False,label="Enable Evaluation Zone",
                                     info="Off disables only scheduled evaluation audio. Quantitative validation on the true holdout still runs at Evaluate Every (Steps).")
            with gr.Accordion("🎧 Prompted Holdout Evaluation",open=False,visible=False) as eval_zone:
                gr.Markdown("At **Evaluate Every (Steps)**, Easy GUI synthesizes the same fixed text with the **same deterministic reference utterance from the dataset holdout** and the same seed. This is the appropriate monitor for voice/timbre evolution with Fish S2's default prompt-conditioned fine-tuning behavior. WAVs are saved under the project and logged to TensorBoard.",elem_classes="compact-status")
                eval_text=gr.Textbox(label="Fixed Eval Text",lines=4,value="This is a fixed evaluation sample generated during training.",
                                     info="Keep this sentence unchanged during a run so checkpoints are directly comparable.")
                eval_status=gr.Markdown("Reference prompt is selected automatically and deterministically from the validation holdout.",elem_classes="compact-status")
            eval_enabled.change(lambda x: gr.update(visible=bool(x)),eval_enabled,eval_zone,queue=False)
            with gr.Row(elem_classes="global-toolbar"):
                train_start=gr.Button("🚀 Start Training",variant="primary")
                train_stop=gr.Button("🛑 Stop Training",variant="stop",interactive=False)
                tensorboard=gr.Button("📊 TensorBoard",variant="secondary")
                tensorboard_reload=gr.Button("↻ Reload TensorBoard",variant="secondary")
            train_progress=gr.HTML(value=training_progress_html())
            train_timer=gr.Timer(1.0)

            datasets_refresh.click(lambda:gr.update(choices=B.get_dataset_choices()),outputs=training_dataset,queue=False)
            autotune.click(_autotune_ui,[training_dataset,vram],[steps,save_steps,eval_steps,rank,alpha,dropout,lr,weight_decay,max_length,causal,interactive_prob,gradient_clip,scheduler,
                            training_seed,dataset_info],queue=False)
            train_refresh.click(lambda:_project_update("training"),outputs=train_project,queue=False)
            resume_refresh.click(_checkpoint_update,[train_project,resume],resume,queue=False).then(_refresh_adapters,adapter,adapter,queue=False)
            train_project.change(_load_training_project,train_project,[training_dataset,vram,steps,lr,rank,alpha,dropout,training_seed,save_steps,eval_steps,resume,eval_enabled,eval_text,weight_decay,max_length,causal,interactive_prob,gradient_clip,scheduler],queue=False).then(_prepare_training_project,[train_project,resume],resume,queue=False).then(_refresh_adapters,adapter,adapter,queue=False).then(lambda x: gr.update(visible=bool(x)),eval_enabled,eval_zone,queue=False)
            train_start.click(B.handle_lora_unified,[training_dataset,train_project,steps,lr,vram,rank,alpha,dropout,training_seed,save_steps,eval_steps,resume,eval_text,eval_enabled,weight_decay,max_length,causal,interactive_prob,gradient_clip,scheduler],hidden_status)
            train_stop.click(
                _stop_training,
                outputs=hidden_status,
                queue=False,
            ).then(
                lambda: (gr.update(interactive=True), gr.update(interactive=False)),
                outputs=[train_start, train_stop],
                queue=False,
            )
            tensorboard.click(B.launch_tensorboard_handler,train_project,hidden_status,queue=False)
            tensorboard_reload.click(B.restart_tensorboard_handler,train_project,hidden_status,queue=False)
            train_timer.tick(training_poll_ui,adapter,[hidden_status,train_progress,train_start,train_stop,adapter],queue=False,show_progress="hidden")

    # ----------------------------------------------------------------------
    # Cross-tab synchronization.  Lists synchronize; active selections do not.
    # ----------------------------------------------------------------------
    dataset_create.click(_create_project,[dataset_new,dataset_name,source_folder,dataset_wm,dataset_batch,train_project],[dataset_project,train_project,dataset_name,dataset_project_status],queue=False)
    dataset_save.click(_save_dataset_project,[dataset_project,dataset_name,source_folder,dataset_wm,dataset_batch,train_project],[train_project,dataset_project_status],queue=False)
    dataset_clone.click(lambda p,t: _clone_project_ui(p,t,"dataset"),[dataset_project,train_project],[dataset_project,train_project,dataset_name,dataset_project_status],queue=False)
    dataset_delete.click(_delete_dataset_project,[dataset_project,train_project],[dataset_project,train_project,dataset_project_status],queue=False)

    train_save.click(_save_training_project,[train_project,training_dataset,vram,steps,lr,rank,alpha,dropout,training_seed,save_steps,eval_steps,resume,eval_enabled,eval_text,weight_decay,max_length,causal,interactive_prob,gradient_clip,scheduler,dataset_project],[dataset_project,train_project_status],queue=False)
    train_clone.click(_clone_training_project_ui,[train_project,dataset_project],[dataset_project,train_project,train_project_status],queue=False)
    train_delete.click(_delete_training_project,[train_project,dataset_project],[train_project,dataset_project,train_project_status],queue=False)
    dataset_build_event.then(lambda name: gr.update(choices=B.get_dataset_choices(), value=name if name in B.get_dataset_choices() else None),dataset_name,training_dataset,queue=False)
    prep_save_event.then(lambda: [_refresh_voice(NONE), *[gr.update(choices=_voice_choices(), value=NONE) for _ in range(MAX_DIALOGUE_SEGMENTS)]], outputs=[infer_voice,*d_voices],queue=False)
    prep_delete_event.then(lambda: [_refresh_voice(NONE), *[gr.update(choices=_voice_choices(), value=NONE) for _ in range(MAX_DIALOGUE_SEGMENTS)]], outputs=[infer_voice,*d_voices],queue=False)

    # One shared, self-contained console below every tab — same renderer as CosyVoice3.
    with gr.Accordion("🖥️ Console", open=True, elem_classes="console-accordion"):
        console=gr.HTML(value=html_view(), elem_id="fish-s2-console")
        console_timer=gr.Timer(0.5)
        console_timer.tick(html_view,outputs=console,queue=False,show_progress="hidden")

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
