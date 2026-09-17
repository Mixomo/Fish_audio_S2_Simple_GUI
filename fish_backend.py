import gradio as gr
import importlib.util
import os
import re
import random
import signal
import subprocess
import time
import warnings
import contextlib
import io
import codecs
import math
import threading
from pathlib import Path

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        message=(
            "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. "
            "Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead."
        ),
        category=StarletteDeprecationWarning,
    )
except ImportError:
    pass

# --- PERSISTENT CACHE CONFIGURATION (Must be set BEFORE importing torch) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
COMPILE_CACHE_DIR = os.path.join(MODELS_DIR, ".cache")
os.makedirs(COMPILE_CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = COMPILE_CACHE_DIR
os.environ["TRITON_CACHE_DIR"] = COMPILE_CACHE_DIR
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

import torch
import shutil
import requests
import json
import io
import torchaudio
import soundfile as sf
import numpy as np
import glob
import yaml
try:
    import winsound
except ImportError:
    winsound = None
import sys
from fish_easy.console import log

warnings.filterwarnings(
    "ignore",
    message=r"Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*",
    category=UserWarning,
    module=r"torch\._inductor\.runtime\.triton_helpers",
)
warnings.filterwarnings("ignore", message=r".*torch\.nn\.utils\.weight_norm.*deprecated.*", category=FutureWarning)
# Triton may emit the same warning from generated modules, so also filter by message globally.
warnings.filterwarnings("ignore", message=r".*Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)
try:
    from loguru import logger as _fish_loguru
    _fish_loguru.disable("fish_speech")
except Exception:
    _fish_loguru = None

import threading as _aux_threading
_aux_cancel_event = _aux_threading.Event()

def begin_aux_job():
    _aux_cancel_event.clear()

def stop_aux_job():
    _aux_cancel_event.set()
    return "Stop requested. The active ASR/dataset operation will stop at the next safe boundary."

def _aux_cancelled():
    return _aux_cancel_event.is_set()

# s2.cpp executable discovery
def _configured_s2_executable():
    """Return an optional s2.exe path from config.py without requiring it."""
    config_path = os.path.join(ROOT_DIR, "config.py")
    if not os.path.isfile(config_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location("_fish_s2_config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except Exception as e:
        print(f"[WARNING] Could not load {config_path}: {e}")
        return None

    for name in (
        "S2_EXECUTABLE",
        "S2_EXECUTABLE_PATH",
        "S2_EXE_PATH",
        "S2_PATH",
        "S2_EXEC",
        "CPP_EXEC",
    ):
        value = getattr(config, name, None)
        if value:
            return os.fspath(value)
    return None


def get_s2_executable_candidates():
    """Return absolute s2.exe candidates in lookup priority order."""
    candidates = []
    configured_path = _configured_s2_executable()
    if configured_path:
        configured_path = os.path.expandvars(os.path.expanduser(configured_path))
        if not os.path.isabs(configured_path):
            configured_path = os.path.join(ROOT_DIR, configured_path)
        candidates.append(configured_path)

    s2_root = os.path.join(ROOT_DIR, "modules", "s2.cpp")
    candidates.extend([
        # Easy GUI installer publishes the validated universal runtime here.
        os.path.join(s2_root, "build", "bin", "s2.exe"),
        os.path.join(s2_root, "build", "bin", "Release", "s2.exe"),
        os.path.join(s2_root, "build", "Release", "s2.exe"),
        os.path.join(s2_root, "build", "s2.exe"),
        os.path.join(s2_root, "s2.exe"),
    ])

    # Keep the error list readable if config.py duplicates a standard path.
    return list(dict.fromkeys(os.path.abspath(path) for path in candidates))


def find_s2_executable():
    """Return (executable, checked_paths), where executable may be None."""
    checked_paths = get_s2_executable_candidates()
    executable = next((path for path in checked_paths if os.path.isfile(path)), None)
    return executable, checked_paths


def format_s2_not_found_error(checked_paths):
    checked = "\n".join(f"  - {path}" for path in checked_paths)
    return (
        "s2.exe was not found, so the C++ synthesis engine cannot start.\n\n"
        f"Locations checked:\n{checked}\n\n"
        "Rerun install.bat. The installer validates MSVC, Windows SDK, CUDA, "
        "Vulkan, and the universal CPU/CUDA/Vulkan s2.cpp runtime."
    )


# Audio Chime Path
CHIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "inference_training_done.wav")

def play_done_chime():
    if os.path.exists(CHIME_PATH):
        try:
            if winsound is not None:
                winsound.PlaySound(CHIME_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Failed to play chime: {e}")
    else:
        # Fallback to system beep if file is missing
        if winsound is not None:
            winsound.MessageBeep()


def normalize_seed(seed_value):
    if seed_value is None or seed_value == "":
        return None
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return None
    if seed == 0:
        return None
    if seed < 0:
        seed = abs(seed)
    if seed > (1 << 31):
        seed = 1 << 31
    return seed


def resolve_seed_for_generation(seed_value):
    normalized_seed = normalize_seed(seed_value)
    if normalized_seed is None:
        normalized_seed = random_seed_value()
    return normalized_seed


def random_seed_value():
    return random.randint(1, (1 << 31) - 1)

# Main Paths
TOKENIZER_PATH = os.path.join(ROOT_DIR, "modules", "s2.cpp", "tokenizer.json")

# Performance Optimizations for OpenMP (Threading Affinity)
os.environ["OMP_PROC_BIND"] = "TRUE"
os.environ["OMP_PLACES"] = "CORES"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"

s2_process = None
_inference_cancel_event = threading.Event()
_inference_state_lock = threading.RLock()
s2_current_model = None
s2_current_backend = None
s2_current_device = None
s2_current_codec_cuda = False
s2_current_lora = None
training_process = None

# Auto backend policy remembers native failures for the current app session.
# This prevents Auto from repeatedly selecting a backend that has already
# crashed for the same GGUF model. Restarting the app clears this memory.
_s2_auto_failed_backends = {}

def _s2_quant_family(filename):
    name = str(filename or "").lower()
    if "f16" in name:
        return "F16"
    if "q8_0" in name:
        return "Q8_0"
    for tag, label in (
        ("q6_k", "Q6_K"),
        ("q5_k", "Q5_K_M"),
        ("q4_k", "Q4_K_M"),
        ("q3_k", "Q3_K"),
        ("q2_k", "Q2_K"),
    ):
        if tag in name:
            return label
    return "Unknown"

def _s2_auto_backend_candidates(filename, cpp_exec):
    """Return the safe ordered backend chain for the selected GGUF."""
    family = _s2_quant_family(filename)
    s2_bin_dir = Path(cpp_exec).resolve().parent
    vulkan_available = (
        (s2_bin_dir / "ggml-vulkan.dll").is_file()
        or (s2_bin_dir / "bin" / "ggml-vulkan.dll").is_file()
    )
    cuda_available = bool(shutil.which("nvidia-smi"))

    # The bundled native runtime's CUDA path is suitable for F16/Q8.
    # K-quants are routed through Vulkan because their quantized embedding
    # lookup path is not safe on this older CUDA backend revision.
    if family in {"F16", "Q8_0"}:
        chain = []
        if cuda_available:
            chain.append("CUDA")
        if vulkan_available:
            chain.append("Vulkan")
        chain.append("CPU")
    else:
        chain = []
        if vulkan_available:
            chain.append("Vulkan")
        chain.append("CPU")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(chain))

def _s2_auto_failure_key(filename):
    return str(filename or "").casefold()

def _s2_auto_mark_failed(filename, backend_name):
    key = _s2_auto_failure_key(filename)
    failed = _s2_auto_failed_backends.setdefault(key, set())
    failed.add(str(backend_name))

def _s2_auto_choose_backend(filename, cpp_exec):
    candidates = _s2_auto_backend_candidates(filename, cpp_exec)
    failed = _s2_auto_failed_backends.get(_s2_auto_failure_key(filename), set())
    remaining = [item for item in candidates if item not in failed]
    return (remaining[0] if remaining else None), candidates, failed

def _s2_auto_has_fallback(filename, cpp_exec, failed_backend):
    candidates = _s2_auto_backend_candidates(filename, cpp_exec)
    failed = set(_s2_auto_failed_backends.get(_s2_auto_failure_key(filename), set()))
    failed.add(str(failed_backend))
    return any(item not in failed for item in candidates)

# CPP server stdout/stderr diagnostics. Keep a rolling tail even after
# startup consumes queue entries, and append the complete native stream to disk.
import queue as _queue
import threading as _threading
from collections import deque as _deque
_s2_log_queue = _queue.Queue()
_s2_drain_thread = None
_s2_log_tail = _deque(maxlen=1200)
_s2_log_lock = _threading.Lock()
_s2_last_command = None
S2_RUNTIME_LOG = os.path.join(ROOT_DIR, "logs", "s2_cpp_runtime.log")
os.makedirs(os.path.dirname(S2_RUNTIME_LOG), exist_ok=True)

def _record_s2_native_line(line):
    text = str(line or "").rstrip("\r\n")
    if not text:
        return
    _s2_log_tail.append(text)
    try:
        with _s2_log_lock:
            if os.path.isfile(S2_RUNTIME_LOG) and os.path.getsize(S2_RUNTIME_LOG) > 8 * 1024 * 1024:
                rotated = S2_RUNTIME_LOG + ".1"
                try:
                    if os.path.isfile(rotated):
                        os.remove(rotated)
                    os.replace(S2_RUNTIME_LOG, rotated)
                except Exception:
                    pass
            with open(S2_RUNTIME_LOG, "a", encoding="utf-8", errors="replace") as logf:
                logf.write(text + "\n")
    except Exception:
        pass

def _s2_exit_description(returncode):
    if returncode is None:
        return "process still running"
    unsigned = int(returncode) & 0xFFFFFFFF
    known = {
        0xC0000409: "STATUS_STACK_BUFFER_OVERRUN / Windows fast-fail (native process aborted)",
        0xC0000005: "STATUS_ACCESS_VIOLATION",
        0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
        0xC0000135: "STATUS_DLL_NOT_FOUND",
        0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
        0xC0000096: "STATUS_PRIVILEGED_INSTRUCTION",
    }
    detail = known.get(unsigned, "native Windows process failure" if unsigned >= 0x80000000 else "normal process exit")
    return f"{returncode} (0x{unsigned:08X}: {detail})"

def _s2_log_tail_text(limit=80):
    lines = list(_s2_log_tail)[-max(1, int(limit)):]
    return "\n".join(lines) if lines else "(no native stdout/stderr captured)"

def _s2_failure_report(original_error, proc=None, context="generation"):
    proc = proc if proc is not None else s2_process
    code = proc.poll() if proc is not None else None
    command = " ".join(map(str, _s2_last_command or [])) or "(command unavailable)"
    return (
        f"s2.cpp native failure during {context}.\n"
        f"Exit: {_s2_exit_description(code)}\n"
        f"Model: {s2_current_model or '(unknown)'}\n"
        f"Backend: {s2_current_backend or '(unknown)'} · Device: {s2_current_device}\n"
        f"Codec: {s2_current_codec_cuda or '(unknown)'}\n"
        f"LoRA: {s2_current_lora or 'None'}\n"
        f"Command: {command}\n\n"
        f"--- Native s2.cpp log tail ---\n{_s2_log_tail_text()}\n"
        f"--- End native log tail ---\n\n"
        f"HTTP/Python error: {original_error}\n"
        f"Full native log: {S2_RUNTIME_LOG}"
    )

def _clear_s2_dead_state():
    global s2_process, s2_current_model, s2_current_backend, s2_current_device
    global s2_current_codec_cuda, s2_current_lora
    s2_process = None
    s2_current_model = None
    s2_current_backend = None
    s2_current_device = None
    s2_current_codec_cuda = None
    s2_current_lora = None

# Fish Python Persistence Cache
fish_python_model = None
fish_python_codec = None
fish_python_decode_one_token = None
fish_python_decode_one_token_base = None
fish_python_checkpoint_dir = None
fish_python_adapter = None
fish_python_compile_choice = None
fish_python_compile_active = "Eager / Disabled"

INTERNAL_TORCH_COMPILE_MODE = "Max Autotune (No CUDA Graphs)"

TORCH_COMPILE_CHOICES = [
    "Auto (Recommended)",
    "Eager / Disabled",
    "Default",
    "Reduce Overhead",
    "Max Autotune (No CUDA Graphs)",
    "Max Autotune",
]
_COMPILE_MODE = {
    "Default": "default",
    "Reduce Overhead": "reduce-overhead",
    "Max Autotune (No CUDA Graphs)": "max-autotune-no-cudagraphs",
    "Max Autotune": "max-autotune",
}

def _compile_fallback_chain(choice):
    choice = choice if choice in TORCH_COMPILE_CHOICES else "Auto (Recommended)"
    if choice == "Auto (Recommended)":
        # Max-autotune is preferred on Windows. PyTorch 2.8 adds
        # max-autotune-no-cudagraphs, which keeps autotuning while avoiding the
        # CUDA Graph launcher path that is fragile on some Windows/Triton builds.
        return ["Max Autotune (No CUDA Graphs)", "Default", "Eager / Disabled"]
    if choice == "Max Autotune":
        return ["Max Autotune", "Max Autotune (No CUDA Graphs)", "Default", "Eager / Disabled"]
    if choice == "Max Autotune (No CUDA Graphs)":
        return ["Max Autotune (No CUDA Graphs)", "Default", "Eager / Disabled"]
    if choice == "Reduce Overhead":
        return ["Reduce Overhead", "Default", "Eager / Disabled"]
    if choice == "Default":
        return ["Default", "Eager / Disabled"]
    return ["Eager / Disabled"]

def _console_progress(label, current, total, detail="", width=26):
    try:
        total = max(1, int(total)); current = max(0, min(int(current), total))
        pct = current / total
        filled = int(round(width * pct))
        bar = "█" * filled + "░" * (width - filled)
        suffix = f" · {detail}" if detail else ""
        print(f"[UI-PROGRESS] {label:<18} [{bar}] {pct*100:6.2f}% ({current}/{total}){suffix}", end="\r", flush=True)
        if current >= total:
            print()
    except Exception:
        pass

def _phase(progress, fraction, message):
    fraction = max(0.0, min(1.0, float(fraction)))
    try:
        progress(fraction, desc=str(message))
    except Exception:
        pass
    # Keep CMD and the embedded console aligned with one compact modern bar.
    _console_progress("Workflow", int(round(fraction * 100)), 100, str(message), width=24)

# Model directories (organized per user request)
FISH_MODELS_DIR = os.path.join(MODELS_DIR, "S2")
S2_CPP_MODELS_DIR = os.path.join(MODELS_DIR, "s2.cpp")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")
CPP_LORA_DIR = os.path.join(MODELS_DIR, "cpp_loras")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
SAMPLE_PREVIEW_DIR = os.path.join(COMPILE_CACHE_DIR, "sample_previews")

for d in [OUTPUTS_DIR, MODELS_DIR, FISH_MODELS_DIR, S2_CPP_MODELS_DIR, SAMPLES_DIR, TRAINED_MODELS_DIR, WHISPER_MODELS_DIR, CPP_LORA_DIR, COMPILE_CACHE_DIR, SAMPLE_PREVIEW_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Startup: Cache Status Report ---
_cache_kernel_count = 0
try:
    _cache_kernel_count = sum(1 for _, _, files in os.walk(COMPILE_CACHE_DIR) for f in files if f.endswith('.py'))
except Exception:
    pass

print("----------------------------------------------------------------")
if _cache_kernel_count >= 50:
    print(f"[Fish Speech] Persistent cache found at models/.cache ({_cache_kernel_count} compiled kernels).")
    print("[Fish Speech] PyTorch engine will use cached kernels — fast startup expected.")
else:
    print(f"[Fish Speech] No persistent compile cache found (or incomplete: {_cache_kernel_count} kernels).")
    print("[Fish Speech] PyTorch compile is managed automatically: max-autotune-no-cudagraphs -> Default -> Eager fallback.")
print("----------------------------------------------------------------")

HAS_COMPILE_CACHE = (_cache_kernel_count >= 50)

import sys
if os.path.join(ROOT_DIR, "modules", "s2") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "modules", "s2"))

# Training Paths
FS_DIR = os.path.join(ROOT_DIR, "modules", "s2")
TRAINING_DATA_DIR = os.path.join(ROOT_DIR, "datasets")
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

# --- Enums / Configs ---
GGUF_MODELS = {
    "F16 (~9.9 GB) — Highest Fidelity": "s2-pro-f16.gguf",
    "Q8_0 (~5.6 GB) — Near-lossless / Recommended for 10+ GB VRAM": "s2-pro-q8_0.gguf",
    "Q6_K (~4.5 GB) — High Quality": "s2-pro-q6_k.gguf",
    "Q5_K_M (~4.0 GB) — Balanced": "s2-pro-q5_k_m.gguf",
    "Q4_K_M (~3.6 GB) — Compact / Recommended for 6–7 GB VRAM": "s2-pro-q4_k_m.gguf",
    "Q3_K (~3.0 GB) — Small": "s2-pro-q3_k.gguf",
    "Q2_K (~2.6 GB) — Smallest / Experimental": "s2-pro-q2_k.gguf"
}
CPP_BACKENDS = [
    "Auto (CUDA on NVIDIA / Vulkan otherwise)",
    "CUDA (NVIDIA)",
    "Vulkan (GPU)",
    "CPU",
]

# --- Engine Names ---
ENGINE_CPP = "Fish Speech S2 Pro (CPP) (VRAM Efficient - Slower)"
ENGINE_PYTORCH = "Fish Speech S2 Pro (PyTorch) (Fastest - 24GB+ VRAM only)"

WHISPER_LANGS = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "Chinese": "zh",
    "Japanese": "ja",
    "German": "de",
    "French": "fr",
    "Korean": "ko",
    "Russian": "ru",
    "Portuguese": "pt",
    "Turkish": "tr"
}

WHISPER_MODELS = {
    "large-v3 (~10 GB VRAM)": "large-v3",
    "large-v2 (~10 GB VRAM)": "large-v2",
    "medium (~5 GB VRAM)": "medium",
    "small (~2 GB VRAM)": "small",
    "base (~1 GB VRAM)": "base",
    "tiny (~1 GB VRAM)": "tiny"
}


NONE = "None"

# Official S2 Pro model-card language coverage. The model card names Tier 1/2
# explicitly and publishes the remaining ISO/language codes as global coverage.
S2_LANGUAGE_NAMES = {
    "ja":"Japanese", "en":"English", "zh":"Chinese", "ko":"Korean", "es":"Spanish",
    "pt":"Portuguese", "ar":"Arabic", "ru":"Russian", "fr":"French", "de":"German",
    "sv":"Swedish", "it":"Italian", "tr":"Turkish", "no":"Norwegian", "nl":"Dutch",
    "cy":"Welsh", "eu":"Basque", "ca":"Catalan", "da":"Danish", "gl":"Galician",
    "ta":"Tamil", "hu":"Hungarian", "fi":"Finnish", "pl":"Polish", "et":"Estonian",
    "hi":"Hindi", "la":"Latin", "ur":"Urdu", "th":"Thai", "vi":"Vietnamese",
    "jw":"Javanese", "bn":"Bengali", "yo":"Yoruba", "sl":"Slovenian", "cs":"Czech",
    "sw":"Swahili", "nn":"Norwegian Nynorsk", "he":"Hebrew", "ms":"Malay", "uk":"Ukrainian",
    "id":"Indonesian", "kk":"Kazakh", "bg":"Bulgarian", "lv":"Latvian", "my":"Burmese",
    "tl":"Tagalog", "sk":"Slovak", "ne":"Nepali", "fa":"Persian", "af":"Afrikaans",
    "el":"Greek", "bo":"Tibetan", "hr":"Croatian", "ro":"Romanian", "sn":"Shona",
    "mi":"Maori", "yi":"Yiddish", "am":"Amharic", "be":"Belarusian", "km":"Khmer",
    "is":"Icelandic", "az":"Azerbaijani", "sd":"Sindhi", "br":"Breton", "sq":"Albanian",
    "ps":"Pashto", "mn":"Mongolian", "ht":"Haitian Creole", "ml":"Malayalam", "sr":"Serbian",
    "sa":"Sanskrit", "te":"Telugu", "ka":"Georgian", "bs":"Bosnian", "pa":"Punjabi",
    "lt":"Lithuanian", "kn":"Kannada", "si":"Sinhala", "hy":"Armenian", "mr":"Marathi",
    "as":"Assamese", "gu":"Gujarati", "fo":"Faroese",
}
S2_LANGUAGE_CHOICES = [("Auto-detect", "auto")] + [(f"{name} ({code})", code) for code, name in S2_LANGUAGE_NAMES.items()]
S2_LANGUAGE_MARKDOWN = "\n".join([
    "**Tier 1:** Japanese (`ja`), English (`en`), Chinese (`zh`)",
    "**Tier 2:** Korean (`ko`), Spanish (`es`), Portuguese (`pt`), Arabic (`ar`), Russian (`ru`), French (`fr`), German (`de`)",
    "**Global coverage published by Fish Audio:** " + ", ".join(f"{S2_LANGUAGE_NAMES[c]} (`{c}`)" for c in S2_LANGUAGE_NAMES if c not in {"ja","en","zh","ko","es","pt","ar","ru","fr","de"}),
    "\nFish Audio describes S2 Pro as supporting 80+ languages; the list above reproduces the concrete codes currently published in the model card.",
])

CHUNK_CHOICES = ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]

def split_long_text(text, mode="None"):
    import re
    text = str(text or "").strip()
    if not text:
        return []
    mode = mode if mode in CHUNK_CHOICES else "None"
    if mode == "None":
        return [text]
    if mode == "Paragraphs":
        return [x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()] or [text]
    if mode == "Lines":
        return [x.strip() for x in text.splitlines() if x.strip()] or [text]
    if mode == "Periods":
        normalized = re.sub(r"\s+", " ", text)
        return [x.strip() for x in re.split(r"(?<=\.)\s+", normalized) if x.strip()] or [text]
    if mode == "Speaker turns":
        chunks, current = [], []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if (re.match(r"^\s*\[?SPEAKER\d+\]?", clean, flags=re.I) or re.match(r"^<\|speaker:\d+\|>", clean)) and current:
                chunks.append("\n".join(current).strip()); current=[clean]
            else:
                current.append(clean)
        if current: chunks.append("\n".join(current).strip())
        return chunks or [text]
    chunks=[]
    paragraphs=[x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()] or [text]
    for paragraph in paragraphs:
        normalized=re.sub(r"[ \t]+", " ", paragraph).strip()
        sentences=[x.strip() for x in re.split(r"(?<=[.!?…。！？;；])\s*", normalized) if x.strip()]
        chunks.extend(sentences or [normalized])
    return chunks or [text]


def _training_project_names():
    root = Path(FS_DIR) / "results"
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()], key=str.casefold)


def _project_uses_fast_lora(project):
    """Return whether a project's saved LoRA topology is Fast-AR-only.

    Projects created by older GUI revisions may still have historical Slow or
    mixed checkpoints under results/. Keep those files intact, but do not
    expose them as current training/resume/inference choices.
    """
    config_path = Path(FS_DIR) / "fish_speech" / "configs" / "lora" / f"run_{project}.yaml"
    if not config_path.is_file():
        # A project without a generated config has no topology to reject yet.
        return True
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        targets = data.get("target_modules")
        return bool(targets) and all(str(target).startswith("fast_") for target in targets)
    except Exception:
        return False


def get_lora_adapter_choices():
    """Fast-AR-only adapter dropdown, with None as a real reset state."""
    choices = [(NONE, NONE)]
    for project in _training_project_names():
        if not _project_uses_fast_lora(project):
            continue
        for ckpt in handle_lora_list_checkpoints(project):
            choices.append((f"{project} / {ckpt}", f"{project}::{ckpt}"))
    return choices


def resolve_lora_checkpoint(choice):
    if not choice or str(choice).strip().lower() in {"none", "base", "base model"}:
        return None, None
    value = str(choice)
    if "::" not in value:
        return None, None
    project, ckpt_value = value.split("::", 1)
    project = project.strip()
    ckpt_value = ckpt_value.strip()
    if not project or not ckpt_value:
        return None, None

    project_dir = Path(FS_DIR) / "results" / project
    checkpoints_dir = project_dir / "checkpoints"

    candidate = Path(ckpt_value)
    if candidate.is_absolute() and candidate.is_file():
        return project, candidate.resolve()

    direct = checkpoints_dir / Path(ckpt_value.replace("\\", "/"))
    if direct.is_file():
        return project, direct.resolve()

    # Compatibility with values produced by older GUI versions that only
    # stored a basename.
    basename = Path(ckpt_value).name
    matches = [p for p in checkpoints_dir.rglob(basename) if p.is_file()] if checkpoints_dir.is_dir() else []
    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        return project, matches[0].resolve()
    return None, None


def _extract_verified_lora_state(model, checkpoint_payload, checkpoint_path):
    """Map Lightning checkpoint LoRA tensors onto the instantiated Fish model.

    The old code used strict=False without checking the result, which can
    silently load zero tensors. This routine only accepts a checkpoint when
    every LoRA tensor expected by the configured model is present, shape
    compatible and actually mapped.
    """
    raw_state = (
        checkpoint_payload.get("state_dict", checkpoint_payload)
        if isinstance(checkpoint_payload, dict)
        else checkpoint_payload
    )
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"LoRA checkpoint has no state_dict: {checkpoint_path}")

    model_state = model.state_dict()
    expected = {
        key: tensor for key, tensor in model_state.items()
        if "lora_" in key.lower()
    }
    if not expected:
        raise RuntimeError("The instantiated model contains no LoRA parameters.")

    normalized = {}
    unmapped = []
    shape_errors = []

    for raw_key, value in raw_state.items():
        if "lora_" not in str(raw_key).lower() or not torch.is_tensor(value):
            continue

        key = str(raw_key)
        candidates = [key]

        # Lightning wraps TextToSemantic.model as `model.*`. Other wrappers
        # may prepend module/_forward_module/model.model.
        stripped = key
        for prefix in ("_forward_module.", "module.", "model."):
            while stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                candidates.append(stripped)

        # Suffix matching is safe here because Fish LoRA parameter paths are
        # unique within the instantiated Fast-AR model.
        suffix_hits = [target for target in expected if key.endswith(target)]
        if len(suffix_hits) == 1:
            candidates.append(suffix_hits[0])

        mapped = next((candidate for candidate in candidates if candidate in expected), None)
        if mapped is None:
            unmapped.append(key)
            continue

        if tuple(value.shape) != tuple(expected[mapped].shape):
            shape_errors.append(
                f"{key} -> {mapped}: checkpoint {tuple(value.shape)} != model {tuple(expected[mapped].shape)}"
            )
            continue
        normalized[mapped] = value

    if shape_errors:
        raise RuntimeError(
            "LoRA checkpoint/model shape mismatch:\n  " + "\n  ".join(shape_errors[:12])
        )
    if not normalized:
        raise RuntimeError(
            f"No LoRA tensors from '{checkpoint_path}' matched the Fish model. "
            "Refusing to run inference with an apparently loaded adapter."
        )

    missing = sorted(set(expected) - set(normalized))
    if missing:
        raise RuntimeError(
            f"Incomplete LoRA checkpoint: loaded {len(normalized)}/{len(expected)} expected tensors. "
            f"Missing: {', '.join(missing[:12])}"
        )

    return normalized, len(expected), unmapped


def _lora_checkpoint_identity(path):
    """Cache key that changes when last.ckpt is overwritten in-place."""
    if path is None:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    return f"{path}::{stat.st_mtime_ns}::{stat.st_size}"


def _project_lora_config(project):
    import yaml
    from fish_speech.models.text2semantic.lora import LoraConfig
    path = Path(FS_DIR) / "fish_speech" / "configs" / "lora" / f"run_{project}.yaml"
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return LoraConfig(
        r=int(data.get("r", 32)),
        lora_alpha=int(data.get("lora_alpha", 16)),
        lora_dropout=float(data.get("lora_dropout", 0.0)),
        target_modules=list(data.get("target_modules", ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"])),
        r_slow=int(data.get("r_slow", 0)),
        alpha_slow=float(data.get("alpha_slow", 0.0)),
        r_fast=int(data.get("r_fast", 0)),
        alpha_fast=float(data.get("alpha_fast", 0.0)),
        freeze_slow=bool(data.get("freeze_slow", False)),
        freeze_fast=bool(data.get("freeze_fast", False)),
    )


def ensure_cpp_lora_adapter(choice):
    """Materialize the native adapter transparently; no explicit Export workflow."""
    project, ckpt = resolve_lora_checkpoint(choice)
    if ckpt is None:
        return None
    cfg = _project_lora_config(project)
    out_dir = Path(CPP_LORA_DIR) / project
    out_path = out_dir / f"{ckpt.stem}.s2lora"
    meta_path = out_dir / f"{ckpt.stem}.json"
    if (not out_path.is_file()) or out_path.stat().st_mtime_ns < ckpt.stat().st_mtime_ns:
        from fish_easy.cpp_lora import export_s2lora
        export_s2lora(ckpt, out_path, int(cfg.r), float(cfg.lora_alpha), meta_path)
        print(f"[LoRA] Cached native C++ adapter: {out_path}")
    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise RuntimeError(f"C++ LoRA adapter materialization failed: {out_path}")
    print(f"[C++] LoRA VERIFIED: {project}/{ckpt.name} -> {out_path.name}")
    return str(out_path.resolve())


def get_cpp_lora_choices():
    """Return display values for native C++ Fast-AR adapters."""
    choices = ["None"]
    if not os.path.isdir(CPP_LORA_DIR):
        return choices
    for path in sorted(Path(CPP_LORA_DIR).rglob("*.s2lora")):
        try:
            choices.append(str(path.relative_to(CPP_LORA_DIR)).replace("\\", "/"))
        except ValueError:
            choices.append(str(path))
    return choices

def resolve_cpp_lora(choice):
    if not choice or str(choice).strip().lower() in {"none", "base", "base model"}:
        return None
    path = Path(str(choice))
    if not path.is_absolute():
        path = Path(CPP_LORA_DIR) / path
    return str(path.resolve()) if path.is_file() else None

def get_sample_choices():
    if not os.path.exists(SAMPLES_DIR): return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
    return sorted([f.replace(".wav", "") for f in files])

def get_sample_preview_path(audio_path):
    """Create a browser-friendly dual-mono preview without modifying the source."""
    if not audio_path or not os.path.isfile(audio_path):
        return audio_path

    source_stat = os.stat(audio_path)
    source_key = f"{source_stat.st_mtime_ns}_{source_stat.st_size}"
    sample_stem = os.path.splitext(os.path.basename(audio_path))[0]
    preview_path = os.path.join(SAMPLE_PREVIEW_DIR, f"{sample_stem}_{source_key}.wav")
    if os.path.isfile(preview_path):
        return preview_path

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        mono = audio[:, 0] if audio.shape[1] == 1 else np.mean(audio, axis=1)
        pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        sf.write(
            preview_path,
            np.column_stack((pcm16, pcm16)),
            sample_rate,
            subtype="PCM_16",
        )

        prefix = f"{sample_stem}_"
        for cached_name in os.listdir(SAMPLE_PREVIEW_DIR):
            cached_path = os.path.join(SAMPLE_PREVIEW_DIR, cached_name)
            if cached_name.startswith(prefix) and cached_path != preview_path:
                try:
                    os.remove(cached_path)
                except OSError:
                    pass
        return preview_path
    except Exception as exc:
        print(f"[Sample Preview] Could not create dual-mono preview for {audio_path}: {exc}")
        return audio_path


def _trim_process_ram():
    """Best-effort return of unused Python/CRT pages to Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.CDLL("msvcrt")._heapmin()
        except Exception:
            pass
        try:
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.EmptyWorkingSet(handle)
        except Exception:
            pass
    except Exception:
        pass


def _deep_cuda_cleanup():
    """Best-effort cleanup of allocator, cuBLAS and FFT caches."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        clear_cublas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_cublas is not None:
            clear_cublas()
    except Exception:
        pass
    try:
        torch.backends.cuda.cufft_plan_cache.clear()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def unload_python_engine(reset_compiler=True, trim_ram=False):
    """Release PyTorch model, codec, compiled wrappers, CPU RAM and CUDA caches."""
    global fish_python_model, fish_python_codec
    global fish_python_decode_one_token, fish_python_decode_one_token_base
    global fish_python_checkpoint_dir, fish_python_adapter
    global fish_python_compile_choice, fish_python_compile_active

    fish_python_model = None
    fish_python_codec = None
    fish_python_decode_one_token = None
    fish_python_decode_one_token_base = None
    fish_python_checkpoint_dir = None
    fish_python_adapter = None
    fish_python_compile_choice = None
    fish_python_compile_active = "Eager / Disabled"

    if reset_compiler:
        try:
            torch.compiler.reset()
        except Exception:
            try:
                torch._dynamo.reset()
            except Exception:
                pass

    import gc
    gc.collect()
    gc.collect()
    _deep_cuda_cleanup()
    if trim_ram:
        _trim_process_ram()


def _terminate_cpp_engine():
    """Terminate the managed s2.cpp process tree and clear all runtime identity."""
    global s2_process
    killed = False
    with _inference_state_lock:
        proc = s2_process
        if proc is not None:
            try:
                if proc.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=8,
                            check=False,
                        )
                    if proc.poll() is None:
                        proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    killed = True
            except Exception as exc:
                print(f"[MEMORY] Could not fully terminate s2.cpp: {exc}", flush=True)
        _clear_s2_dead_state()
    return killed


def unload_all_models(reason="manual", reset_compiler=True, trim_ram=True):
    """Aggressively unload persistent inference runtimes and free host/GPU caches.

    The training process runs in a separate child and is deliberately never
    terminated here; starting training calls this function before spawning it.
    """
    global training_process

    _inference_cancel_event.set()
    killed_cpp = _terminate_cpp_engine()
    unload_python_engine(reset_compiler=reset_compiler, trim_ram=False)

    import gc
    gc.collect()
    gc.collect()
    _deep_cuda_cleanup()
    if trim_ram:
        _trim_process_ram()

    training_note = ""
    if training_process is not None and training_process.poll() is None:
        training_note = f" Active training PID {training_process.pid} was not stopped."

    parts = ["PyTorch model/codec + compile state released"]
    parts.append("s2.cpp process terminated" if killed_cpp else "s2.cpp not resident")
    parts.append("CUDA caches cleared")
    parts.append("RAM working set trimmed" if trim_ram else "RAM GC completed")
    message = f"[MEMORY] {reason}: " + " · ".join(parts) + "." + training_note
    print(message, flush=True)
    return message


def switch_inference_engine(engine):
    """Engine changes are hard boundaries: release the previous runtime first."""
    return unload_all_models(
        reason=f"inference engine switch -> {engine}",
        reset_compiler=True,
        trim_ram=True,
    )


def get_dataset_choices():
    if not os.path.exists(TRAINING_DATA_DIR): return ["(No datasets)"]
    subdirs = [d for d in os.listdir(TRAINING_DATA_DIR) if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))]
    return sorted(subdirs) if subdirs else ["(No datasets)"]

def get_trained_models():
    base = ["Base Model (Fish S2 Pro)"]
    trained_dir = TRAINED_MODELS_DIR
    if os.path.exists(trained_dir):
        subdirs = [d for d in os.listdir(trained_dir) if os.path.isdir(os.path.join(trained_dir, d))]
        base += sorted(subdirs)
    return base

def handle_clear_results():
    results_dir = os.path.join(FS_DIR, "results")
    if os.path.exists(results_dir):
        import shutil
        for item in os.listdir(results_dir):
             path = os.path.join(results_dir, item)
             try:
                 if os.path.isdir(path): shutil.rmtree(path)
                 else: os.remove(path)
             except: pass
    return "All intermediate training results (LoRAs) have been cleared."

def get_existing_training_projects():
    results_dir = os.path.join(FS_DIR, "results")
    if not os.path.exists(results_dir): return []
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    return sorted(subdirs)

def load_sample(sample_name):
    """Load sample audio and text. Returns (audio_path_or_None, text_str)."""
    if not sample_name:
        return None, ""
    audio_path = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    txt_path = os.path.join(SAMPLES_DIR, f"{sample_name}.txt")
    json_path = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
    
    text = ""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("Text", "") or data.get("text", "")
        except: pass
    
    if not text and os.path.exists(txt_path):
        for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    text = f.read().strip()
                if text: break
            except: pass
            
    if os.path.exists(audio_path):
        return get_sample_preview_path(audio_path), text
    return None, text

# --- Helper functions ---


@contextlib.contextmanager
def _quiet_pytorch_engine(label="PyTorch"):
    """Hide upstream/tqdm chatter while preserving a compact tail on failure."""
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield
    except Exception:
        tail = []
        for raw in (out.getvalue() + "\n" + err.getvalue()).splitlines():
            line = raw.strip()
            if line and not line.startswith(("0%|", "100%|")):
                tail.append(line)
        if tail:
            print(f"[{label}] Engine detail before failure:")
            for line in tail[-8:]:
                print(f"  {line}")
        raise


def _apply_compile_choice(choice):
    """Compile only the AR token decoder; return the mode actually armed."""
    global fish_python_decode_one_token, fish_python_compile_choice, fish_python_compile_active
    if fish_python_decode_one_token_base is None:
        raise RuntimeError("PyTorch decoder is not initialized.")
    fish_python_compile_choice = choice if choice in TORCH_COMPILE_CHOICES else "Auto (Recommended)"
    last_error = None
    for candidate in _compile_fallback_chain(fish_python_compile_choice):
        if candidate == "Eager / Disabled" or not torch.cuda.is_available():
            fish_python_decode_one_token = fish_python_decode_one_token_base
            fish_python_compile_active = "Eager / Disabled"
            print("[PyTorch] Compile fallback: Eager")
            return fish_python_compile_active
        try:
            mode = _COMPILE_MODE[candidate]
            print(f"[PyTorch] Compile: {candidate}")
            fish_python_decode_one_token = torch.compile(
                fish_python_decode_one_token_base,
                mode=mode,
                fullgraph=True,
            )
            fish_python_compile_active = candidate
            
            return candidate
        except Exception as exc:
            last_error = exc
            print(f"[WARNING] Could not arm {candidate}: {exc}. Trying safer fallback.")
            try:
                torch.compiler.reset()
            except Exception:
                pass
    fish_python_decode_one_token = fish_python_decode_one_token_base
    fish_python_compile_active = "Eager / Disabled"
    if last_error:
        print(f"[WARNING] All compile modes failed to initialize: {last_error}")
    return fish_python_compile_active


def _next_runtime_compile_fallback(active):
    order = {
        "Max Autotune": "Max Autotune (No CUDA Graphs)",
        "Max Autotune (No CUDA Graphs)": "Default",
        "Reduce Overhead": "Default",
        "Default": "Eager / Disabled",
    }
    return order.get(active, "Eager / Disabled")


def generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen,
                         chunk_mode, chunk_gap, adapter_select, compile_choice, seed, progress):
    global fish_python_model, fish_python_codec, fish_python_decode_one_token
    global fish_python_decode_one_token_base, fish_python_checkpoint_dir, fish_python_adapter
    global fish_python_compile_choice, fish_python_compile_active

    normalized_seed = resolve_seed_for_generation(seed)
    if normalized_seed is not None:
        try:
            from fish_speech.utils import set_seed
            set_seed(normalized_seed)
            print(f"[PyTorch] Seed: {normalized_seed}")
        except Exception as exc:
            print(f"[WARNING] Seed setup failed: {exc}")

    def force_seed(target_seed):
        if target_seed is None:
            return
        try:
            from fish_speech.utils import set_seed
            set_seed(target_seed)
        except Exception:
            pass

    target_dir = FISH_MODELS_DIR
    adapter_project, adapter_ckpt = resolve_lora_checkpoint(adapter_select)
    if adapter_select and str(adapter_select).strip().lower() not in {"none", "base", "base model"} and adapter_ckpt is None:
        raise FileNotFoundError(f"Selected LoRA checkpoint no longer exists: {adapter_select}")
    adapter_key = _lora_checkpoint_identity(adapter_ckpt)
    selected_compile = INTERNAL_TORCH_COMPILE_MODE
    model_changed = (fish_python_model is None or fish_python_checkpoint_dir != target_dir or fish_python_adapter != adapter_key)

    if model_changed:
        import gc
        if fish_python_model is not None:
            unload_python_engine()
        _phase(progress, 0.08, f"Loading Fish S2 Pro · adapter: {adapter_select or NONE}")
        from huggingface_hub import snapshot_download
        target_dir = snapshot_download(repo_id="fishaudio/s2-pro", local_dir=FISH_MODELS_DIR)
        fish_python_checkpoint_dir = target_dir

        from fish_speech.models.text2semantic.inference import init_model, decode_one_token_ar
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        _phase(progress, 0.20, "Initializing text-to-semantic model")

        if adapter_ckpt:
            from fish_speech.models.text2semantic.llama import DualARTransformer
            cfg = _project_lora_config(adapter_project)
            fish_python_model = DualARTransformer.from_pretrained(
                fish_python_checkpoint_dir, load_weights=True, lora_config=cfg
            )
            raw = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
            normalized, expected_count, ignored = _extract_verified_lora_state(
                fish_python_model, raw, adapter_ckpt
            )
            load_result = fish_python_model.load_state_dict(normalized, strict=False)
            unexpected_lora = [k for k in load_result.unexpected_keys if "lora_" in k.lower()]
            if unexpected_lora:
                raise RuntimeError(
                    "Unexpected LoRA keys while loading adapter: " + ", ".join(unexpected_lora[:12])
                )

            global_step = raw.get("global_step") if isinstance(raw, dict) else None
            step_text = f" · global_step={global_step}" if global_step is not None else ""
            rel = adapter_ckpt
            try:
                rel = adapter_ckpt.relative_to(Path(FS_DIR) / "results" / adapter_project)
            except ValueError:
                pass
            print(
                f"[PyTorch] LoRA VERIFIED: {adapter_project}/{rel} · "
                f"{len(normalized)}/{expected_count} tensors loaded{step_text}"
            )
            if ignored:
                print(f"[PyTorch] LoRA: ignored {len(ignored)} non-matching LoRA-like key(s).")
            fish_python_model = fish_python_model.to(device=device, dtype=precision).eval()
            fish_python_decode_one_token_base = decode_one_token_ar
        else:
            with _quiet_pytorch_engine("PyTorch model load"):
                fish_python_model, base_decode = init_model(
                    checkpoint_path=fish_python_checkpoint_dir,
                    device=device,
                    precision=precision,
                    compile=False,
                )
            fish_python_decode_one_token_base = base_decode
            print(f"[PyTorch] Model ready · {device.upper()} · {str(precision).replace('torch.', '')}")
        fish_python_adapter = adapter_key
        _apply_compile_choice(selected_compile)

        _phase(progress, 0.34, "Loading Fish audio codec")
        fish_dir = Path(ROOT_DIR) / "modules" / "s2"
        codec_cfg = OmegaConf.load(fish_dir / "fish_speech" / "configs" / "modded_dac_vq.yaml")
        fish_python_codec = instantiate(codec_cfg)
        codec_checkpoint_path = os.path.join(fish_python_checkpoint_dir, "codec.pth")
        if not os.path.exists(codec_checkpoint_path):
            codec_checkpoint_path = os.path.join(FISH_MODELS_DIR, "codec.pth")
        state_dict = torch.load(codec_checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any("generator" in k for k in state_dict):
            state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}
        with _quiet_pytorch_engine("PyTorch codec load"):
            fish_python_codec.load_state_dict(state_dict, strict=False)
            fish_python_codec.eval().to(device=device, dtype=precision)
        del state_dict
        gc.collect()
        print(f"[PyTorch] Codec ready · {sample_rate if 'sample_rate' in locals() else fish_python_codec.sample_rate} Hz")
    else:
        from fish_speech.models.text2semantic.inference import generate_long
        if fish_python_compile_choice != selected_compile:
            print(f"[PyTorch] Compile selection changed: {fish_python_compile_choice} -> {selected_compile}")
            _apply_compile_choice(selected_compile)

    from fish_speech.models.text2semantic.inference import generate_long
    device = next(fish_python_model.parameters()).device
    prompt_tokens = None
    if ref_audio:
        _phase(progress, 0.45, "Encoding reference audio")
        wav, source_sr = torchaudio.load(str(ref_audio))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_sr = int(fish_python_codec.sample_rate)
        if int(source_sr) != target_sr:
            wav = torchaudio.functional.resample(wav, int(source_sr), target_sr)
        wav = wav[0].to(device)
        audios = wav[None, None, :].to(dtype=next(fish_python_codec.parameters()).dtype)
        audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, feature_lengths = fish_python_codec.encode(audios, audio_lengths)
        prompt_tokens = [indices[0, :, : feature_lengths[0]].cpu()]
    else:
        _phase(progress, 0.45, "No reference audio · native unconditioned timbre")
        print("[PyTorch] No reference audio: generating with Fish S2's native unconditioned/random timbre mode.")
    chunks = split_long_text(text, chunk_mode)
    gap_seconds = max(0.0, float(chunk_gap or 0.0))

    def generate_codes(piece):
        global fish_python_compile_active, fish_python_decode_one_token
        attempted = set()
        while True:
            active = fish_python_compile_active
            attempted.add(active)
            force_seed(normalized_seed)
            try:
                generator = generate_long(
                    model=fish_python_model,
                    device=device,
                    decode_one_token=fish_python_decode_one_token,
                    text=piece,
                    num_samples=1,
                    max_new_tokens=max(128, int(len(piece) * 4.5)),
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temp,
                    repetition_penalty=rep_pen,
                    compile=False,
                    iterative_prompt=True,
                    chunk_length=200,
                    prompt_text=[ref_text] if prompt_tokens is not None and ref_text else None,
                    prompt_tokens=prompt_tokens,
                )
                # Keep upstream tqdm visible: it is the useful token/frame progress signal.
                # Other noisy model/codec phases remain filtered by _quiet_pytorch_engine.
                codes = [r.codes for r in generator if getattr(r, "action", None) == "sample" and r.codes is not None]
                if not codes:
                    raise RuntimeError("Fish S2 returned no semantic codes.")
                return codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
            except Exception as exc:
                if active == "Eager / Disabled":
                    raise
                fallback = _next_runtime_compile_fallback(active)
                print(f"\n[WARNING] torch.compile runtime failed in '{active}': {type(exc).__name__}: {exc}")
                print(f"[WARNING] Automatic fallback -> {fallback}; retrying this chunk without restarting the app.")
                try:
                    torch.compiler.reset()
                except Exception:
                    pass
                # Runtime fallback must be exact, not restart from the user's original chain.
                if fallback == "Eager / Disabled":
                    fish_python_decode_one_token = fish_python_decode_one_token_base
                    fish_python_compile_active = fallback
                else:
                    mode = _COMPILE_MODE[fallback]
                    try:
                        fish_python_decode_one_token = torch.compile(fish_python_decode_one_token_base, mode=mode, fullgraph=True)
                        fish_python_compile_active = fallback
                    except Exception as compile_exc:
                        print(f"[WARNING] Fallback '{fallback}' could not initialize: {compile_exc}")
                        fish_python_decode_one_token = fish_python_decode_one_token_base
                        fish_python_compile_active = "Eager / Disabled"
                if fish_python_compile_active in attempted:
                    fish_python_decode_one_token = fish_python_decode_one_token_base
                    fish_python_compile_active = "Eager / Disabled"

    audio_segments = []
    sample_rate = fish_python_codec.sample_rate
    with torch.no_grad():
        for idx, piece in enumerate(chunks or [text]):
            _phase(progress, 0.52 + 0.40 * idx / max(1, len(chunks)), f"Generating chunk {idx+1}/{len(chunks)} · {fish_python_compile_active}")
            _console_progress("PyTorch synthesis", idx, len(chunks), f"chunk {idx+1}/{len(chunks)} · {fish_python_compile_active}")
            merged_codes = generate_codes(piece).to(device)
            print(f"[PyTorch] Chunk {idx+1}/{len(chunks)} · semantic frames: {merged_codes.shape[-1]}")
            _phase(progress, 0.56 + 0.40 * idx / max(1, len(chunks)), f"Decoding audio {idx+1}/{len(chunks)}")
            with _quiet_pytorch_engine("PyTorch codec"):
                waveform = fish_python_codec.from_indices(merged_codes[None])[0, 0].cpu().float().numpy()
            audio_segments.append(waveform)
            if idx < len(chunks) - 1 and gap_seconds > 0:
                audio_segments.append(np.zeros(int(sample_rate * gap_seconds), dtype=np.float32))
        _console_progress("PyTorch synthesis", len(chunks), len(chunks), f"{fish_python_compile_active}")

    if not audio_segments:
        raise RuntimeError("Fish S2 produced no audio.")
    audio_np = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    audio_int16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
    duration = len(audio_int16) / float(sample_rate) if sample_rate else 0.0
    print(f"[PyTorch] Complete · {duration:.2f}s audio · {fish_python_compile_active}")
    _phase(progress, 0.98, f"PyTorch synthesis complete · {fish_python_compile_active}")
    return sample_rate, audio_int16

def process_audio_array(audio_data):
    """Converts audio to mono and normalizes volume to [-1, 1]"""
    # Convert to mono if stereo
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize volume
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    return audio_data


def write_synthesized_audio(path, audio_data, sample_rate):
    """Write PCM16 dual-mono so browser players reproduce the signal centered."""
    mono = process_audio_array(np.asarray(audio_data))
    pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    dual_mono = np.column_stack((pcm16, pcm16))
    sf.write(path, dual_mono, sample_rate, subtype="PCM_16")


def begin_inference():
    """Reset the cooperative cancellation token for a new inference job."""
    _inference_cancel_event.clear()


def inference_cancelled():
    return _inference_cancel_event.is_set()


def stop_inference():
    """Cancel the active inference without terminating the Gradio application.

    For the native C++ path, killing only the managed s2.exe server immediately
    releases a blocked HTTP /generate request (including a stuck CUDA codec graph).
    The next inference transparently starts a fresh server.  PyTorch jobs use the
    same cancellation token plus Gradio event cancellation and stop cooperatively
    between chunks.
    """
    _inference_cancel_event.set()
    killed_cpp = _terminate_cpp_engine()

    if killed_cpp:
        print("[STOP] Active s2.cpp inference cancelled; s2.exe terminated. GUI remains running.")
        return "Inference stopped. The C++ runtime was terminated and will restart automatically on the next Generate."
    print("[STOP] Inference cancellation requested.")
    return "Inference stop requested. The current job has been cancelled; the GUI remains running."


def clone_voice(engine, cpp_model_str, cpp_backend, cpp_device, codec_cuda, lora_adapter, text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, compile_mode, seed, progress=gr.Progress()):
    global s2_process, s2_current_model, s2_current_backend, s2_current_device, s2_current_codec_cuda, s2_current_lora
    global _s2_last_command

    normalized_seed = resolve_seed_for_generation(seed)
    compile_mode = INTERNAL_TORCH_COMPILE_MODE

    if not text:
        return None, "Please enter some text to synthesize.", normalized_seed
    if ref_audio and not str(ref_text or "").strip():
        return None, "Reference transcript is required when reference audio is provided.", normalized_seed
        
    timestamp = int(time.time() * 1000)
    out_wav = os.path.join(OUTPUTS_DIR, f"output_{timestamp}.wav")
    
    # Calculate auto tokens based on target text length dynamically
    expected_new_tokens = int(len(text) * 4.5)
    
    if engine == ENGINE_CPP:
        cpp_exec, checked_paths = find_s2_executable()
        if not cpp_exec:
            return None, format_s2_not_found_error(checked_paths), normalized_seed

        # Auto-Unload PyTorch if switching to CPP
        if fish_python_model is not None:
            print("Auto-Unloading PyTorch Model to free VRAM for CPP...")
            unload_python_engine()

        from huggingface_hub import hf_hub_download
        
        filename = GGUF_MODELS.get(cpp_model_str)
        if not filename:
             return None, "Invalid GGUF model selected.", normalized_seed
        
        progress(0.2, desc=f"Checking GGUF model: {filename}")
        print(f"Checking/Downloading {filename}...")
        try:
            model_path = hf_hub_download(
                repo_id="rodrigomt/s2-pro-gguf",
                filename=filename,
                local_dir=S2_CPP_MODELS_DIR
            )
        except Exception as e:
            return None, f"Failed to download GGUF model: {e}", normalized_seed
        
        # Native C++ Fast-AR LoRA adapters remain separate from the base GGUF.
        # This makes the same adapter usable with F16 or quantized models.
        cpp_lora_path = ensure_cpp_lora_adapter(lora_adapter)
        codec_policy = str(codec_cuda or "Full GPU (Recommended)")
        requested_backend = str(cpp_backend or "Auto (CUDA on NVIDIA / Vulkan otherwise)")
        try:
            device_index = max(0, int(cpp_device))
        except Exception:
            device_index = 0

        family = _s2_quant_family(filename)
        is_auto_backend = requested_backend.startswith("Auto")

        if is_auto_backend:
            backend, auto_chain, failed_backends = _s2_auto_choose_backend(filename, cpp_exec)
            if backend is None:
                return None, (
                    f"s2.cpp Auto exhausted every safe backend for {family}: "
                    f"{' -> '.join(_s2_auto_backend_candidates(filename, cpp_exec))}. "
                    f"Failed this session: {', '.join(sorted(failed_backends)) or 'none'}. "
                    f"See {S2_RUNTIME_LOG}"
                ), normalized_seed
            chain_text = " -> ".join(auto_chain)
            failed_text = ", ".join(sorted(failed_backends)) if failed_backends else "none"
            print(
                f"[s2.cpp] Auto policy · {family}: {chain_text} · "
                f"selected {backend} · failed this session: {failed_text}"
            )
            _phase(progress, 0.27, f"s2.cpp Auto · {family} · {backend}")
        elif requested_backend.startswith("CUDA"):
            backend = "CUDA"
            auto_chain = [backend]
        elif requested_backend.startswith("Vulkan"):
            backend = "Vulkan"
            auto_chain = [backend]
        else:
            backend = "CPU"
            auto_chain = [backend]

        def _auto_fallback_after_native_failure(report, context):
            if not is_auto_backend:
                return None
            failed_backend = backend
            _s2_auto_mark_failed(filename, failed_backend)
            if not _s2_auto_has_fallback(filename, cpp_exec, failed_backend):
                return None

            next_backend, chain, failed = _s2_auto_choose_backend(filename, cpp_exec)
            print(
                f"[s2.cpp] Auto fallback · {family} · {failed_backend} failed during {context}; "
                f"retrying with {next_backend}. Safe chain: {' -> '.join(chain)}"
            )
            print(f"[s2.cpp] Previous native failure:\n{report}")
            _phase(progress, 0.22, f"Auto fallback · {failed_backend} → {next_backend}")
            _clear_s2_dead_state()

            # Preserve the exact generation seed. The recursive call sees the
            # failed-backend memory and selects the next safe backend.
            return clone_voice(
                engine, cpp_model_str, cpp_backend, cpp_device, codec_cuda,
                lora_adapter, text, ref_audio, ref_text, top_p, top_k, temp,
                rep_pen, chunk_mode, chunk_gap, compile_mode, normalized_seed,
                progress
            )

        # Start server if model, compute backend/offload, codec policy, or adapter changed.
        if (
            s2_process is None or
            (s2_process is not None and s2_process.poll() is not None) or
            s2_current_model != filename or
            s2_current_backend != backend or
            s2_current_device != device_index or
            s2_current_codec_cuda != codec_policy or
            s2_current_lora != cpp_lora_path
        ):
            if s2_process is not None:
                try:
                    if s2_process.poll() is None:
                        s2_process.kill()
                    s2_process.wait(timeout=5)
                except Exception:
                    pass
                s2_process = None
                time.sleep(1.0)  # Let Windows release port 3030
                
            progress(0.3, desc="Starting Fish CPP Server...")
            # Optimized thread count for modern CPUs (P-cores/E-cores and high core counts)
            # Use physical cores if available (psutil), otherwise logical/2 for efficiency.
            try:
                import psutil
                threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
            except:
                threads = (os.cpu_count() // 2) if (os.cpu_count() and os.cpu_count() > 8) else (os.cpu_count() or 4)
            threads = max(1, int(os.environ.get("S2_THREADS", threads)))
            print(f"[s2.cpp] Using {threads} CPU worker threads (override with S2_THREADS).")

            cmd = [
                cpp_exec,
                "-m", model_path,
                "-t", TOKENIZER_PATH,
                "--server",
                "-threads", str(threads),
            ]
            # Universal Windows runtime: CPU + CUDA + Vulkan are compiled into
            # the same s2.exe. The embedded s2.cpp revision uses full-backend
            # execution rather than layer-by-layer partial offload.
            if backend == "CUDA":
                cmd.extend(["--cuda", str(device_index)])
            elif backend == "Vulkan":
                cmd.extend(["--vulkan", str(device_index)])

            if codec_policy.startswith("Hybrid") or backend == "CPU":
                cmd.append("--codec-cpu")
                print(f"[s2.cpp] Codec execution: CPU ({'hybrid GPU-model/CPU-codec' if backend != 'CPU' else 'full CPU'}).")
            else:
                # Full CUDA/Vulkan codec path. The patched codec decoder keeps
                # VQ lookup + quantizer + waveform decoder in one backend graph.
                cmd.append("--codec-follow-backend")
                print(f"[s2.cpp] Codec execution: Full {backend} (reference encoder + waveform decoder).")

            if cpp_lora_path:
                cmd.extend(["--lora", cpp_lora_path])

            # Fix PATH correctly for s2.exe to find cublas64_##.dll
            env = os.environ.copy()
            path_key = "PATH" if "PATH" in env else ("Path" if "Path" in env else "PATH")
            
            # 1. Start with mandatory app paths
            s2_dir_path = os.path.dirname(cpp_exec)
            s2_bin_path = os.path.join(s2_dir_path, "bin")
            extra_paths = [s2_dir_path, s2_bin_path]
            
            # 2. Add System CUDA_PATH if exists
            if "CUDA_PATH" in env:
                extra_paths.append(os.path.join(env["CUDA_PATH"], "bin"))
                
            # 3. Robustly scan for ALL installed CUDA Toolkit versions (for any user)
            import glob
            default_cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.exists(default_cuda_base):
                # Find all "vX.X" directories and sort them (highest version first)
                found_versions = glob.glob(os.path.join(default_cuda_base, "v*"))
                for v_dir in sorted(found_versions, reverse=True):
                    # Check both standard bin and bin/x64 (common in CUDA 13/lib)
                    for sub in ["bin", os.path.join("bin", "x64")]:
                        bin_path = os.path.join(v_dir, sub)
                        if os.path.exists(bin_path):
                            extra_paths.append(bin_path)

            # 4. Construct the new PATH (prioritize our detected CUDA paths)
            new_path_str = os.pathsep.join(list(dict.fromkeys(extra_paths))) # Remove duplicates
            env[path_key] = new_path_str + os.pathsep + env.get(path_key, "")

            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            _s2_log_tail.clear()
            _s2_last_command = list(cmd)
            _record_s2_native_line("=" * 72)
            _record_s2_native_line(
                f"[GUI] Starting s2.cpp · model={filename} · backend={backend} "
                f"· device={device_index} · codec={codec_policy} · lora={cpp_lora_path or 'None'}"
            )
            _record_s2_native_line("[GUI] Command: " + " ".join(map(str, cmd)))

            try:
                s2_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    startupinfo=startupinfo
                )
            except OSError as e:
                s2_process = None
                s2_current_model = None
                s2_current_backend = None
                s2_current_device = None
                s2_current_codec_cuda = False
                s2_current_lora = None
                return None, (
                    f"Failed to start s2.exe:\n{cpp_exec}\n\n"
                    f"Windows reported: {e}"
                ), normalized_seed
            s2_current_model = filename
            s2_current_backend = backend
            s2_current_device = device_index
            s2_current_codec_cuda = codec_policy
            s2_current_lora = cpp_lora_path
            
            # Wait for server ready.
            # Re-use module-level log_queue & drain_thread to avoid re-creating
            # closures on every call (was causing RAM leak).
            global _s2_log_queue, _s2_drain_thread
            # Drain any leftover items from a previous startup
            while not _s2_log_queue.empty():
                try: _s2_log_queue.get_nowait()
                except: break

            captured_logs = []

            def _drain_stdout(proc, q):
                try:
                    for line in proc.stdout:
                        q.put(line)
                        _record_s2_native_line(line)
                        # Mirror the native runtime continuously to both CMD and
                        # the shared HTML console instead of showing only startup logs.
                        try:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        except Exception:
                            pass
                except Exception:
                    pass
                finally:
                    q.put(None)  # sentinel

            _s2_drain_thread = _threading.Thread(target=_drain_stdout, args=(s2_process, _s2_log_queue), daemon=True)
            _s2_drain_thread.start()

            start_time = time.time()
            ready = False
            print("--- Starting s2.exe logs ---")

            while time.time() - start_time < 300:
                # Drain all log lines currently available (non-blocking)
                while True:
                    try:
                        line = _s2_log_queue.get_nowait()
                    except _queue.Empty:
                        break
                    if line is None:
                        break
                    line = line.strip()
                    if line:
                        captured_logs.append(line)

                # Check if process died
                if s2_process.poll() is not None:
                    proc = s2_process
                    time.sleep(0.10)
                    report = _s2_failure_report(
                        "s2.exe exited before the HTTP server became ready.",
                        proc=proc,
                        context="startup",
                    )
                    fallback = _auto_fallback_after_native_failure(report, "startup")
                    if fallback is not None:
                        return fallback
                    _clear_s2_dead_state()
                    return None, report, normalized_seed

                # Health-check the HTTP endpoint
                try:
                    requests.get("http://localhost:3030/", timeout=0.5)
                    ready = True
                    break
                except Exception:
                    pass

                time.sleep(0.5)

            if not ready:
                proc = s2_process
                report = _s2_failure_report(
                    "HTTP server did not become ready within 300 seconds.",
                    proc=proc,
                    context="startup timeout",
                )
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=5)
                except Exception:
                    pass
                fallback = _auto_fallback_after_native_failure(report, "startup timeout")
                if fallback is not None:
                    return fallback
                _clear_s2_dead_state()
                return None, report, normalized_seed
            
            # Extra wait: server binds port before model is fully loaded into VRAM
            time.sleep(2)
                
        def _cpp_http_generate(piece, token_budget):
            data = {
                'text': piece,
                'params': json.dumps({
                    'max_new_tokens': token_budget,
                    'temperature': temp, 'top_p': top_p, 'top_k': top_k,
                    'repetition_penalty': rep_pen, 'verbose': True,
                    'seed': normalized_seed,
                }),
            }
            if ref_audio:
                data['ref_text'] = ref_text
                with open(ref_audio, 'rb') as f:
                    files = {'reference_audio': (os.path.basename(ref_audio), f, 'audio/wav')}
                    return requests.post(
                        "http://localhost:3030/generate", data=data, files=files, timeout=600
                    )
            # Force multipart/form-data even without an audio file because the
            # native s2.cpp server reads req.form. The sentinel field is ignored.
            return requests.post(
                "http://localhost:3030/generate", data=data,
                files={'_no_reference': (None, '1')}, timeout=600
            )

        paragraphs = split_long_text(text, chunk_mode)
        
        # --- Shared Easy-GUI long-form chunking ---
        if chunk_mode != "None" and len(paragraphs) > 1:
            print(f"[s2.cpp] Processing {len(paragraphs)} chunks ({chunk_mode})...")
            all_audio_segments = []
            final_sr = 32000 # Default fallback, normally detected from model

            for idx, para in enumerate(paragraphs):
                if inference_cancelled():
                    return None, "Inference stopped by user.", normalized_seed
                # Calculate tokens for THIS paragraph
                para_tokens = int(len(para) * 4.5)
                progress_pct = 0.7 + (idx / len(paragraphs)) * 0.25
                progress(progress_pct, desc=f"Synthesizing paragraph {idx + 1}/{len(paragraphs)}...")
                _console_progress("s2.cpp synthesis", idx, len(paragraphs), f"chunk {idx+1}/{len(paragraphs)}")
                
                # Retry loop for THIS paragraph
                current_audio = None
                max_para_retries = 2
                for att in range(max_para_retries):
                    if inference_cancelled():
                        return None, "Inference stopped by user.", normalized_seed
                    try:
                        res = _cpp_http_generate(para, para_tokens)
                        res.raise_for_status()
                            
                        import soundfile as sf
                        audio_data, sr = sf.read(io.BytesIO(res.content))
                        final_sr = sr
                        # Store segment as float32 for clean concatenation
                        all_audio_segments.append(audio_data.astype(np.float32))
                        
                        # Add a tiny bit of silence (0.5s) between paragraphs? optional but usually good
                        silence = np.zeros(int(sr * max(0.0, float(chunk_gap or 0.0))), dtype=np.float32)
                        all_audio_segments.append(silence)
                        
                        current_audio = True
                        break # Success for this para
                    except Exception as ecc:
                        if inference_cancelled():
                            return None, "Inference stopped by user.", normalized_seed
                        if s2_process is not None and s2_process.poll() is not None:
                            proc = s2_process
                            time.sleep(0.10)
                            report = _s2_failure_report(
                                ecc, proc=proc, context=f"paragraph {idx+1} generation"
                            )
                            fallback = _auto_fallback_after_native_failure(
                                report, f"paragraph {idx+1} generation"
                            )
                            if fallback is not None:
                                return fallback
                            print(report)
                            _clear_s2_dead_state()
                            return None, report, normalized_seed
                        print(f"Error in paragraph {idx+1} (attempt {att+1}): {ecc}")
                        time.sleep(1)
                
                if current_audio is None:
                    return None, f"Failed to generate paragraph {idx+1} after multiple attempts.", normalized_seed
            
            # Concatenate all segments
            if all_audio_segments:
                combined_audio = np.concatenate(all_audio_segments)
                
                write_synthesized_audio(out_wav, combined_audio, final_sr)
                
                import gc
                del all_audio_segments, combined_audio
                gc.collect()
                
                play_done_chime()
                _console_progress("s2.cpp synthesis", len(paragraphs), len(paragraphs), "complete")
                progress(1.0, desc="Done!")
                backend_note = f" · s2.cpp {backend}" + (" (Auto)" if is_auto_backend else "")
                return out_wav, f"Synthesis completed successfully (multi-paragraph){backend_note}.", normalized_seed
            else:
                return None, "No audio segments were generated correctly.", normalized_seed

        # --- Standard Single-Pass Logic (Used if 1 paragraph or split disabled) ---
        # Retry loop: server may reset connections while finishing VRAM allocation
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            if inference_cancelled():
                return None, "Inference stopped by user.", normalized_seed
            try:
                res = _cpp_http_generate(text, expected_new_tokens)
                res.raise_for_status()
                    
                import soundfile as sf
                audio_data, sr = sf.read(io.BytesIO(res.content))
                
                write_synthesized_audio(out_wav, audio_data, sr)
                # Explicit GC after each generation to prevent RAM growth
                import gc
                del audio_data
                gc.collect()
                play_done_chime()
                progress(1.0, desc="Done!")
                backend_note = f" · s2.cpp {backend}" + (" (Auto)" if is_auto_backend else "")
                return out_wav, f"Synthesis completed successfully{backend_note}.", normalized_seed
            except (ConnectionError, requests.exceptions.ConnectionError) as e:
                if inference_cancelled():
                    return None, "Inference stopped by user.", normalized_seed
                last_error = e
                if s2_process is not None and s2_process.poll() is not None:
                    break
                if attempt < max_retries - 1:
                    print(f"[s2.cpp] Connection reset (attempt {attempt+1}/{max_retries}), retrying in 2s...")
                    time.sleep(2)
                    continue
            except Exception as e:
                if inference_cancelled():
                    return None, "Inference stopped by user.", normalized_seed
                last_error = e
                break
        
        if s2_process is not None and s2_process.poll() is not None:
            proc = s2_process
            time.sleep(0.10)
            report = _s2_failure_report(
                last_error, proc=proc, context="HTTP /generate"
            )
            fallback = _auto_fallback_after_native_failure(report, "HTTP /generate")
            if fallback is not None:
                return fallback
            print(report)
            _clear_s2_dead_state()
            return None, report, normalized_seed

        return None, (
            "s2.cpp HTTP generation error (native process is still alive).\n"
            f"Error: {last_error}\n"
            f"Native log tail:\n{_s2_log_tail_text(40)}\n"
            f"Full native log: {S2_RUNTIME_LOG}"
        ), normalized_seed
            
    elif engine == ENGINE_PYTORCH:
        # Auto-Unload CPP Server if switching to PyTorch
        if s2_process is not None:
            print("Auto-Unloading CPP Server to free VRAM for PyTorch...")
            s2_process.kill()
            s2_process = None
            s2_current_model = None
            s2_current_backend = None
            s2_current_device = None
            s2_current_codec_cuda = False
            s2_current_lora = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            import soundfile as sf
            # Compile-mode-aware startup feedback. Actual compile failures can occur
            # on the first generated token, so generate_fish_python also performs
            # runtime fallback and retries the current chunk automatically.
            if HAS_COMPILE_CACHE:
                _phase(progress, 0.05, "Loading PyTorch model · safe compile · cached kernels available")
            else:
                _phase(progress, 0.05, "Loading PyTorch model · safe compile · first-run kernel compilation")
                print("[Fish Speech] torch.compile = max-autotune-no-cudagraphs; automatic fallback to Default/Eager is enabled.")
            sr, audio_int16 = generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, lora_adapter, compile_mode, seed, progress)
            
            # Convert to float32 to process, then back
            audio_data = audio_int16.astype(np.float32) / 32767.0
            progress(0.9, desc="Saving audio...")
            write_synthesized_audio(out_wav, audio_data, sr)
            play_done_chime()
            
            # Clean VRAM after logic
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            return out_wav, "Synthesis completed.", normalized_seed
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error generating with PyTorch: {str(e)}", normalized_seed

    return None, "Engine not supported.", normalized_seed

def generate_dialogue(engine, cpp_model_str, cpp_backend, cpp_device, codec_cuda, lora_adapter, top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, compile_mode, seed, row_count, silence_duration, *args, progress=gr.Progress()):
    # args is [sample1, ..., sample20, text1, ..., text20]
    num_max = 20 # Should match MAX_DIALOGUE_SEGMENTS
    samples = args[:num_max]
    texts = args[num_max:]
    
    segments = []
    for i in range(int(row_count)):
        s = samples[i]
        t = texts[i]
        if s and t:
            segments.append((s, t))
            
    if not segments:
        return None, "Please add at least one speaker and text.", normalize_seed(seed)
        
    all_audio_segments = []
    final_sr = 32000
    
    for i, (sample_name, text) in enumerate(segments):
        progress((i / len(segments)), desc=f"Processing segment {i+1}/{len(segments)} ({sample_name})...")
        _console_progress("Dialogue", i, len(segments), f"turn {i+1}/{len(segments)} · {sample_name}")
        
        # Load sample
        ref_audio, ref_text = load_sample(sample_name)
        if not ref_audio:
            print(f"Sample {sample_name} not found, skipping segment {i+1}")
            continue
            
        # Generate
        wav_path, status, _ = clone_voice(
            engine, cpp_model_str, cpp_backend, cpp_device, codec_cuda, lora_adapter, text, ref_audio, ref_text,
            top_p, top_k, temp, rep_pen, chunk_mode, chunk_gap, compile_mode, seed, progress=progress
        )
        
        if wav_path and os.path.exists(wav_path):
            audio_data, sr = sf.read(wav_path)
            # Normalize each speaker individually before concatenation
            audio_data = process_audio_array(audio_data)
            final_sr = sr
            all_audio_segments.append(audio_data.astype(np.float32))
            
            # Add silence between speakers
            if silence_duration > 0:
                silence = np.zeros(int(sr * silence_duration), dtype=np.float32)
                all_audio_segments.append(silence)
        else:
            return None, f"Error in segment {i+1} ({sample_name}): {status}", normalize_seed(seed)
            
    if all_audio_segments:
        # Concatenate (exclude last silence if added)
        if silence_duration > 0 and len(all_audio_segments) > 1:
            combined = np.concatenate(all_audio_segments[:-1])
        else:
            combined = np.concatenate(all_audio_segments)
            
        combined = process_audio_array(combined)
            
        # Output file
        out_wav = os.path.join(OUTPUTS_DIR, f"dialogue_{int(time.time()*1000)}.wav")
        write_synthesized_audio(out_wav, combined, final_sr)
        _console_progress("Dialogue", len(segments), len(segments), "complete")
        return out_wav, f"Dialogue generated successfully with {len(segments)} segments!", normalize_seed(seed)
    
    return None, "No audio generated.", normalize_seed(seed)

def transcribe_only(audio_path, model_size, language_name, batch_size=4, progress=gr.Progress()):
    begin_aux_job()
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        return "Error: faster-whisper module not found. Please run install.ps1 again."
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lang_code = WHISPER_LANGS.get(language_name)
    
    try:
        _phase(progress, 0.2, f"Loading Faster-Whisper {model_size} · {device}")
        
        whisper_cache = os.path.join(MODELS_DIR, "whisper")
        os.makedirs(whisper_cache, exist_ok=True)
        
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=whisper_cache)
        batched_model = BatchedInferencePipeline(model=model)
        
        _phase(progress, 0.5, f"Transcribing reference audio · batch {max(1, int(batch_size or 1))}")
        segments, info = batched_model.transcribe(
            audio_path, language=lang_code, batch_size=max(1, int(batch_size or 1)), beam_size=5
        )
        
        # Gather all text segments
        pieces = []
        for segment in segments:
            if _aux_cancelled():
                return "Transcription stopped by user."
            pieces.append(segment.text)
        text = " ".join(pieces).strip()
        
        # Unload from VRAM after transcription is done
        import gc
        del batched_model
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        _phase(progress, 1.0, "Transcription complete")
        play_done_chime()
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Faster-Whisper Error: {str(e)}"

def handle_full_batch_process(source_folder, dataset_name, model_size, language_name, batch_size, progress=gr.Progress()):
    begin_aux_job()
    if not source_folder or not os.path.isdir(source_folder):
        return "Error: Please provide a valid source folder path."
    if not dataset_name or not dataset_name.strip():
        return "Error: Please provide a target dataset name."

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
        audio_files.extend(glob.glob(os.path.join(source_folder, ext)))
        if os.name != "nt":
            audio_files.extend(glob.glob(os.path.join(source_folder, ext.upper())))
    audio_files = list(dict.fromkeys(audio_files))
    if not audio_files:
        return "Error: No audio files found in the source folder."

    target_dir = os.path.join(TRAINING_DATA_DIR, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    metadata = []
    needs_asr = []
    paired = 0

    def read_pair_text(audio_path):
        stem = os.path.splitext(audio_path)[0]
        for ext in (".txt", ".lab"):
            candidate = stem + ext
            if os.path.isfile(candidate):
                for enc in ("utf-8", "utf-8-sig", "latin-1"):
                    try:
                        text = Path(candidate).read_text(encoding=enc).strip()
                        if text:
                            return text
                    except Exception:
                        pass
        return None

    _phase(progress, 0.08, f"Preparing {len(audio_files)} audio/text pairs")
    for i, audio_path in enumerate(audio_files, 1):
        if _aux_cancelled():
            return f"Dataset build stopped by user after {i-1}/{len(audio_files)} files."
        stem = Path(audio_path).stem
        dest_wav = os.path.join(target_dir, stem + ".wav")
        pair_text = read_pair_text(audio_path)
        try:
            if Path(audio_path).suffix.lower() == ".wav":
                if os.path.abspath(audio_path) != os.path.abspath(dest_wav):
                    shutil.copy2(audio_path, dest_wav)
            else:
                # Fish training ultimately consumes WAV. Convert only when necessary;
                # do not normalize or otherwise alter already-valid source audio.
                y, sr = sf.read(audio_path, always_2d=True, dtype="float32")
                if y.shape[1] > 1:
                    y = np.mean(y, axis=1, keepdims=True)
                sf.write(dest_wav, y[:, 0], sr)
            if pair_text:
                Path(os.path.splitext(dest_wav)[0] + ".lab").write_text(pair_text, encoding="utf-8")
                metadata.append(f"{os.path.abspath(dest_wav)}|{pair_text}")
                paired += 1
            else:
                needs_asr.append(dest_wav)
        except Exception as exc:
            print(f"[Dataset] Failed {Path(audio_path).name}: {exc}")
        _console_progress("Dataset pairs", i, len(audio_files), Path(audio_path).name)
        progress(0.08 + 0.32 * (i / len(audio_files)), desc=f"Prepared {i}/{len(audio_files)} files")

    if needs_asr:
        _phase(progress, 0.42, f"{paired} transcript pairs reused · transcribing {len(needs_asr)} missing transcripts")
        try:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_cache = os.path.join(MODELS_DIR, "whisper")
            os.makedirs(whisper_cache, exist_ok=True)
            model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8", download_root=whisper_cache)
            batched_model = BatchedInferencePipeline(model=model)
            for i, audio_path in enumerate(needs_asr, 1):
                if _aux_cancelled():
                    del batched_model, model
                    if device == "cuda": torch.cuda.empty_cache()
                    return f"Dataset build stopped by user. Reused {paired} existing transcripts; ASR stopped at {i-1}/{len(needs_asr)}."
                _console_progress("Transcription", i, len(needs_asr), Path(audio_path).name)
                progress(0.45 + 0.45 * (i / len(needs_asr)), desc=f"Transcribing missing transcript {i}/{len(needs_asr)}")
                segments, _ = batched_model.transcribe(audio_path, language=None, batch_size=max(1, int(batch_size)))
                pieces=[]
                for segment in segments:
                    if _aux_cancelled(): break
                    pieces.append(segment.text)
                if _aux_cancelled():
                    return "Dataset transcription stopped by user."
                text=" ".join(pieces).strip()
                Path(os.path.splitext(audio_path)[0] + ".lab").write_text(text, encoding="utf-8")
                metadata.append(f"{os.path.abspath(audio_path)}|{text}")
            del batched_model, model
            import gc; gc.collect()
            if device == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            import traceback; traceback.print_exc()
            return f"Dataset audio prepared, but transcription failed: {exc}"
    else:
        _phase(progress, 0.82, f"All {paired} WAV/TXT pairs found · Faster-Whisper skipped")

    metadata_path = os.path.join(target_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        f.write("\n".join(metadata))
    _phase(progress, 1.0, "Dataset ready")
    play_done_chime()
    return (f"✨ Dataset ready: {len(metadata)} paired samples.\n"
            f"📍 datasets/{dataset_name}\n"
            f"♻️ Reused existing transcripts: {paired}\n"
            f"🛰️ Transcribed missing transcripts: {len(needs_asr)}")

def fix_audio_single(audio_path, normalize=True, to_mono=True):
    if not audio_path or not os.path.exists(audio_path):
        return audio_path, "Error: File not found."
        
    try:
        # Load audio without Librosa/Numba.
        y, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        if to_mono and y.shape[1] > 1:
            y = np.mean(y, axis=1, keepdims=True)
        if to_mono:
            y = y[:, 0]

        # Normalize
        if normalize:
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95

        # Overwrite file
        sf.write(audio_path, y, sr)
        
        msg = "Processed: "
        if normalize: msg += "Normalized "
        if to_mono: msg += "Mono "
        return audio_path, msg.strip()
    except Exception as e:
        return audio_path, f"Error: {str(e)}"

def fix_audio_batch(folder_path, normalize=True, to_mono=True, progress=gr.Progress()):
    if not folder_path or not os.path.isdir(folder_path):
        return "Please provide a valid folder path."
        
    import glob
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not audio_files:
        return "No audio files found."

    total = len(audio_files)
    processed = 0
    
    for i, audio_path in enumerate(audio_files):
        progress((i/total), desc=f"Processing ({i+1}/{total}): {os.path.basename(audio_path)}")
        try:
            # Overwrite logic without Librosa/Numba.
            y, sr = sf.read(audio_path, always_2d=True, dtype="float32")
            if to_mono and y.shape[1] > 1:
                y = np.mean(y, axis=1, keepdims=True)
            if to_mono:
                y = y[:, 0]
            if normalize:
                max_val = np.max(np.abs(y))
                if max_val > 0:
                    y = y / max_val * 0.95
            sf.write(audio_path, y, sr)
            processed += 1
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            
    progress(1.0, desc="Done!")
    return f"Batch processing complete. Processed {processed}/{total} files."

def save_prep_sample(audio_path, sample_name, transcription, language="auto"):
    if not audio_path:
        return "Please provide an audio file to save.", gr.update()
    if not sample_name or sample_name.strip() == "":
        sample_name = f"sample_{int(time.time())}"
        
    sample_name = "".join([c for c in sample_name if c.isalnum() or c in (" ", "_")]).replace(" ", "_").strip("_")
    
    dest_wav = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    dest_json = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
    
    try:
        shutil.copy2(audio_path, dest_wav)
        
        # Save JSON metadata pair as requested
        metadata = {
            "Type": "Sample",
            "Text": transcription.strip(),
            "Language": str(language or "auto")
        }
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        # Cleanup gc and update choices
        import gc
        gc.collect()
        return f"Successfully saved sample: {sample_name} in {SAMPLES_DIR}", gr.update(choices=get_sample_choices(), value=sample_name)
    except Exception as e:
        return f"Error saving sample: {str(e)}", gr.update()

def delete_sample(sample_name):
    if not sample_name:
        return "No sample selected.", gr.update()
    try:
        dest_audio = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
        dest_txt = os.path.join(SAMPLES_DIR, f"{sample_name}.txt")
        if os.path.exists(dest_audio): os.remove(dest_audio)
        if os.path.exists(dest_txt): os.remove(dest_txt)
        dest_json = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
        if os.path.exists(dest_json): os.remove(dest_json)
        return f"Deleted sample '{sample_name}'.", gr.update(choices=get_sample_choices(), value=None)
    except Exception as e:
        return f"Error deleting: {e}", gr.update()

# --- Training Backend Functions ---

def run_training_step(cmd, desc, progress):
    """Run a preparation subprocess while preserving tqdm/CR output.

    stdout is mirrored character-for-character so CMD and the shared HTML
    console show the same dynamic progress instead of waiting for a newline.
    """
    global training_process
    _phase(progress, 0.1, f"Starting {desc} · see CMD / Live Console")
    training_process = subprocess.Popen(
        cmd,
        cwd=FS_DIR if "python" in str(cmd).lower() else ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=True if os.name == "nt" else False,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    logs=[]
    line_buffer=[]
    p=training_process
    if p and p.stdout:
        try:
            while True:
                ch=p.stdout.read(1)
                if ch == "" and p.poll() is not None:
                    break
                if not ch:
                    continue
                sys.stdout.write(ch); sys.stdout.flush()
                if ch in ("\n", "\r"):
                    text="".join(line_buffer).strip()
                    if text:
                        logs.append(text)
                        if len(logs) > 200:
                            del logs[:-200]
                    line_buffer.clear()
                else:
                    line_buffer.append(ch)
        finally:
            tail="".join(line_buffer).strip()
            if tail: logs.append(tail)
        p.wait(); ret_code=p.returncode
    else:
        ret_code=0
    training_process=None
    if ret_code not in (0,-1):
        return False, f"{desc} failed with return code {ret_code}.\n\nLast logs:\n" + "\n".join(logs[-10:])
    print(f"\n[UI] {desc} completed successfully.")
    return True, f"{desc} completed successfully."

def handle_lora_dataset_prep(output_name, progress=gr.Progress()):
    # Our new batch processor already structures the dataset perfectly
    dataset_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    if not os.path.exists(dataset_dir):
        return f"failed: Dataset directory not found at {dataset_dir}."
        
    wavs = glob.glob(os.path.join(dataset_dir, "*.wav"))
    labs = glob.glob(os.path.join(dataset_dir, "*.lab"))
    
    if not wavs or not labs:
        return "failed: No .wav or .lab files found. Did you run the Batch Processor?"
        
    return f"Dataset structure verified natively. Found {len(wavs)} wavs and {len(labs)} labs."

def handle_lora_vq_extraction(output_name, progress=gr.Progress()):
    data_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    if not os.path.exists(data_dir):
        return f"failed: Dataset directory not found at {data_dir}."
        
    # Check for codec.pth
    codec_path = os.path.join(FISH_MODELS_DIR, "codec.pth")
    if not os.path.exists(codec_path):
        # Trigger download via hf_hub if missing
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="fishaudio/s2-pro", filename="codec.pth", local_dir=FISH_MODELS_DIR)

    cmd = f"\"{sys.executable}\" tools/vqgan/extract_vq.py \"{data_dir}\" --config-name modded_dac_vq --checkpoint-path \"{codec_path}\" --num-workers 1 --batch-size 1"
    success, msg = run_training_step(cmd, "VQ Extraction", progress)
    return msg

def _training_split_paths(output_name):
    root = Path(TRAINING_DATA_DIR) / str(output_name)
    return root, root / "_fish_split", root / "protos_train", root / "protos_val"


def _build_true_eval_split(output_name, val_fraction=0.10, min_samples=10):
    """Create a deterministic file-level train/validation holdout.

    build_dataset.py only consumes .npy + same-stem .lab, so the split tree uses
    hardlinks (copy fallback) for those two files only. Original dataset files
    remain untouched.
    """
    import hashlib

    root, split_root, _, _ = _training_split_paths(output_name)
    pairs = []
    for npy in sorted(root.glob("*.npy")):
        lab = npy.with_suffix(".lab")
        if lab.is_file():
            key = hashlib.sha256(npy.stem.encode("utf-8")).hexdigest()
            pairs.append((key, npy, lab))
    pairs.sort(key=lambda x: x[0])

    if len(pairs) < int(min_samples):
        shutil.rmtree(split_root, ignore_errors=True)
        return {
            "enabled": False, "total": len(pairs), "train": len(pairs), "val": 0,
            "reason": f"fewer than {min_samples} usable samples",
        }

    val_count = max(1, int(round(len(pairs) * float(val_fraction))))
    val_count = min(val_count, len(pairs) - 1)
    val_stems = {npy.stem for _, npy, _ in pairs[:val_count]}

    shutil.rmtree(split_root, ignore_errors=True)
    for subset in ("train", "val"):
        (split_root / subset).mkdir(parents=True, exist_ok=True)

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    train_count = val_count_actual = 0
    for _, npy, lab in pairs:
        subset = "val" if npy.stem in val_stems else "train"
        target = split_root / subset
        link_or_copy(npy, target / npy.name)
        link_or_copy(lab, target / lab.name)
        if subset == "val":
            val_count_actual += 1
        else:
            train_count += 1

    manifest = {
        "version": 1,
        "fraction": float(val_fraction),
        "total": len(pairs),
        "train": train_count,
        "val": val_count_actual,
        "validation_stems": sorted(val_stems),
    }
    (split_root / "split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"enabled": True, **manifest}


def _select_training_eval_reference(dataset_dir: Path, split_manifest: dict):
    """Choose one stable prompt utterance for perceptual training evaluation.

    Prefer a validation-holdout sample so the prompted monitor is never taken
    from optimization data. Tiny datasets without a holdout fall back to the
    deterministic first usable dataset sample; this affects only audio
    monitoring, never quantitative validation.
    """
    audio_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    stems = list(split_manifest.get("validation_stems") or [])
    source = "validation_holdout"
    if not stems:
        stems = sorted(p.stem for p in dataset_dir.glob("*.npy") if p.with_suffix(".lab").is_file())
        source = "dataset_fallback"

    for stem in stems:
        lab = dataset_dir / f"{stem}.lab"
        if not lab.is_file():
            continue
        audio = None
        for ext in audio_exts:
            candidate = dataset_dir / f"{stem}{ext}"
            if candidate.is_file():
                audio = candidate
                break
        if audio is None:
            # Prepared datasets are normally flat, but tolerate nested sources.
            for candidate in dataset_dir.rglob(f"{stem}.*"):
                if candidate.suffix.lower() in audio_exts and candidate.is_file():
                    audio = candidate
                    break
        if audio is None:
            continue
        transcript = lab.read_text(encoding="utf-8", errors="replace").strip()
        if transcript:
            return {
                "audio": str(audio.resolve()),
                "text": transcript,
                "stem": stem,
                "source": source,
            }
    return None


def handle_lora_sharding(output_name, progress=gr.Progress()):
    data_dir, split_root, train_proto, val_proto = _training_split_paths(output_name)
    split = _build_true_eval_split(output_name)

    for folder in (train_proto, val_proto):
        shutil.rmtree(folder, ignore_errors=True)

    if split["enabled"]:
        train_proto.mkdir(parents=True, exist_ok=True)
        val_proto.mkdir(parents=True, exist_ok=True)
        train_input = split_root / "train"
        val_input = split_root / "val"

        cmd = (
            f'"{sys.executable}" tools/llama/build_dataset.py '
            f'--input "{train_input}" --output "{train_proto}" '
            f'--text-extension .lab --num-workers 4'
        )
        success, msg = run_training_step(cmd, "Sharding Train Split", progress)
        if not success:
            return msg

        cmd = (
            f'"{sys.executable}" tools/llama/build_dataset.py '
            f'--input "{val_input}" --output "{val_proto}" '
            f'--text-extension .lab --num-workers 4'
        )
        success, msg = run_training_step(cmd, "Sharding Validation Split", progress)
        if not success:
            return msg

        print(
            f"[UI] True holdout split: {split['train']} train + {split['val']} validation "
            f"({split['fraction']*100:.0f}% holdout). No sample is shared.",
            flush=True,
        )
        return f"Sharding complete: {split['train']} train / {split['val']} validation."

    # Tiny datasets: keep all samples for training and do not pretend that
    # validation is independent. Prompted audio monitoring can still use a deterministic dataset fallback.
    train_proto.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'"{sys.executable}" tools/llama/build_dataset.py '
        f'--input "{data_dir}" --output "{train_proto}" '
        f'--text-extension .lab --num-workers 4'
    )
    success, msg = run_training_step(cmd, "Sharding Training Dataset", progress)
    if success:
        print(
            f"[UI] Validation holdout disabled: {split['reason']}. "
            "All usable samples remain in training; no val/* metrics will be claimed.",
            flush=True,
        )
    return msg

def handle_lora_train(output_name, model_name, max_steps, lr, progress=gr.Progress()):
    dataset_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    proto_dir = os.path.join(dataset_dir, "protos")
    if not os.path.exists(proto_dir):
        return f"Sharded data not found at {proto_dir}. Please run sharding first."
        
    # Prepare result dir
    if not model_name:
        model_name = f"{output_name}_{int(time.time())}"
    
    # Check for pretrained model
    if not os.path.exists(os.path.join(FISH_MODELS_DIR, "model.pth")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=FISH_MODELS_DIR)

    # Note: Use forward slashes for hydra on Windows or escape properly
    proto_dir_abs = os.path.abspath(proto_dir).replace("\\", "/")
    ckpt_dir_abs = os.path.abspath(FISH_MODELS_DIR).replace("\\", "/")
    
    cmd = (
        f"set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"\"{sys.executable}\" fish_speech/train.py "
        f"--config-name text2semantic_finetune "
        f"project={model_name} "
        f"+lora@model.model.lora_config=r_32_alpha_16_fast "
        f"trainer.max_steps={max_steps} "
        f"~model.lr_scheduler.lr_lambda "
        f"model.lr_scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR "
        f"+model.lr_scheduler.T_max={max_steps} "
        f"+model.lr_scheduler.eta_min={float(lr) * 0.10} "
        f"model.optimizer.lr={lr} "
        f"trainer.strategy=auto "
        f"trainer.devices=1 "
        f"data.num_workers=0 "
        f"pretrained_ckpt_path=\"{ckpt_dir_abs}\" "
        f"train_dataset.proto_files=[{proto_dir_abs}] "
        f"val_dataset.proto_files=[{proto_dir_abs}]"
    )
    
    success, msg = run_training_step(cmd, "LoRA Training", progress)
    return success, msg

# -----------------------------------------------------------------------------
# Modern project-aware training controller
# -----------------------------------------------------------------------------
_training_log_thread = None
_training_log_path = None
_training_progress_file = None
_training_started_monotonic = None
_training_total_steps = 0
_training_target_value = 0
_training_last_exit_code = None
_training_project = None


def _training_profile(vram_preset):
    text = str(vram_preset or "24GB VRAM")
    # Windows: keep dataloading in-process to avoid multiprocessing/spawn overhead.
    if "48GB" in text:
        return 2, 2, 0
    if "32GB" in text:
        return 2, 2, 0
    # S2-Pro validated Fast-AR LoRA recipe: one sample per micro-batch with
    # four-way gradient accumulation (effective batch 4).
    return 1, 4, 0


def dataset_training_statistics(folder, vram_preset="24GB VRAM"):
    dataset_dir = Path(TRAINING_DATA_DIR) / str(folder or "")
    wav_files = sorted(dataset_dir.glob("*.wav")) if dataset_dir.is_dir() else []
    total_seconds = 0.0
    for path in wav_files:
        try:
            info = sf.info(path)
            total_seconds += info.frames / float(info.samplerate)
        except Exception:
            pass
    bs, accum, workers = _training_profile(vram_preset)
    samples = len(wav_files)
    effective_batch = max(1, bs * accum)
    # Reserve the same deterministic 10% holdout used by dataset preparation.
    # For tiny datasets (<10), no holdout is created.
    train_samples = samples - max(1, int(round(samples * 0.10))) if samples >= 10 else samples
    # AutoTune budget: approximately three train-set-sized sample exposures,
    # expressed directly as optimizer steps. Training itself is step-only.
    autotune_steps = max(1, (train_samples // effective_batch) * 3) if train_samples else 1
    return {
        "samples": samples,
        "train_samples": train_samples,
        "val_samples": samples - train_samples,
        "minutes": total_seconds / 60.0,
        "mean_seconds": (total_seconds / samples) if samples else 0.0,
        "batch_size": bs,
        "accumulate": accum,
        "workers": workers,
        "effective_batch": effective_batch,
        "autotune_steps": autotune_steps,
    }


def _nice_interval(target, unit="steps"):
    target = max(1, int(target))
    candidates = (1000, 500, 250, 200, 100, 50, 25, 10) if unit == "steps" else (25, 20, 10, 5, 2, 1)
    ceiling = max(1, target // 4)
    for value in candidates:
        if value <= ceiling and target % value == 0:
            return value
    if unit == "steps":
        return max(10, min(target, round(target / 5 / 10) * 10 or 10))
    return max(1, min(target, round(target / 5) or 1))


def autotune_training(dataset, vram_preset="24GB VRAM"):
    stats = dataset_training_statistics(dataset, vram_preset)
    hours = stats["minutes"] / 60.0
    base_steps = 1200 if hours >= 20 else 1000 if hours >= 5 else 750 if hours >= 1 else 500

    steps = base_steps
    learning_rate = 1e-5
    rank = 32
    alpha = 16
    dropout = 0.10
    save_steps = 50 if steps <= 500 else 100
    report = (
        f"**Dataset:** {stats['samples']:,} clips · {stats['minutes']:.1f} min · train {stats['train_samples']:,} / validation {stats['val_samples']:,}.\n\n"
        f"**Training:** Fast AR LoRA only · {steps:,} optimizer steps · save/eval {save_steps} · "
        f"r/a {rank}/{alpha} · LR {learning_rate:.1e} · WD 0.01 · max length 4096 · Cosine 10% floor.\n\n"
        "Validated speaker/timbre adaptation with the non-targeted model branch kept frozen."
    )
    return {**stats, "steps":steps, "save_steps":save_steps, "eval_steps":save_steps,
            "rank":rank, "alpha":alpha, "dropout":dropout, "lr":learning_rate,
            "weight_decay":0.01, "max_length":4096, "causal":False,
            "interactive_prob":1.0, "gradient_clip":1.0,
            "scheduler":"Cosine 10% floor", "training_seed":1234, "report":report}


def normalize_training_steps(steps, save_steps, eval_steps):
    """Normalize the only supported training-length unit: optimizer steps."""
    total_steps = max(1, int(steps))
    checkpoint_steps = min(max(1, int(save_steps)), total_steps)
    evaluation_steps = min(max(1, int(eval_steps)), total_steps)
    return total_steps, checkpoint_steps, evaluation_steps


def _training_inputs_are_fresh(output_name):
    dataset_dir, split_root, train_proto, val_proto = _training_split_paths(output_name)
    sources = [*dataset_dir.glob("*.npy"), *dataset_dir.glob("*.lab")]
    train_shards = [*train_proto.rglob("*.proto"), *train_proto.rglob("*.protos")]
    if not sources or not train_shards:
        return False
    manifest = split_root / "split.json"
    if not manifest.is_file():
        return False
    try:
        split = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(split.get("total", -1)) != len([p for p in dataset_dir.glob("*.npy") if p.with_suffix(".lab").is_file()]):
        return False
    outputs = train_shards + ([*val_proto.rglob("*.proto"), *val_proto.rglob("*.protos")] if int(split.get("val", 0)) else [])
    if int(split.get("val", 0)) and not outputs[len(train_shards):]:
        return False
    newest_source = max(p.stat().st_mtime_ns for p in sources)
    oldest_output = min(p.stat().st_mtime_ns for p in outputs)
    return oldest_output >= newest_source


def _pump_training_output(proc, log_path):
    """Mirror child output to CMD + embedded console while preserving CR progress."""
    global _training_last_exit_code
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace", buffering=1) as file:
        try:
            while True:
                ch = proc.stdout.read(1) if proc.stdout is not None else ""
                if ch == "" and proc.poll() is not None:
                    break
                if ch:
                    file.write(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
        except Exception as exc:
            print(f"\n[WARNING] Training log mirror stopped: {exc}")
    proc.wait()
    _training_last_exit_code = proc.returncode
    print(f"\n[UI] Training process finished with exit code {proc.returncode}.")
    if proc.returncode == 0:
        play_done_chime()


def _launch_training_process(command, env, project, total_steps):
    global training_process, _training_log_thread, _training_log_path, _training_progress_file
    global _training_started_monotonic, _training_total_steps
    global _training_target_value, _training_last_exit_code, _training_project
    if training_process is not None and training_process.poll() is None:
        raise RuntimeError(f"Training is already running (PID {training_process.pid}).")
    result_dir = Path(FS_DIR) / "results" / project
    result_dir.mkdir(parents=True, exist_ok=True)
    _training_log_path = str(result_dir / "easy_gui_training.log")
    _training_progress_file = str(result_dir / "easy_gui_progress.json")
    try:
        Path(_training_progress_file).unlink(missing_ok=True)
    except Exception:
        pass
    env = dict(env)
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "FISH_EASY_PROGRESS_FILE": _training_progress_file,
        "FISH_EASY_TOTAL_STEPS": str(int(total_steps)),
    })
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    training_process = subprocess.Popen(
        command,
        cwd=FS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=flags,
    )
    _training_started_monotonic = time.monotonic()
    _training_total_steps = int(total_steps)
    _training_target_value = int(total_steps)
    _training_last_exit_code = None
    _training_project = project
    _training_log_thread = threading.Thread(
        target=_pump_training_output,
        args=(training_process, _training_log_path),
        name="fish-easy-training-log-tee",
        daemon=True,
    )
    _training_log_thread.start()
    return training_process.pid


def training_progress_snapshot():
    proc = training_process
    running = proc is not None and proc.poll() is None
    exit_code = None if running else (_training_last_exit_code if _training_last_exit_code is not None else (proc.returncode if proc else None))
    payload = {}
    if _training_progress_file and Path(_training_progress_file).is_file():
        try:
            payload = json.loads(Path(_training_progress_file).read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    step = int(payload.get("step", 0) or 0)
    total_steps = int(payload.get("total_steps", _training_total_steps) or _training_total_steps or 0)
    loss = payload.get("loss")
    elapsed = max(0.0, time.monotonic() - _training_started_monotonic) if _training_started_monotonic else float(payload.get("elapsed", 0.0) or 0.0)
    pct = min(100.0, step / total_steps * 100.0) if total_steps else 0.0
    if not running and exit_code == 0 and total_steps and step >= total_steps:
        pct = 100.0
    eta = None
    if running and step > 0 and total_steps > step:
        eta = elapsed / step * (total_steps - step)
    completed_units = step
    total_units = _training_target_value or total_steps
    unit = "Steps"
    return {
        "running": running,
        "exit_code": exit_code,
        "step": step,
        "total_steps": total_steps,
        "completed_units": completed_units,
        "total_units": total_units,
        "unit": unit,
        "loss": loss,
        "save_every_steps": int(payload.get("save_every_steps", 0) or 0),
        "eval_every_steps": int(payload.get("eval_every_steps", 0) or 0),
        "next_save_step": int(payload.get("next_save_step", 0) or 0),
        "next_eval_step": int(payload.get("next_eval_step", 0) or 0),
        "pct": pct,
        "elapsed": elapsed,
        "eta": eta,
        "project": _training_project,
    }


def training_status():
    snap = training_progress_snapshot()
    if snap["running"]:
        return f"Running '{snap.get('project')}' · PID {training_process.pid} · step {snap['step']}/{snap['total_steps']}"
    if snap["exit_code"] is None:
        return "Idle."
    return f"Finished with exit code {snap['exit_code']}."


def stop_training():
    """Request a graceful training stop and always release GUI process state.

    On Windows CTRL_BREAK reaches Intel/Fortran runtimes as a fatal console
    event (`forrtl: error (200)`), so do not use it for normal UI cancellation.
    terminate() asks the child to exit normally; a short asynchronous fallback
    kills only a genuinely stuck child without blocking Gradio.
    """
    global training_process
    proc = training_process
    if proc is None:
        return "Training is not running."

    try:
        if proc.poll() is not None:
            training_process = None
            return "Training process had already finished."

        proc.terminate()

        def _finish_stop(p):
            global training_process
            try:
                p.wait(timeout=8)
            except Exception:
                try:
                    if p.poll() is None:
                        p.kill()
                    p.wait(timeout=3)
                except Exception:
                    pass
            finally:
                if training_process is p:
                    training_process = None

        _aux_threading.Thread(target=_finish_stop, args=(proc,), daemon=True).start()
        return "Graceful stop requested. Waiting for the training process to exit..."
    except Exception as exc:
        # Never leave a stale process object permanently locking Start Training.
        if proc.poll() is not None:
            training_process = None
        return f"Could not request training stop cleanly: {exc}"

def handle_lora_unified(output_name, model_name, max_steps, lr, vram_preset,
                        lora_rank=32, lora_alpha=16, lora_dropout=0.10, training_seed=1234,
                        save_every_steps=100, eval_every_steps=100,
                        resume_checkpoint="Fresh / None",
                        eval_text="", eval_enabled=True,
                        weight_decay=0.01, max_length=4096, causal=False,
                        interactive_prob=1.0, gradient_clip=1.0,
                        scheduler="Cosine 10% floor",
                        progress=gr.Progress()):
    """Prepare the Fish dataset, then launch the fixed Fast-AR LoRA recipe asynchronously."""
    if not output_name or output_name == "(No datasets)":
        return "Error: select a prepared dataset."
    if not model_name:
        return "Error: select a Training project."

    if training_process is not None and training_process.poll() is None:
        return f"Error: training is already running (PID {training_process.pid}). Stop it before starting another run."

    _phase(progress, 0.02, "Freeing inference models / RAM / VRAM")
    unload_all_models(
        reason=f"training start -> {model_name}",
        reset_compiler=True,
        trim_ram=True,
    )

    total_steps, checkpoint_steps, evaluation_steps = normalize_training_steps(
        max_steps, save_every_steps, eval_every_steps,
    )
    if int(lora_rank) < 1 or int(lora_alpha) < 1:
        return "Error: LoRA rank and alpha must be positive."
    if float(lora_alpha) / float(lora_rank) > 1.0:
        return "Error: alpha/r > 1.0 is blocked: S2-Pro reports show a high risk of destructive drift."
    if int(max_length) < 512 or int(max_length) > 4096:
        return "Error: max_length must be 512..4096."
    if not 0.0 <= float(interactive_prob) <= 1.0:
        return "Error: interactive probability must be 0..1."
    print(
        f"[UI] Training contract: {total_steps} optimizer steps · "
        f"save every {checkpoint_steps} · validation every {evaluation_steps} · "
        f"prompted audio {'enabled' if bool(eval_enabled) else 'disabled'}."
    )
    # Dataset validation is a true holdout only when the prepared split exists.

    _phase(progress, 0.05, "Verifying prepared dataset")
    msg = handle_lora_dataset_prep(output_name, progress)
    if "failed" in msg.lower():
        return msg

    if _training_inputs_are_fresh(output_name):
        print("[SKIP] VQ/protobuf dataset is current; preprocessing will not be repeated.")
    else:
        _phase(progress, 0.14, "Extracting VQ codes")
        msg = handle_lora_vq_extraction(output_name, progress)
        if "failed" in msg.lower():
            return msg
        _phase(progress, 0.34, "Building Fish protobuf dataset")
        msg = handle_lora_sharding(output_name, progress)
        if "failed" in msg.lower():
            return msg

    dataset_dir, split_root, train_proto_dir, val_proto_dir = _training_split_paths(output_name)
    train_proto_files = [*train_proto_dir.rglob("*.proto"), *train_proto_dir.rglob("*.protos")]
    if not train_proto_files:
        return f"Error: no training protobuf shards found in {train_proto_dir}."
    split_manifest = {}
    manifest_path = split_root / "split.json"
    if manifest_path.is_file():
        try:
            split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            split_manifest = {}
    has_true_val = bool(split_manifest.get("val", 0)) and any(val_proto_dir.rglob("*.protos"))

    if not os.path.exists(os.path.join(FISH_MODELS_DIR, "model.pth")):
        _phase(progress, 0.48, "Downloading Fish S2 Pro training weights")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=FISH_MODELS_DIR)

    lora_config_name = f"run_{model_name}"
    lora_config_dir = Path(FS_DIR) / "fish_speech" / "configs" / "lora"
    lora_config_dir.mkdir(parents=True, exist_ok=True)
    dropout = max(0.0, min(0.5, float(lora_dropout)))
    with open(lora_config_dir / f"{lora_config_name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "_target_": "fish_speech.models.text2semantic.lora.LoraConfig",
            "r": int(lora_rank),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(dropout),
            "target_modules": ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"],
        }, f, sort_keys=False)

    bs, acc_grad, workers = _training_profile(vram_preset)
    project_dir = Path(FS_DIR) / "results" / model_name
    checkpoints_dir = project_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    resume_path = None
    fresh_training = not resume_checkpoint or str(resume_checkpoint) in {"None", "Fresh / None"}

    if not fresh_training:
        candidate = Path(str(resume_checkpoint))
        if not candidate.is_absolute():
            candidate = checkpoints_dir / str(resume_checkpoint)
        if candidate.is_file():
            resume_path = candidate.resolve()
        else:
            matches = list(project_dir.rglob(str(resume_checkpoint))) if project_dir.exists() else []
            if matches:
                resume_path = matches[0].resolve()
            else:
                return f"Error: selected resume checkpoint was not found: {resume_checkpoint}"

    if fresh_training:
        # "Fresh / None" is destructive by design: this project name starts a
        # completely new run. Keep the prepared dataset, but remove every
        # previous run artifact (checkpoints, TensorBoard events, evaluation
        # audio/config, Hydra metadata and any other files under results/<name>).
        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                print(f"[UI] Fresh training: deleted previous run directory: {project_dir}")
            except Exception as exc:
                return f"Error: could not clear previous run '{project_dir}': {exc}"
        project_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = project_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print("[UI] Fresh training selected; previous run deleted. Starting a clean run.")

    # Evaluation reference is stored as a pre-training contract.
    eval_dir = project_dir / "evaluation_audio"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_contract = project_dir / "evaluation.json"
    eval_reference = _select_training_eval_reference(dataset_dir, split_manifest) if bool(eval_enabled) else None
    eval_payload = {
        "text": str(eval_text or "This is a fixed evaluation sample generated during training."),
        "every_n_steps": int(evaluation_steps),
        "output_dir": str(eval_dir.resolve()),
        "audio_format": "wav",
        "seed": int(training_seed),
        "reference_audio_path": eval_reference["audio"] if eval_reference else None,
        "reference_text": eval_reference["text"] if eval_reference else None,
        "reference_stem": eval_reference["stem"] if eval_reference else None,
        "reference_source": eval_reference["source"] if eval_reference else "none",
    }
    if bool(eval_enabled):
        if eval_reference:
            print(
                f"[UI] Prompted evaluation reference: {eval_reference['stem']} "
                f"({eval_reference['source']}); fixed seed {int(training_seed)}.",
                flush=True,
            )
        else:
            print(
                "[WARNING] No usable audio+LAB pair was found for prompted evaluation; "
                "evaluation audio will fall back to deterministic no-prompt mode.",
                flush=True,
            )
        eval_contract.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif eval_contract.exists():
        eval_contract.unlink()

    train_proto_abs = str(train_proto_dir.resolve()).replace("\\", "/")
    val_proto_abs = str(val_proto_dir.resolve()).replace("\\", "/") if has_true_val else train_proto_abs
    ckpt_abs = str(Path(FISH_MODELS_DIR).resolve()).replace("\\", "/")
    command = [
        sys.executable, "fish_speech/train.py",
        "--config-name", "text2semantic_finetune",
        f"project={model_name}",
        f"+seed={int(training_seed)}",
        f"+deterministic=false",
        f"+lora@model.model.lora_config={lora_config_name}",
        f"trainer.max_steps={int(total_steps)}",
        "trainer.log_every_n_steps=1",
        f"trainer.accumulate_grad_batches={int(acc_grad)}",
        f"data.batch_size={int(bs)}",
        f"data.num_workers={int(workers)}",
        f"max_length={int(max_length)}",
        f"model.optimizer.lr={float(lr)}",
        f"model.optimizer.weight_decay={float(weight_decay)}",
        f"trainer.gradient_clip_val={float(gradient_clip)}",
        "trainer.gradient_clip_algorithm=norm",
        "trainer.strategy=auto",
        "trainer.devices=1",
        f"pretrained_ckpt_path={ckpt_abs}",
        f"train_dataset.proto_files=[{train_proto_abs}]",
        f"val_dataset.proto_files=[{val_proto_abs}]",
        f"train_dataset.causal={str(bool(causal)).lower()}",
        f"val_dataset.causal={str(bool(causal)).lower()}",
        f"train_dataset.interactive_prob={float(interactive_prob)}",
        f"val_dataset.interactive_prob={float(interactive_prob)}",
        f"trainer.val_check_interval={int(evaluation_steps) * int(acc_grad) if has_true_val else (int(total_steps) * int(acc_grad)) + 1}",
        f"trainer.limit_val_batches={10 if has_true_val else 0}",
        f"callbacks.model_checkpoint.every_n_train_steps={int(checkpoint_steps)}",
        "callbacks.model_checkpoint.save_last=True",
        "+callbacks.easy_gui_progress._target_=fish_speech.callbacks.easy_gui_progress.EasyGUIProgressCallback",
    ]
    # The public GUI exposes one reproducible Fast-AR schedule.  Normalize old
    # project state here as well, so a legacy saved value cannot silently
    # reactivate the former constant/warm-up path.
    if str(scheduler) != "Cosine 10% floor":
        print(
            f"[UI] Ignoring legacy scheduler '{scheduler}'; using the supported Fast-AR cosine schedule.",
            flush=True,
        )
    command.extend([
        "model.lr_scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR",
        f"model.lr_scheduler.T_max={int(total_steps)}",
        f"model.lr_scheduler.eta_min={float(lr) * 0.10}",
    ])

    if resume_path:
        command.append(f"+ckpt_path={str(resume_path).replace(chr(92), '/')}")
    else:
        # Explicit sentinel: train.py must not auto-discover last.ckpt when the
        # user selected Fresh / None. Existing checkpoints remain untouched.
        command.append("+ckpt_path=__FRESH__")

    env = os.environ.copy()
    env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "FISH_AUDIO_SAMPLE_CONFIG": str(eval_contract.resolve()) if bool(eval_enabled) else "",
        "FISH_EASY_SAVE_EVERY_STEPS": str(int(checkpoint_steps)),
        "FISH_EASY_EVAL_EVERY_STEPS": str(int(evaluation_steps)) if bool(eval_enabled) else "0",
    })
    _phase(progress, 0.62, f"Launching Fast-AR LoRA · {total_steps} optimizer steps")
    try:
        pid = _launch_training_process(command, env, model_name, total_steps)
    except Exception as exc:
        return f"Training launch failed: {exc}"
    start_mode = f"Resuming from {resume_path.name}." if resume_path else "Starting clean fresh training."
    return (
        f"{start_mode} Training started (PID {pid}). Target = {total_steps} optimizer steps. "
        f"Checkpoint every {checkpoint_steps} steps; evaluation audio every {evaluation_steps} steps. "
        "The HTML progress card, CMD and Live Console update while training runs."
    )


def analyze_dataset(folder, vram_preset):
    if not folder or folder == "(No datasets)":
        return "*Select a dataset to analyze.*", 250, 50, 50
    tuned = autotune_training(folder, vram_preset)
    return tuned["report"], tuned["steps"], tuned["save_steps"], tuned["eval_steps"]

def _lora_checkpoint_paths(model_name):
    """Return every Lightning LoRA checkpoint for a project, recursively.

    Lightning/custom checkpoint callbacks may place files in nested folders.
    Never collapse to basenames: relative paths keep duplicate checkpoint names
    unambiguous and make both Resume and Inference address the same file.
    """
    if not model_name:
        return []
    base = Path(FS_DIR) / "results" / str(model_name) / "checkpoints"
    if not base.is_dir():
        return []

    paths = [p for p in base.rglob("*.ckpt") if p.is_file()]

    def step_number(path):
        # Handles common Lightning names such as step=250.ckpt,
        # epoch=0-step=250.ckpt, step_250.ckpt, checkpoint-250.ckpt.
        matches = re.findall(r"(?:step|global_step)[=_-]?(\d+)", path.stem, flags=re.I)
        if matches:
            return int(matches[-1])
        nums = re.findall(r"(\d+)", path.stem)
        return int(nums[-1]) if nums else -1

    def sort_key(path):
        is_last = path.name.casefold() == "last.ckpt"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        # last.ckpt first, then highest step, then newest file/path.
        return (1 if is_last else 0, step_number(path), mtime, str(path).casefold())

    return sorted(paths, key=sort_key, reverse=True)


def handle_lora_list_checkpoints(model_name):
    """Fast-AR checkpoint values relative to results/<project>/checkpoints."""
    if not _project_uses_fast_lora(model_name):
        return []
    base = Path(FS_DIR) / "results" / str(model_name) / "checkpoints"
    results = []
    for path in _lora_checkpoint_paths(model_name):
        try:
            results.append(path.relative_to(base).as_posix())
        except ValueError:
            results.append(str(path.resolve()))
    return results


# Explicit LoRA merge/export controls were intentionally removed.  Inference
# consumes project checkpoints dynamically; s2.cpp adapters are materialized
# transparently by ensure_cpp_lora_adapter().

_sample_choices = get_sample_choices()
_default_sample = _sample_choices[0] if _sample_choices else None
_default_audio, _default_text = load_sample(_default_sample)



# Per-project TensorBoard ownership, matching the newest sister GUI behavior.
_tensorboard_process = None
_tensorboard_project = None
_tensorboard_port = None

def prepare_tensorboard_project(project):
    global _tensorboard_process, _tensorboard_project, _tensorboard_port
    target = str(project or "").strip() or None
    if _tensorboard_process is not None and _tensorboard_process.poll() is None and _tensorboard_project != target:
        try:
            _tensorboard_process.terminate(); _tensorboard_process.wait(timeout=3)
        except Exception:
            try: _tensorboard_process.kill()
            except Exception: pass
        _tensorboard_process = None
        _tensorboard_port = None
    _tensorboard_project = target
    return f"TensorBoard project selected: {target or 'None'}"

def launch_tensorboard_handler(model_name):
    global _tensorboard_process, _tensorboard_project, _tensorboard_port
    if not model_name:
        return "Select a training project before opening TensorBoard."
    import socket, webbrowser
    project = str(model_name)
    log_dir = Path(FS_DIR) / "results" / project / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    if _tensorboard_process is not None and _tensorboard_process.poll() is None:
        if _tensorboard_project == project:
            url=f"http://127.0.0.1:{_tensorboard_port}"
            webbrowser.open(url)
            return f"TensorBoard is already running for '{project}' at {url}."
        prepare_tensorboard_project(project)
    def available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try: sock.bind(("127.0.0.1", port)); return True
            except OSError: return False
    port=6006
    while not available(port): port += 1
    cmd=[sys.executable, "-m", "tensorboard.main", "--logdir_spec", f"{project}:{log_dir}", "--host", "127.0.0.1", "--port", str(port), "--reload_interval", "1"]
    flags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _tensorboard_process=subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    _tensorboard_project=project; _tensorboard_port=port
    time.sleep(0.5)
    url=f"http://127.0.0.1:{port}"; webbrowser.open(url)
    return f"TensorBoard started for '{project}' at {url} (PID {_tensorboard_process.pid})."


def restart_tensorboard_handler(model_name):
    """Restart only the TensorBoard instance owned by the selected training project."""
    global _tensorboard_process, _tensorboard_project, _tensorboard_port
    if not model_name:
        return "Select a training project before reloading TensorBoard."
    if _tensorboard_process is not None and _tensorboard_process.poll() is None:
        try:
            _tensorboard_process.terminate()
            _tensorboard_process.wait(timeout=3)
        except Exception:
            try:
                _tensorboard_process.kill()
            except Exception:
                pass
    _tensorboard_process = None
    _tensorboard_port = None
    _tensorboard_project = str(model_name)
    return launch_tensorboard_handler(model_name)
