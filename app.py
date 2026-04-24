import gradio as gr
import subprocess
import os
import time

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
import librosa
import soundfile as sf
import numpy as np
import glob
import yaml
import winsound
import sys

# Audio Chime Path
CHIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "inference_training_done.wav")

def play_done_chime():
    if os.path.exists(CHIME_PATH):
        try:
            winsound.PlaySound(CHIME_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Failed to play chime: {e}")
    else:
        # Fallback to system beep if file is missing
        winsound.MessageBeep()

# Main Paths
CPP_EXEC = os.path.join(ROOT_DIR, "modules", "s2.cpp", "build", "Release", "s2.exe")
if not os.path.exists(CPP_EXEC):
    CPP_EXEC = os.path.join(ROOT_DIR, "modules", "s2.cpp", "build", "s2.exe")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "modules", "s2.cpp", "tokenizer.json")

# Performance Optimizations for OpenMP (Threading Affinity)
os.environ["OMP_PROC_BIND"] = "TRUE"
os.environ["OMP_PLACES"] = "CORES"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"

s2_process = None
s2_current_model = None
training_process = None

# CPP server stdout drain thread & queue (module-level to avoid recreation/leak)
import queue as _queue
import threading as _threading
_s2_log_queue = _queue.Queue()
_s2_drain_thread = None

# Fish Python Persistence Cache
fish_python_model = None
fish_python_codec = None
fish_python_decode_one_token = None
fish_python_checkpoint_dir = None

# Model directories (organized per user request)
FISH_MODELS_DIR = os.path.join(MODELS_DIR, "S2")
S2_CPP_MODELS_DIR = os.path.join(MODELS_DIR, "s2.cpp")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")

for d in [OUTPUTS_DIR, MODELS_DIR, FISH_MODELS_DIR, S2_CPP_MODELS_DIR, SAMPLES_DIR, TRAINED_MODELS_DIR, WHISPER_MODELS_DIR, COMPILE_CACHE_DIR]:
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
    print(f"[Fish Speech] No persistent cache found (or incomplete: {_cache_kernel_count} kernels).")
    print("[Fish Speech] NOTICE: First PyTorch generation will compile kernels (~5 min). Subsequent runs will be fast.")
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
    "F16 [CUDA] (>12GB VRAM) ~14GB Studio Quality": "s2-pro-f16.gguf",
    "Q8_0 [CUDA] (≥8GB VRAM) ~8GB Best Balance": "s2-pro-q8_0.gguf",
    "Q6_K [Vulkan] (6-8GB VRAM) ~6GB Good Quality": "s2-pro-q6_k.gguf",
    "Q5_K_M [Vulkan] (4-6GB VRAM) ~5GB Balanced": "s2-pro-q5_k_m.gguf",
    "Q4_K_M [Vulkan/CPU] (3-4GB VRAM) ~4GB Decent": "s2-pro-q4_k_m.gguf",
    "Q3_K [Vulkan/CPU] ~3GB Low Quality": "s2-pro-q3_k.gguf",
    "Q2_K [Vulkan/CPU] <3GB Very Low Quality": "s2-pro-q2_k.gguf"
}
# GPU backend selection:
#   F16, Q8_0       -> CUDA (-c 0) — native CUDA get_rows support
#   Q6_K..Q4_K_M    -> Vulkan (-v 0) — k-quants need Vulkan backend
#   Q3_K, Q2_K      -> CPU only (no GPU flag)
CUDA_NATIVE_MODELS = {"s2-pro-f16.gguf", "s2-pro-q8_0.gguf"}

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

def get_sample_choices():
    if not os.path.exists(SAMPLES_DIR): return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
    return sorted([f.replace(".wav", "") for f in files])

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
        return audio_path, text
    return None, text

# --- Helper functions ---

def generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, model_select, progress):
    global fish_python_model, fish_python_codec, fish_python_decode_one_token, fish_python_checkpoint_dir
    from pathlib import Path
    
    target_dir = os.path.join(TRAINED_MODELS_DIR, model_select) if model_select and model_select != "Base Model (Fish S2 Pro)" else FISH_MODELS_DIR
        
    if fish_python_model is None or fish_python_checkpoint_dir != target_dir:
        import gc
        if fish_python_model is not None:
            del fish_python_model
            del fish_python_codec
            fish_python_model = None
            fish_python_codec = None
            gc.collect()
            torch.cuda.empty_cache()
            
        progress(0.1, desc=f"Loading PyTorch Model: {model_select}...")
        
        if target_dir == FISH_MODELS_DIR:
            from huggingface_hub import snapshot_download
            print("Checking Base Fish Speech Model...")
            target_dir = snapshot_download(repo_id="fishaudio/s2-pro", local_dir=FISH_MODELS_DIR)
            
        fish_python_checkpoint_dir = target_dir
        
        from fish_speech.models.text2semantic.inference import init_model, generate_long
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        progress(0.3, desc="Initializing model...")
        print("Initializing model...")
        fish_python_model, fish_python_decode_one_token = init_model(
            checkpoint_path=fish_python_checkpoint_dir,
            device=device,
            precision=precision,
            compile=False,
        )

        if device == "cuda":
            # Check for existing cache to provide feedback
            has_cache = False
            try:
                if os.path.exists(CACHE_DIR) and any(os.scandir(CACHE_DIR)):
                    has_cache = True
            except: pass

            msg = "Speed-up inference activated (Using cached kernels)..." if has_cache else "Optimizing model (torch.compile)..."
            if has_cache: print(f"🚀 {msg}")
            
            progress(0.4, desc=msg)
            try:
                print(f"Attempting torch.compile with mode='max-autotune' (Cache: {'Found' if has_cache else 'None'})...")
                fish_python_decode_one_token = torch.compile(
                    fish_python_decode_one_token, 
                    mode="max-autotune",
                    fullgraph=True
                )
                if has_cache:
                    print("Optimization level: max-autotune (Rapid start from cache)")
                else:
                    print("Optimization level: max-autotune (Wait for first inference to finish compilation)")
            except Exception as e:
                print(f"max-autotune is not supported on this environment: {e}. Falling back to 'reduce-overhead'...")
                try:
                    fish_python_decode_one_token = torch.compile(
                        fish_python_decode_one_token, 
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    print("Optimization level: reduce-overhead")
                except Exception as e2:
                    print(f"reduce-overhead also failed: {e2}. Proceeding without torch.compile.")
        
        progress(0.45, desc="Initializing codec...")
        print("Initializing codec...")
        fish_dir = Path(os.path.join(ROOT_DIR, "modules", "s2"))
        codec_cfg = OmegaConf.load(fish_dir / "fish_speech" / "configs" / "modded_dac_vq.yaml")
        fish_python_codec = instantiate(codec_cfg)
        
        codec_checkpoint_path = os.path.join(fish_python_checkpoint_dir, "codec.pth")
        if not os.path.exists(codec_checkpoint_path):
            codec_checkpoint_path = os.path.join(FISH_MODELS_DIR, "codec.pth")
            
        state_dict = torch.load(codec_checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }
        
        fish_python_codec.load_state_dict(state_dict, strict=False)
        fish_python_codec.eval()
        fish_python_codec.to(device=device, dtype=precision)
    else:
        print("Using Fish Speech Python model already in VRAM...")
        from fish_speech.models.text2semantic.inference import generate_long
        
    device = next(fish_python_model.parameters()).device
    model_dtype = next(fish_python_model.parameters()).dtype
    
    import librosa
    import numpy as np
    
    with torch.no_grad():
        progress(0.7, desc="Encoding audio reference...")
        wav_np, _ = librosa.load(ref_audio, sr=fish_python_codec.sample_rate, mono=True)
        wav = torch.from_numpy(wav_np).to(device)
        audios = wav[None, None, :].to(dtype=next(fish_python_codec.parameters()).dtype)
        audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)

        indices, feature_lengths = fish_python_codec.encode(audios, audio_lengths)
        prompt_tokens = [indices[0, :, : feature_lengths[0]].cpu()]
        
        # Split by paragraphs logic (optional)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if split_by_paragraph and len(paragraphs) > 1:
            import numpy as np
            print(f"Fish Speech chunking: {len(paragraphs)} paragraphs")
            audio_segments = []
            
            for idx, para in enumerate(paragraphs):
                progress_pct = 0.7 + (idx / len(paragraphs)) * 0.25
                progress(progress_pct, desc=f"Generating paragraph {idx + 1}/{len(paragraphs)}...")
                
                para_generator = generate_long(
                    model=fish_python_model,
                    device=device,
                    decode_one_token=fish_python_decode_one_token,
                    text=para,
                    num_samples=1,
                    max_new_tokens=int(len(para) * 4.5),
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temp,
                    repetition_penalty=rep_pen,
                    compile=False,
                    iterative_prompt=True,
                    chunk_length=200,
                    prompt_text=[ref_text] if ref_text else None,
                    prompt_tokens=prompt_tokens,
                )
                
                para_codes = []
                for response in para_generator:
                    if response.action == "sample":
                        para_codes.append(response.codes)
                    elif response.action == "next":
                        break
                
                if para_codes:
                    merged_para_codes = para_codes[0] if len(para_codes) == 1 else torch.cat(para_codes, dim=1)
                    merged_para_codes = merged_para_codes.to(device)
                    para_waveform = fish_python_codec.from_indices(merged_para_codes[None])
                    segment_np = para_waveform[0, 0].cpu().float().numpy()
                    audio_segments.append(segment_np)
                    
                    # Add 0.5 second of silence after each paragraph (except the last one)
                    if idx < len(paragraphs) - 0.5:
                        silence = np.zeros(int(fish_python_codec.sample_rate), dtype=np.float32)
                        audio_segments.append(silence)
            
            if audio_segments:
                audio_np = np.concatenate(audio_segments)
                sample_rate = fish_python_codec.sample_rate
                # Jump to cleanup
                goto_cleanup = True 
            else:
                raise RuntimeError("Fish Speech failed to generate any audio segments.")
        else:
            # Single paragraph or original logic
            progress(0.8, desc="Generating voice...")
            generator = generate_long(
                model=fish_python_model,
                device=device,
                decode_one_token=fish_python_decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=int(len(text) * 4.5),
                top_p=top_p,
                top_k=top_k,
                temperature=temp,
                repetition_penalty=rep_pen,
                compile=False,
                iterative_prompt=True,
                chunk_length=200,
                prompt_text=[ref_text] if ref_text else None,
                prompt_tokens=prompt_tokens,
            )
            
            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break
                    
            merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
            merged_codes = merged_codes.to(device)
            
            audio_waveform = fish_python_codec.from_indices(merged_codes[None])
            audio_waveform = audio_waveform[0, 0]
            audio_np = audio_waveform.cpu().float().numpy()
            sample_rate = fish_python_codec.sample_rate
        
    # Standard cleanup (GC) but NO UNLOADING of models
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
        
    # Apply fix for Gradio: convert to int16 before returning
    audio_int16 = (audio_np * 32767).astype(np.int16)
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

def clone_voice(engine, cpp_model_str, trained_model_select, text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, progress=gr.Progress()):
    global s2_process, s2_current_model

    if not text:
        return None, "Please enter some text to synthesize."
    if not ref_audio:
        return None, "Please upload a reference audio."
        
    timestamp = int(time.time() * 1000)
    out_wav = os.path.join(OUTPUTS_DIR, f"output_{timestamp}.wav")
    
    # Calculate auto tokens based on target text length dynamically
    expected_new_tokens = int(len(text) * 4.5)
    
    if engine == ENGINE_CPP:
        # Auto-Unload PyTorch if switching to CPP
        global fish_python_model, fish_python_codec
        if fish_python_model is not None:
            print("Auto-Unloading PyTorch Model to free VRAM for CPP...")
            del fish_python_model
            del fish_python_codec
            fish_python_model = None
            fish_python_codec = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        from huggingface_hub import hf_hub_download
        
        filename = GGUF_MODELS.get(cpp_model_str)
        if not filename:
             return None, "Invalid GGUF model selected."
        
        progress(0.2, desc=f"Checking GGUF model: {filename}")
        print(f"Checking/Downloading {filename}...")
        try:
            model_path = hf_hub_download(
                repo_id="rodrigomt/s2-pro-gguf",
                filename=filename,
                local_dir=S2_CPP_MODELS_DIR,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            return None, f"Failed to download GGUF model: {e}"
        
        # Start server if not running with the same model
        if s2_process is None or s2_current_model != filename:
            if s2_process is not None:
                s2_process.kill()
                s2_process.wait()  # Block until dead
                s2_process = None
                time.sleep(1.5)  # Wait for Windows TIME_WAIT to release port 3030
                
            progress(0.3, desc="Starting Fish CPP Server...")
            # Optimized thread count for modern CPUs (P-cores/E-cores and high core counts)
            # Use physical cores if available (psutil), otherwise logical/2 for efficiency.
            try:
                import psutil
                threads = psutil.cpu_count(logical=False) or (os.cpu_count() // 2) or 4
            except:
                threads = (os.cpu_count() // 2) if (os.cpu_count() and os.cpu_count() > 8) else (os.cpu_count() or 4)

            cmd = [
                CPP_EXEC,
                "-m", model_path,
                "-t", TOKENIZER_PATH,
                "--server",
                "-threads", str(threads),
            ]
            if "CPU ONLY" not in cpp_model_str:
                if filename in CUDA_NATIVE_MODELS:
                    cmd.extend(["-c", "0"])  # CUDA for F16/Q8_0
                else:
                    cmd.extend(["-v", "0"])  # Vulkan for k-quants

            # Fix PATH correctly for s2.exe to find cublas64_##.dll
            env = os.environ.copy()
            path_key = "PATH" if "PATH" in env else ("Path" if "Path" in env else "PATH")
            
            # 1. Start with mandatory app paths
            s2_dir_path = os.path.dirname(CPP_EXEC)
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
                
            s2_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                startupinfo=startupinfo
            )
            s2_current_model = filename
            
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
                        print(f"[s2.exe] {line}")
                        captured_logs.append(line)

                # Check if process died
                if s2_process.poll() is not None:
                    last_msg = "\n".join(captured_logs[-5:]) if captured_logs else "No logs captured."
                    s2_process = None
                    s2_current_model = None
                    return None, f"Fish CPP Engine crashed during startup.\n\nErrors:\n{last_msg}"

                # Health-check the HTTP endpoint
                try:
                    requests.get("http://localhost:3030/", timeout=0.5)
                    ready = True
                    break
                except Exception:
                    pass

                time.sleep(0.5)

            if not ready:
                return None, "Fish CPP server timed out during startup."
            
            # Extra wait: server binds port before model is fully loaded into VRAM
            time.sleep(2)
                
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # --- Handle Multiple Paragraphs (Split by Paragraph enabled) ---
        if split_by_paragraph and len(paragraphs) > 1:
            print(f"[s2.cpp] Processing {len(paragraphs)} paragraphs (Split by Paragraph: Active)...")
            all_audio_segments = []
            final_sr = 32000 # Default fallback, normally detected from model

            for idx, para in enumerate(paragraphs):
                # Calculate tokens for THIS paragraph
                para_tokens = int(len(para) * 4.5)
                progress_pct = 0.7 + (idx / len(paragraphs)) * 0.25
                progress(progress_pct, desc=f"Synthesizing paragraph {idx + 1}/{len(paragraphs)}...")
                
                # Retry loop for THIS paragraph
                current_audio = None
                max_para_retries = 2
                for att in range(max_para_retries):
                    try:
                        with open(ref_audio, 'rb') as f:
                            files = {'reference_audio': (os.path.basename(ref_audio), f, 'audio/wav')}
                            data = {
                                'text': para,
                                'ref_text': ref_text,
                                'params': json.dumps({'max_new_tokens': para_tokens, 'temperature': temp, 'top_p': top_p, 'top_k': top_k, 'repetition_penalty': rep_pen, 'verbose': True})
                            }
                            res = requests.post("http://localhost:3030/generate", data=data, files=files, timeout=600)
                            res.raise_for_status()
                            
                        import soundfile as sf
                        audio_data, sr = sf.read(io.BytesIO(res.content))
                        final_sr = sr
                        # Store segment as float32 for clean concatenation
                        all_audio_segments.append(audio_data.astype(np.float32))
                        
                        # Add a tiny bit of silence (0.5s) between paragraphs? optional but usually good
                        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
                        all_audio_segments.append(silence)
                        
                        current_audio = True
                        break # Success for this para
                    except Exception as ecc:
                        print(f"Error in paragraph {idx+1} (attempt {att+1}): {ecc}")
                        time.sleep(1)
                
                if current_audio is None:
                    return None, f"Failed to generate paragraph {idx+1} after multiple attempts."
            
            # Concatenate all segments
            if all_audio_segments:
                combined_audio = np.concatenate(all_audio_segments)
                
                # Convert to mono and normalize
                combined_audio = process_audio_array(combined_audio)
                
                # Convert to int16 for Gradio consistency
                audio_int16 = (combined_audio * 32767).astype(np.int16)
                import soundfile as sf
                sf.write(out_wav, audio_int16, final_sr)
                
                import gc
                del all_audio_segments, combined_audio, audio_int16
                gc.collect()
                
                play_done_chime()
                progress(1.0, desc="Done!")
                return out_wav, "Synthesis completed successfully (multi-paragraph)!"
            else:
                return None, "No audio segments were generated correctly."

        # --- Standard Single-Pass Logic (Used if 1 paragraph or split disabled) ---
        # Retry loop: server may reset connections while finishing VRAM allocation
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                with open(ref_audio, 'rb') as f:
                    files = {'reference_audio': (os.path.basename(ref_audio), f, 'audio/wav')}
                    data = {
                        'text': text,
                        'ref_text': ref_text,
                        'params': json.dumps({'max_new_tokens': expected_new_tokens, 'temperature': temp, 'top_p': top_p, 'top_k': top_k, 'repetition_penalty': rep_pen, 'verbose': True})
                    }
                    res = requests.post("http://localhost:3030/generate", data=data, files=files, timeout=600)
                    res.raise_for_status()
                    
                import soundfile as sf
                audio_data, sr = sf.read(io.BytesIO(res.content))
                
                # Convert to mono and normalize
                audio_data = process_audio_array(audio_data)
                
                # Apply fix for Gradio: convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                sf.write(out_wav, audio_int16, sr)
                # Explicit GC after each generation to prevent RAM growth
                import gc
                del audio_data, audio_int16
                gc.collect()
                play_done_chime()
                progress(1.0, desc="Done!")
                return out_wav, "Synthesis completed successfully!"
            except (ConnectionError, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[s2.cpp] Connection reset (attempt {attempt+1}/{max_retries}), retrying in 2s...")
                    time.sleep(2)
                    continue
            except Exception as e:
                last_error = e
                break
        
        error_msg = str(last_error)
        if s2_process and s2_process.poll() is not None:
            crash_logs = []
            try:
                while True:
                    line = _s2_log_queue.get_nowait()
                    if line: crash_logs.append(line.strip())
            except Exception: pass
            last_logs = "\n".join(crash_logs[-10:])
            error_msg = f"Crash Detected (Exit code {s2_process.returncode}).\nLogs:\n{last_logs}\n\nOriginal Error: {error_msg}"
            
        return None, f"s2.cpp REST API Error:\n{error_msg}"
            
    elif engine == ENGINE_PYTORCH:
        # Auto-Unload CPP Server if switching to PyTorch
        if s2_process is not None:
            print("Auto-Unloading CPP Server to free VRAM for PyTorch...")
            s2_process.kill()
            s2_process = None
            s2_current_model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            import soundfile as sf
            # Cache-aware progress notice
            if HAS_COMPILE_CACHE:
                progress(0.05, desc="Loading PyTorch model (kernels cached — fast start)...")
            else:
                progress(0.05, desc="[First Run] Compiling CUDA kernels... up to 5 min. See console.")
                print("[Fish Speech] NOTICE: torch.compile is building kernels for the first time. This may take up to 5 minutes.")
                print("[Fish Speech] Subsequent generations will be significantly faster.")
            sr, audio_int16 = generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, trained_model_select, progress)
            
            # Convert to float32 to process, then back
            audio_data = audio_int16.astype(np.float32) / 32767.0
            audio_data = process_audio_array(audio_data)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            progress(0.9, desc="Saving audio...")
            sf.write(out_wav, audio_int16, sr)
            play_done_chime()
            
            # Clean VRAM after logic
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            return out_wav, "Synthesis completed."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error generating with PyTorch: {str(e)}"

    return None, "Engine not supported."

def generate_dialogue(engine, cpp_model_str, trained_model_select, top_p, top_k, temp, rep_pen, split_para, row_count, silence_duration, *args, progress=gr.Progress()):
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
        return None, "Please add at least one speaker and text."
        
    all_audio_segments = []
    final_sr = 32000
    
    for i, (sample_name, text) in enumerate(segments):
        progress((i / len(segments)), desc=f"Processing segment {i+1}/{len(segments)} ({sample_name})...")
        
        # Load sample
        ref_audio, ref_text = load_sample(sample_name)
        if not ref_audio:
            print(f"Sample {sample_name} not found, skipping segment {i+1}")
            continue
            
        # Generate
        wav_path, status = clone_voice(
            engine, cpp_model_str, trained_model_select, text, ref_audio, ref_text, 
            top_p, top_k, temp, rep_pen, split_para, progress=progress
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
            return None, f"Error in segment {i+1} ({sample_name}): {status}"
            
    if all_audio_segments:
        # Concatenate (exclude last silence if added)
        if silence_duration > 0 and len(all_audio_segments) > 1:
            combined = np.concatenate(all_audio_segments[:-1])
        else:
            combined = np.concatenate(all_audio_segments)
            
        combined = process_audio_array(combined)
            
        # Output file
        out_wav = os.path.join(OUTPUTS_DIR, f"dialogue_{int(time.time()*1000)}.wav")
        # Gradio fix: int16
        audio_int16 = (combined * 32767).astype(np.int16)
        sf.write(out_wav, audio_int16, final_sr)
        return out_wav, f"Dialogue generated successfully with {len(segments)} segments!"
    
    return None, "No audio generated."

def transcribe_only(audio_path, model_size, language_name, progress=gr.Progress()):
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "Error: faster-whisper module not found. Please run install.ps1 again."
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lang_code = WHISPER_LANGS.get(language_name)
    
    try:
        progress(0.2, desc=f"Loading Faster-Whisper {model_size}...")
        print(f"Loading Faster-Whisper {model_size} via {device}...")
        
        whisper_cache = os.path.join(MODELS_DIR, "whisper")
        os.makedirs(whisper_cache, exist_ok=True)
        
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=whisper_cache)
        
        progress(0.5, desc="Transcribing audio...")
        segments, info = model.transcribe(audio_path, language=lang_code, beam_size=5)
        
        # Gather all text segments
        text = " ".join([segment.text for segment in segments]).strip()
        
        # Unload from VRAM after transcription is done
        import gc
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        progress(1.0, desc="Done!")
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Faster-Whisper Error: {str(e)}"

def handle_full_batch_process(source_folder, dataset_name, model_size, language_name, batch_size, progress=gr.Progress()):
    if not source_folder or not os.path.isdir(source_folder):
        return "Error: Please provide a valid source folder path."
    if not dataset_name or dataset_name.strip() == "":
        return "Error: Please provide a target dataset name."

    import glob
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]:
        audio_files.extend(glob.glob(os.path.join(source_folder, ext)))
        audio_files.extend(glob.glob(os.path.join(source_folder, ext.upper())))
    
    if not audio_files:
        return "Error: No audio files found in the source folder."

    # 1. Create target directory in datasets/
    target_dir = os.path.join(TRAINING_DATA_DIR, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    
    total = len(audio_files)
    processed = 0
    metadata = []

    # 2. Audio Processing & Copying (Multi-threaded)
    progress(0.1, desc="Processing audios (Multi-threaded Normalize + Mono)...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_audio_file(audio_path):
        filename = os.path.basename(audio_path)
        dest_audio = os.path.join(target_dir, filename)
        dest_audio_wav = os.path.splitext(dest_audio)[0] + ".wav"
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
            sf.write(dest_audio_wav, y, sr)
            return True, filename
        except Exception as e:
            return False, f"{filename}: {e}"

    futures = []
    with ThreadPoolExecutor() as executor:
        for audio_path in audio_files:
            futures.append(executor.submit(process_audio_file, audio_path))
            
    completed = 0
    for future in as_completed(futures):
        success, result = future.result()
        if success:
            processed += 1
        else:
            print(f"Error processing: {result}")
        completed += 1
        progress(0.1 + (0.3 * (completed/total)), desc=f"Processed {completed}/{total} audios")

    if processed == 0:
        return "Error: Failed to process any audio files."

    # 3. Transcription using Faster-Whisper
    progress(0.4, desc=f"Loading Faster-Whisper {model_size}...")
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lang_code = WHISPER_LANGS.get(language_name)
        whisper_cache = os.path.join(MODELS_DIR, "whisper")
        os.makedirs(whisper_cache, exist_ok=True)
        
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=whisper_cache)
        batched_model = BatchedInferencePipeline(model=model)
        
        target_files = glob.glob(os.path.join(target_dir, "*.wav"))
        for i, audio_path in enumerate(target_files):
            filename = os.path.basename(audio_path)
            progress(0.5 + (0.4 * (i/len(target_files))), desc=f"Transcribing {i+1}/{len(target_files)}: {filename}")
            
            try:
                segments, info = batched_model.transcribe(audio_path, language=lang_code, batch_size=int(batch_size))
                
                # Gather all text segments
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Save .lab file
                lab_path = os.path.splitext(audio_path)[0] + ".lab"
                with open(lab_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Add to metadata
                metadata.append(f"{os.path.abspath(audio_path)}|{text}")
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
                
        # Cleanup Whisper
        del batched_model
        del model
        import gc
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Audio processed but Transcription failed: {str(e)}"

    # 4. Generate metadata.csv for Fish Speech
    if metadata:
        metadata_path = os.path.join(target_dir, "metadata.csv")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("audio_file|text\n")
            f.write("\n".join(metadata))

    progress(1.0, desc="Done!")
    return (f"✨ Success! Processed {processed} files.\n"
            f"📍 Location: datasets/{dataset_name}\n"
            f"✅ Actions: Normalized, Mono, Faster-Whisper Transcribed, metadata.csv generated.")
    
    moved_count = 0
    metadata = []
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        lab_path = os.path.splitext(audio_path)[0] + ".lab"
        
        # Check if .lab exists
        if not os.path.exists(lab_path):
            continue
            
        with open(lab_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        dest_audio = os.path.join(target_dir, filename)
        shutil.copy2(audio_path, dest_audio)
        
        # In the training pipeline, it expects metadata.csv with: audio_file|text
        # We'll use absolute path or relative? prepare_dataset.py uses Path(row["audio_file"])
        # If we put metadata.csv in the same folder, relative should work.
        metadata.append(f"{dest_audio}|{text}")
        moved_count += 1
        
    # Write metadata.csv
    metadata_path = os.path.join(target_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        f.write("\n".join(metadata))
        
    progress(1.0, desc="Done!")
    return f"Successfully copied {moved_count} samples to datasets folder: {target_dir}.\nGenerated metadata.csv."

def fix_audio_single(audio_path, normalize=True, to_mono=True):
    if not audio_path or not os.path.exists(audio_path):
        return audio_path, "Error: File not found."
        
    try:
        # Load audio (to_mono handles mono conversion)
        y, sr = librosa.load(audio_path, sr=None, mono=to_mono)
        
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
            # Overwrite logic
            y, sr = librosa.load(audio_path, sr=None, mono=to_mono)
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

def save_prep_sample(audio_path, sample_name, transcription):
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
            "Text": transcription.strip()
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
        return f"Deleted sample '{sample_name}'.", gr.update(choices=get_sample_choices(), value=None)
    except Exception as e:
        return f"Error deleting: {e}", gr.update()

# --- Training Backend Functions ---

def run_training_step(cmd, desc, progress):
    global training_process
    progress(0.1, desc=f"Starting {desc}... (Check the console for the progress)")
    import subprocess
    
    # Run from FS_DIR to ensure hydra and relative paths work
    # Use CREATE_NEW_PROCESS_GROUP to allow sending CTRL_BREAK_EVENT for graceful save on Windows
    training_process = subprocess.Popen(
        cmd,
        cwd=FS_DIR if "python" in cmd else ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True if os.name == 'nt' else False,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    logs = []
    
    # Use a local reference to avoid race conditions when the process is stopped/reset from UI
    p = training_process
    if p:
        for line in iter(p.stdout.readline, ""):
            line = line.strip()
            if line:
                print(f"[{desc}] {line}")
                logs.append(line)
        
        p.wait()
        ret_code = p.returncode
    else:
        ret_code = 0 # Assume finished or aborted cleanly if none
        
    training_process = None # Reset global on finish
    
    if ret_code != 0 and ret_code != -1: # -1 might be manual kill
        return False, f"{desc} failed with return code {ret_code}.\n\nLast logs:\n" + "\n".join(logs[-10:])
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

def handle_lora_sharding(output_name, progress=gr.Progress()):
    data_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    proto_dir = os.path.join(data_dir, "protos")
    os.makedirs(proto_dir, exist_ok=True)
    
    cmd = f"\"{sys.executable}\" tools/llama/build_dataset.py --input \"{data_dir}\" --output \"{proto_dir}\" --text-extension .lab --num-workers 4"
    success, msg = run_training_step(cmd, "Sharding", progress)
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
        f"model.lr_scheduler.T_max={max_steps} "
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

def handle_lora_unified(output_name, model_name, max_steps, lr, vram_preset, lora_rank=32, lora_alpha=16, save_every=50, progress=gr.Progress()):
    msg_log = []
    
    msg_log.append("--- Step 1: Dataset Preparation ---")
    progress(0.1, desc="Dataset Preparation...")
    msg = handle_lora_dataset_prep(output_name, progress)
    msg_log.append(msg)
    if "failed" in msg.lower(): return "\n".join(msg_log)
    
    msg_log.append("--- Step 2: VQ Code Extraction ---")
    progress(0.3, desc="Extracting VQ Codes...")
    msg = handle_lora_vq_extraction(output_name, progress)
    msg_log.append(msg)
    if "failed" in msg.lower(): return "\n".join(msg_log)
    
    msg_log.append("--- Step 3: Protobuf Sharding ---")
    progress(0.5, desc="Sharding Dataset...")
    msg = handle_lora_sharding(output_name, progress)
    msg_log.append(msg)
    if "failed" in msg.lower(): return "\n".join(msg_log)
    
    msg_log.append("--- Step 4: LoRA Training ---")
    progress(0.7, desc="Starting LoRA Training...")
    
    # Overwrite logic (Always fresh start for stability)
    proj_dir = os.path.join(FS_DIR, "results", model_name)
    if os.path.exists(proj_dir):
         import shutil
         progress(0.71, desc="Wiping existing project for a fresh start...")
         try:
             shutil.rmtree(proj_dir)
         except Exception as e:
             print(f"Warning: Could not clear existing project: {e}")

    dataset_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    proto_dir = os.path.join(dataset_dir, "protos")
    if not os.path.exists(proto_dir):
        msg_log.append(f"Sharded data not found at {proto_dir}. Sharding might have failed silently.")
        return "\n".join(msg_log)
        
    if not model_name:
        model_name = f"{output_name}_{int(time.time())}"
    
    if not os.path.exists(os.path.join(FISH_MODELS_DIR, "model.pth")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="fishaudio/s2-pro", local_dir=FISH_MODELS_DIR)

    proto_dir_abs = os.path.abspath(proto_dir).replace("\\", "/")
    ckpt_dir_abs = os.path.abspath(FISH_MODELS_DIR).replace("\\", "/")
    
    # We will pass lora parameters inside a json file or directly via override.
    # To be safe, we'll keep using the built-in fast_attention lora config but modify r and alpha if possible.
    # Let's write a dynamic config file inside fish_speech.
    lora_config_name = f"run_{model_name}"
    lora_config_dir = os.path.join(FS_DIR, "fish_speech", "configs", "lora")
    os.makedirs(lora_config_dir, exist_ok=True)
    lora_config_path = os.path.join(lora_config_dir, f"{lora_config_name}.yaml")
    
    import yaml
    lora_config = {
        "_target_": "fish_speech.models.text2semantic.lora.LoraConfig",
        "r": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": 0.05,
        "target_modules": ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"]
    }
    with open(lora_config_path, 'w') as f:
        yaml.dump(lora_config, f, default_flow_style=False)
        
    # VRAM Presets Setup
    if "+32GB VRAM" in vram_preset:
        bs = 2
        acc_grad = 2
    else:  # 24GB VRAM Default Profile
        bs = 1
        acc_grad = 4
        
    ckpt_dir_save = os.path.abspath(os.path.join(FS_DIR, "results", model_name, "checkpoints")).replace("\\", "/")
    os.makedirs(ckpt_dir_save, exist_ok=True)

    # End-to-end training (Resume disabled for stability)
    resume_flags = ""

    cmd = (
        f"set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"\"{sys.executable}\" fish_speech/train.py "
        f"--config-name text2semantic_finetune "
        f"project={model_name} "
        f"{resume_flags}"
        f"+lora@model.model.lora_config={lora_config_name} "
        f"trainer.max_steps={int(max_steps)} "
        f"model.lr_scheduler.T_max={int(max_steps)} "
        f"trainer.accumulate_grad_batches={acc_grad} "
        f"data.batch_size={bs} "
        f"model.optimizer.lr={float(lr)} "
        f"trainer.strategy=auto "
        f"trainer.devices=1 "
        f"data.num_workers=0 "
        f"pretrained_ckpt_path=\"{ckpt_dir_abs}\" "
        f"train_dataset.proto_files=[{proto_dir_abs}] "
        f"val_dataset.proto_files=[{proto_dir_abs}] "
        f"~callbacks.audio_sample "
        f"trainer.val_check_interval={int(save_every)} "
        f"callbacks.model_checkpoint.every_n_train_steps={int(save_every)} "
        f"callbacks.model_checkpoint.save_last=True "
        f"model.optimizer.weight_decay=0.01"
    )
    
    success, msg = run_training_step(cmd, "LoRA Training", progress)
    msg_log.append(msg)
    
    if success:
        play_done_chime()
        msg_log.append("--- Training Completed Successfully! ---")
        msg_log.append("To use your LoRA model, specify it in the Inference dropdown once exported.")
        
    return "\n".join(msg_log) + "\n\n(Check the console for the full history of the training)"

def analyze_dataset(folder, vram_preset):
    if not folder or folder == "(No datasets)":
        return "*Select a dataset to analyze*", 1000
    
    dataset_dir = os.path.join(TRAINING_DATA_DIR, folder)
    if not os.path.exists(dataset_dir):
        return f"**Dataset {folder} not found.**", 1000
        
    wav_files = glob.glob(os.path.join(dataset_dir, "*.wav"))
    if not wav_files:
        return f"**0 .wav files found in {folder}.**", 1000
        
    total_duration = 0.0
    for f in wav_files:
        try:
            info = sf.info(f)
            total_duration += info.frames / info.samplerate
        except Exception: pass
        
    num_samples = len(wav_files)
    mins = int(total_duration // 60)
    secs = int(total_duration % 60)
    
    # Hardware Presets
    if "+32GB VRAM" in vram_preset:
        bs = 2
        acc_grad = 2
    else:
        bs = 1
        acc_grad = 4
        
    effective_batch = bs * acc_grad
    steps_per_epoch = max(1, num_samples // effective_batch)
    target_steps = max(50, min(1500, steps_per_epoch * 3))
    
    report = f"**Dataset Analytics:**\n"
    report += f"- **Samples:** {num_samples} audio files\n"
    report += f"- **Duration:** {mins}m {secs}s total\n"
    report += f"\n**Auto-Tuned Params (3 Epochs):**\n"
    report += f"- Target Steps -> **{target_steps}**\n"
    report += f"- Profile `{vram_preset}` -> `batch_size={bs}`, `acc_grad={acc_grad}`"
    
    return report, target_steps

def handle_lora_list_checkpoints(model_name):
    if not model_name:
        return []
    ckpt_base = os.path.join(FS_DIR, "results", model_name, "checkpoints")
    if not os.path.exists(ckpt_base):
        return []
    import glob
    ckpts = glob.glob(os.path.join(ckpt_base, "*.ckpt"))
    # Always include last.ckpt if it exists, and sort it to the top
    ckpts_basenames = [os.path.basename(c) for c in ckpts]
    results = sorted([c for c in ckpts_basenames if c != "last.ckpt"], reverse=True)
    if "last.ckpt" in ckpts_basenames:
        results = ["last.ckpt"] + results
    return results

def launch_tensorboard_handler(model_name):
    if not model_name:
        return "Please enter a Model Name first."
    
    log_dir = os.path.join(FS_DIR, "results", model_name, "tensorboard")
    if not os.path.exists(log_dir):
        # Check if the parent results dir exists to avoid confusion
        parent_dir = os.path.join(FS_DIR, "results", model_name)
        if not os.path.exists(parent_dir):
            return f"Model directory '{model_name}' not found. Start training first."
        os.makedirs(log_dir, exist_ok=True)

    import socket
    def get_free_port(start):
        for port in range(start, start + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return start

    port = get_free_port(6006)
    cmd = f"tensorboard --logdir \"{log_dir}\" --port {port}"
    
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        
    subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags
    )
    
    # Give Tensorboard a moment to start before opening the browser
    time.sleep(5.0)
    
    # Auto-open browser
    import webbrowser
    url = f"http://localhost:{port}"
    try:
        webbrowser.open(url)
    except: pass
    
    return f"Tensorboard launched for '{model_name}' at {url} (Opened in browser)"

def handle_lora_export(model_name, ckpt_name=None, fs_lora_rank=32, fs_lora_alpha=16, progress=gr.Progress()):
    if not model_name:
        return "Error: Please enter a Model Name to export."
    
    # Auto-detect latest if ckpt_name is None
    if ckpt_name is None:
        # Check for last.ckpt first, then highest step
        ckpts = handle_lora_list_checkpoints(model_name)
        if not ckpts:
            return "No checkpoints found to export."
        
        if "last.ckpt" in ckpts:
            ckpt_name = "last.ckpt"
        else:
            ckpt_name = ckpts[0] # handle_lora_list_checkpoints returns sorted reverse=True
        
    progress(0.1, desc=f"Preparing Export for {ckpt_name}...")
    ckpt_path = os.path.join(FS_DIR, "results", model_name, "checkpoints", ckpt_name)
    
    # We always export to the TRAINED_MODELS_DIR now
    output_dir = os.path.join(TRAINED_MODELS_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge LoRA script
    base_weight = os.path.abspath(FISH_MODELS_DIR).replace("\\", "/") # Path to base model inside models/fish-speech
    ckpt_path_abs = os.path.abspath(ckpt_path).replace("\\", "/")
    output_dir_abs = os.path.abspath(output_dir).replace("\\", "/")
    
    # Try to use the exact config used during training for this project
    # This prevents rank mismatch if the user changed sliders after training
    project_config_name = f"run_{model_name}"
    project_config_path = os.path.join(FS_DIR, "fish_speech", "configs", "lora", f"{project_config_name}.yaml")
    
    if os.path.exists(project_config_path):
        lora_config_name = project_config_name
        print(f"Using project-specific LoRA config: {lora_config_name}")
    else:
        # Fallback to manual construction from sliders (for legacy or external models)
        lora_config_name = f"r_{fs_lora_rank}_alpha_{fs_lora_alpha}"
        if fs_lora_rank == 32 and fs_lora_alpha == 16:
            lora_config_name = "r_32_alpha_16_fast"
        print(f"Project config not found, using manual LoRA config: {lora_config_name}")
        
    cmd = (
        f"\"{sys.executable}\" tools/llama/merge_lora.py "
        f"--lora-config {lora_config_name} "
        f"--base-weight \"{base_weight}\" "
        f"--lora-weight \"{ckpt_path_abs}\" "
        f"--output \"{output_dir_abs}\""
    )
    
    success, msg = run_training_step(cmd, "Merging LoRA to Base Model", progress)
    if not success:
        return f"failed: {msg}"
        
    # Copy essential codec and topology files required for inference
    progress(0.8, desc="Copying inference topologies...")
    import shutil
    for file_to_copy in ["codec.pth", "tokenizer.json", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"]:
        src = os.path.join(FISH_MODELS_DIR, file_to_copy)
        dst = os.path.join(output_dir, file_to_copy)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    return f"✅ Export Complete!\n\nYour model is now ready at:\n`{output_dir}`"

# --- Pre-calculate defaults before building UI (VoxCPM pattern) ---
# This avoids any app.load() calls that block the browser on startup.
_sample_choices = get_sample_choices()
_default_sample = _sample_choices[0] if _sample_choices else None
_default_audio, _default_text = load_sample(_default_sample)

CUSTOM_CSS = """
.green-btn { background-color: #28a745 !important; color: white !important; border: none !important; }
.red-btn { background-color: #dc3545 !important; color: white !important; border: none !important; }
"""

with gr.Blocks(title="Fish Speech S2 Pro - Voice Clone & Training GUI") as app:
    with gr.Row():
        with gr.Column(scale=20):
            gr.Markdown("""
                # 🎙️ Fish Speech S2 Pro - Voice Clone & Training GUI
                <p style="font-size: 0.9em; color: var(--body-text-color-subdued); margin-top: -10px;">Powered by Fish Speech S2 Pro & Gradio</p>
            """)
        with gr.Column(scale=1, min_width=180):
            unload_all_btn = gr.Button("Clear VRAM", size="sm", variant="secondary")
            unload_status = gr.Markdown(" ", visible=True)
            
            def clear_vram():
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "VRAM freed."
            def clear_vram_msg():
                time.sleep(2)
                return " "
            unload_all_btn.click(clear_vram, outputs=[unload_status]).then(clear_vram_msg, outputs=[unload_status])

    def add_dialogue_row_at(index, count, *args):
        num = 20
        samples = list(args[:num])
        texts = list(args[num:])
        if count < num:
            samples.insert(index + 1, samples[index])
            texts.insert(index + 1, "")
            samples.pop()
            texts.pop()
            count += 1
        
        update_samples = [gr.update(value=samples[i], visible=(i < count)) for i in range(num)]
        update_texts = [gr.update(value=texts[i], visible=(i < count)) for i in range(num)]
        update_rows = [gr.update(visible=(i < count)) for i in range(num)]
        return [count] + update_samples + update_texts + update_rows

    def rem_dialogue_row_at(index, count, *args):
        num = 20
        samples = list(args[:num])
        texts = list(args[num:])
        if count > 1:
            samples.pop(index)
            texts.pop(index)
            samples.append(None)
            texts.append("")
            count -= 1
            
        update_samples = [gr.update(value=samples[i], visible=(i < count)) for i in range(num)]
        update_texts = [gr.update(value=texts[i], visible=(i < count)) for i in range(num)]
        update_rows = [gr.update(visible=(i < count)) for i in range(num)]
        return [count] + update_samples + update_texts + update_rows

    def clone_dialogue_row_at(index, count, *args):
        num = 20
        samples = list(args[:num])
        texts = list(args[num:])
        if count < num:
            samples.insert(index + 1, samples[index])
            texts.insert(index + 1, texts[index])
            samples.pop()
            texts.pop()
            count += 1
            
        update_samples = [gr.update(value=samples[i], visible=(i < count)) for i in range(num)]
        update_texts = [gr.update(value=texts[i], visible=(i < count)) for i in range(num)]
        update_rows = [gr.update(visible=(i < count)) for i in range(num)]
        return [count] + update_samples + update_texts + update_rows

    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab("Voice Clone", id="tab_voice_clone"):
            gr.Markdown("Clone Voices from Samples. <small>(Use Prep Samples to add samples)</small>")
            
            def update_engine_ui(engine):
                is_cpp = (engine == ENGINE_CPP)
                if is_cpp:
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            def update_split_count(text):
                if not text: return "### ✂️ Splits\n**0** Clips"
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                count = len(paragraphs)
                return f"### ✂️ Splits\n**{count}** {'Clip' if count == 1 else 'Clips'}"

            # --- Global Settings at Top ---
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Inference & Models")
                    engine_dropdown = gr.Dropdown(
                        choices=[ENGINE_CPP, ENGINE_PYTORCH],
                        value=ENGINE_CPP,
                        label="Inference Engine"
                    )
                    with gr.Row():
                        cpp_model_row = gr.Row(visible=True)
                        with cpp_model_row:
                            cpp_model_dropdown = gr.Dropdown(
                                choices=list(GGUF_MODELS.keys()), 
                                label="GGUF Model", 
                                value=list(GGUF_MODELS.keys())[1]
                            )
                        trained_model_row = gr.Row(visible=False)
                        with trained_model_row:
                            trained_model_dropdown = gr.Dropdown(
                                choices=get_trained_models(),
                                label="Trained LoRA Model",
                                value="Base Model (Fish S2 Pro)",
                                scale=10
                            )
                            trained_model_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ Advanced Settings")
                    with gr.Row():
                        with gr.Column():
                            top_p_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Top-P")
                            top_k_slider = gr.Slider(1, 100, value=30, step=1, label="Top-K")
                        with gr.Column():
                            temperature_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                            rep_pen_slider = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
                    
                    with gr.Row():
                        with gr.Column():
                            split_para_check = gr.Checkbox(label="Split by Paragraphs (Recommended for long texts)", value=False)
                            gr.Markdown("ℹ️ *To apply splits, you must press **Enter** after each sentence or point where you want a cut; each line break will generate an independent audio clip that will be automatically merged.*")
                        with gr.Column():
                            dialogue_silence_slider = gr.Slider(0, 5, value=0.5, step=0.1, label="Silence between speakers (s)")
                            if HAS_COMPILE_CACHE:
                                gr.Markdown(f"✅ **Cache Found:** ({_cache_kernel_count} kernels)")
                            else:
                                gr.Markdown(f"⚠️ **No Cache:** First PyTorch run ~5 min.")
            
            with gr.Accordion("ℹ️ Supported Generation Tags & Tips", open=False):
                gr.Markdown("""
                **Supported Generation Tags**
                S2 Pro enables localized control over speech generation by embedding natural-language instructions directly within the text using `[tag]` syntax. Rather than relying on a fixed set of predefined tags, S2 Pro accepts free-form textual descriptions - such as `[whisper in small voice]`, `[professional broadcast tone]`, or `[pitch up]` - allowing open-ended expression control at the word level.
                
                **Common tags (15,000+ unique tags supported):**
                `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]` and much more...
                """)
            
            with gr.Accordion("🌐 Supported Languages", open=False):
                gr.Markdown("""
                **S2 Pro supports 80+ languages.**
                *   **Tier 1:** Japanese (ja), English (en), Chinese (zh)
                *   **Tier 2:** Korean (ko), Spanish (es), Portuguese (pt), Arabic (ar), Russian (ru), French (fr), German (de)
                *   **Other supported languages:** sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, xsl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo, and more.
                """)

            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Voice Sample")
                            with gr.Row():
                                vc_sample_dropdown = gr.Dropdown(
                                    choices=_sample_choices,
                                    value=_default_sample,
                                    label="Select Sample",
                                    interactive=True,
                                    scale=10
                                )
                                vc_sample_refresh_btn = gr.Button("🔄", scale=1, min_width=50)
                            vc_sample_audio = gr.Audio(label="Sample Preview", type="filepath", interactive=False, elem_id="sample-audio-player", value=_default_audio)
                            vc_sample_text = gr.Textbox(label="Sample Text", interactive=False, max_lines=10, value=_default_text)

                        with gr.Column(scale=2):
                            gr.Markdown("### Generate Speech")
                            with gr.Row():
                                target_text = gr.Textbox(
                                    label="Text to Generate",
                                    placeholder="Enter text to speak...",
                                    lines=6,
                                    scale=10
                                )
                                with gr.Column(scale=1, min_width=100):
                                    split_counter_display = gr.Markdown("### ✂️ Splits\n**1** Clip", elem_id="split-counter", visible=False)

                            with gr.Row():
                                generate_btn = gr.Button("Generate Audio 🚀", variant="primary", size="lg")
                            
                            with gr.Row():
                                output_audio = gr.Audio(label="Generated Audio", type="filepath")
                            
                            with gr.Row():
                                clone_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("### 💬 Multi-Speaker Dialogue Builder")
                    dialogue_segments = []
                    MAX_DIALOGUE_SEGMENTS = 20
                    
                    with gr.Column():
                        for i in range(MAX_DIALOGUE_SEGMENTS):
                            with gr.Row(visible=(i < 2)) as row:
                                s = gr.Dropdown(choices=_sample_choices, label=f"Speaker {i+1}", scale=3, value=_default_sample if i < 2 else None)
                                t = gr.Textbox(placeholder=f"Enter text for speaker {i+1}...", label=f"Text {i+1}", scale=7, lines=6)
                                with gr.Row():
                                    add_btn = gr.Button("➕", variant="secondary", size="sm", elem_classes=["green-btn"])
                                    clone_btn = gr.Button("📋", variant="secondary", size="sm")
                                    rem_btn = gr.Button("🗑️", variant="stop", size="sm", elem_classes=["red-btn"])
                                
                                dialogue_segments.append({
                                    "row": row, 
                                    "sample": s, 
                                    "text": t,
                                    "add": add_btn,
                                    "clone": clone_btn,
                                    "rem": rem_btn
                                })
                        
                        dialogue_row_count = gr.State(2)
                        
                        with gr.Row():
                            generate_dialogue_btn = gr.Button("Generate Dialogue 🚀", variant="primary", size="lg")
                        
                        with gr.Row():
                            dialogue_output_audio = gr.Audio(label="Generated Dialogue", type="filepath")
                        
                        with gr.Row():
                            dialogue_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    

        with gr.Tab("Prep Samples", id="tab_prep_samples"):
            gr.Markdown("Prepare audio samples for voice cloning.")
            with gr.Row():
                with gr.Column(scale=1) as prep_sidebar:
                    with gr.Group() as audio_samples_group:
                        gr.Markdown("### Audio Samples")
                        prep_sample_dropdown = gr.Dropdown(
                            choices=_sample_choices,
                            value=_default_sample,
                            label="Select Sample",
                            interactive=True
                        )
                        with gr.Row():
                            delete_btn = gr.Button("Delete", size="sm", variant="stop")
                    
                    gr.Markdown("### Transcription Settings")
                    with gr.Row():
                        whisper_model_size = gr.Dropdown(
                            choices=["large-v3", "large-v2", "medium", "small", "base"], 
                            value="large-v3", 
                            label="Whisper Model Size",
                            scale=1
                        )
                        whisper_language = gr.Dropdown(
                            choices=list(WHISPER_LANGS.keys()),
                            value="Auto-detect",
                            label="Language",
                            scale=1
                        )
                    
                with gr.Column(scale=2):
                    with gr.Tabs() as prep_tabs:
                        with gr.Tab("Single Editor"):
                            gr.Markdown("""
                            ### 🎙️ Add or Edit Audio 
                            Use the **'X'** (top right of the player) to clear the preview and drag or click to upload a new audio. 
                            *Once uploaded, click **Transcribe** to get the text, then **Save Sample** to add it to your library.*
                            """)
                            prep_audio_editor = gr.Audio(label="Audio Editor (Use Trim icon to edit)", type="filepath", interactive=True, value=_default_audio)
                            
                            gr.Markdown("### Reference Text")
                            transcription_output = gr.Textbox(
                                label="Text",
                                lines=4,
                                max_lines=10,
                                interactive=True,
                                placeholder="Transcription will appear here, or enter/edit text manually...",
                                value=_default_text
                            )
                            
                            with gr.Row():
                                transcribe_btn = gr.Button("Transcribe Audio", variant="primary", scale=1)
                                norm_single_btn = gr.Button("Normalize Volume", scale=1)
                                mono_single_btn = gr.Button("Convert to Mono", scale=1)
                            
                            with gr.Row():
                                save_name_input = gr.Textbox(label="Sample Name", placeholder="e.g. my_new_voice", scale=2)
                                save_btn = gr.Button("Save Sample", variant="primary", scale=1)
                        
                        with gr.Tab("Dataset Creation"):
                            gr.Markdown("### 📂 Dataset Creation for Training")
                            with gr.Row():
                                batch_folder_input = gr.Textbox(label="Source Audio Folder", placeholder="C:\\path\\to\\your\\audio\\files", scale=4)
                                
                                explorer_btn = gr.Button("📂 Browse", size="sm", scale=1)
                                def open_folder_explorer():
                                    import tkinter as tk
                                    from tkinter import filedialog
                                    root = tk.Tk()
                                    root.attributes('-topmost', 1)
                                    root.withdraw()
                                    path = filedialog.askdirectory(title="Select Source Audio Folder")
                                    root.destroy()
                                    return path if path else gr.update()
                                    
                                explorer_btn.click(fn=open_folder_explorer, inputs=[], outputs=[batch_folder_input])
                                
                            with gr.Row():
                                batch_dataset_name = gr.Textbox(label="Target Dataset Name (Subfolder in datasets/)", value="my_voice", placeholder="e.g. my_voice_v1", scale=2)
                                faster_whisper_batch = gr.Slider(1, 32, value=16, step=1, label="Faster Whisper Batch Size", scale=2)
                                batch_process_btn = gr.Button("🚀 Process & Transcribe All", variant="primary", scale=1)
                            
                            gr.Markdown("""
                            <p style="font-size: 0.85em; color: gray;">
                            * This will: <b>Copy</b> audios to <code>datasets/</code> → <b>Multi-thread Normalize</b> → <b>Mono Convert</b> → <b>Faster-Whisper Batched Transcribe (.lab)</b> → <b>Link everything</b>.
                            </p>
                            """)
                            batch_status = gr.Textbox(label="Batch Status Console", lines=6, interactive=False)

                    prep_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    
                    def show_samples_group():
                        return gr.update(visible=True)
                    def hide_samples_group():
                        return gr.update(visible=False)

                    prep_tab_single = prep_tabs.children[0]
                    prep_tab_dataset = prep_tabs.children[1]

                    def on_prep_sample_select(sample_name):
                        """load_sample now returns (path_or_None, text_str) directly."""
                        if not sample_name:
                            return None, ""
                        return load_sample(sample_name)


                    # 2. Main Generation Logic
                    engine_dropdown.change(
                        fn=update_engine_ui,
                        inputs=engine_dropdown,
                        outputs=[cpp_model_row, trained_model_row]
                    )

                    target_text.change(
                        fn=update_split_count,
                        inputs=[target_text],
                        outputs=[split_counter_display]
                    )

                    split_para_check.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[split_para_check],
                        outputs=[split_counter_display]
                    )

                    generate_btn.click(
                        fn=clone_voice,
                        inputs=[engine_dropdown, cpp_model_dropdown, trained_model_dropdown, target_text, vc_sample_audio, vc_sample_text, top_p_slider, top_k_slider, temperature_slider, rep_pen_slider, split_para_check],
                        outputs=[output_audio, clone_status]
                    )

                    # Dialogue Builder Handlers
                    all_samples_ui = [d["sample"] for d in dialogue_segments]
                    all_texts_ui = [d["text"] for d in dialogue_segments]
                    all_rows_ui = [d["row"] for d in dialogue_segments]

                    for i, d in enumerate(dialogue_segments):
                        d["add"].click(
                            fn=add_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count, *all_samples_ui, *all_texts_ui],
                            outputs=[dialogue_row_count] + all_samples_ui + all_texts_ui + all_rows_ui
                        )
                        d["rem"].click(
                            fn=rem_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count, *all_samples_ui, *all_texts_ui],
                            outputs=[dialogue_row_count] + all_samples_ui + all_texts_ui + all_rows_ui
                        )
                        d["clone"].click(
                            fn=clone_dialogue_row_at,
                            inputs=[gr.State(i), dialogue_row_count, *all_samples_ui, *all_texts_ui],
                            outputs=[dialogue_row_count] + all_samples_ui + all_texts_ui + all_rows_ui
                        )

                    generate_dialogue_btn.click(
                        fn=generate_dialogue,
                        inputs=[
                            engine_dropdown, cpp_model_dropdown, trained_model_dropdown, 
                            top_p_slider, top_k_slider, temperature_slider, rep_pen_slider, split_para_check,
                            dialogue_row_count, dialogue_silence_slider,
                            *all_samples_ui,
                            *all_texts_ui
                        ],
                        outputs=[dialogue_output_audio, dialogue_status]
                    )

                    # 3. Sample Selection & Management
                    vc_sample_dropdown.change(
                        fn=load_sample,
                        inputs=[vc_sample_dropdown],
                        outputs=[vc_sample_audio, vc_sample_text]
                    )

                    prep_sample_dropdown.change(
                        fn=on_prep_sample_select,
                        inputs=[prep_sample_dropdown],
                        outputs=[prep_audio_editor, transcription_output]
                    )

                    def refresh_all_samples():
                        choices = get_sample_choices()
                        updates = [gr.update(choices=choices)] * (1 + MAX_DIALOGUE_SEGMENTS)
                        return updates

                    vc_sample_refresh_btn.click(
                        fn=refresh_all_samples,
                        inputs=None,
                        outputs=[vc_sample_dropdown] + [d["sample"] for d in dialogue_segments]
                    )

                    trained_model_refresh_btn.click(
                        fn=lambda: gr.update(choices=get_trained_models()),
                        inputs=None,
                        outputs=[trained_model_dropdown]
                    )

                    prep_audio_editor.clear(
                        fn=lambda: ("", ""),
                        inputs=[],
                        outputs=[transcription_output, save_name_input]
                    )

                    transcribe_btn.click(
                        fn=transcribe_only,
                        inputs=[prep_audio_editor, whisper_model_size, whisper_language],
                        outputs=[transcription_output]
                    )

                    norm_single_btn.click(
                        fn=lambda x: fix_audio_single(x, normalize=True, to_mono=False),
                        inputs=[prep_audio_editor],
                        outputs=[prep_audio_editor, prep_status]
                    )

                    mono_single_btn.click(
                        fn=lambda x: fix_audio_single(x, normalize=False, to_mono=True),
                        inputs=[prep_audio_editor],
                        outputs=[prep_audio_editor, prep_status]
                    )

                    batch_process_btn.click(
                        fn=handle_full_batch_process,
                        inputs=[batch_folder_input, batch_dataset_name, whisper_model_size, whisper_language, faster_whisper_batch],
                        outputs=[batch_status]
                    )

                    save_btn.click(
                        fn=save_prep_sample,
                        inputs=[prep_audio_editor, save_name_input, transcription_output],
                        outputs=[prep_status, prep_sample_dropdown]
                    ).then(
                        fn=lambda: gr.update(choices=get_sample_choices()),
                        inputs=[], outputs=[vc_sample_dropdown]
                    )

                    prep_tab_single.select(fn=show_samples_group, inputs=[], outputs=[audio_samples_group])
                    prep_tab_dataset.select(fn=hide_samples_group, inputs=[], outputs=[audio_samples_group])

        with gr.Tab("Lora Training (Experimental)", id="tab_lora_training"):
            gr.Markdown("### 🏋️ Fish Speech S2 Pro LoRA Training Pipeline")
            
            gr.Markdown("""
---
### 🧪 **Notice: LoRA Fine-Tuning is Experimental**
*Fish Speech S2 PRO* is a highly-tuned foundation model. LoRA training might not show significant improvements for small or standard datasets. However, it can make a noticeable difference when:
- Working with **extremely large datasets**.
- Teaching the model a **new language**, unique **accent**, or specific **dialect**.
- Fine-tuning for **style-specific** speech patterns.

⚠️ **Note:** Training is computationally intensive and exclusive to GPUs with **more than 24 GB of VRAM**.
---
""")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🗂️ Dataset Selection & Auto-Tune")
                    with gr.Row():
                        training_output_name = gr.Dropdown(label="Dataset Folder", choices=get_dataset_choices(), value=get_dataset_choices()[0] if get_dataset_choices() else None, scale=4)
                        refresh_folders_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                        
                    vram_preset_radio = gr.Radio(["24GB VRAM", "+32GB VRAM"], label="Hardware Preset", value="24GB VRAM")
                    analyze_btn = gr.Button("📊 Analyze & Auto-Tune Dataset", variant="secondary")
                    
                    dataset_info = gr.Markdown("*Select a dataset and preset, then click Analyze.*")
                    
                    train_quick_guide = """
**Fish Speech S2 PRO Training Guide:**
1. Use the "Dataset Creation" tab to prepare your dataset.
2. Select your Dataset Folder and your Hardware Preset.
3. Click **Analyze Dataset** to auto-tune max steps.
4. Click **Start Auto-Training**.
"""
                    gr.Markdown(train_quick_guide)
                        
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Training Configuration")
                    with gr.Accordion("Training Settings", open=True):
                        with gr.Row():
                            model_name_input = gr.Dropdown(
                                label="Trained Model Name", 
                                choices=get_existing_training_projects(),
                                allow_custom_value=True,
                                info="Select existing project or type a new name",
                                scale=8
                            )
                            refresh_models_btn = gr.Button("🔄", scale=1, min_width=50)
                        
                        with gr.Row():
                            train_max_steps = gr.Slider(50, 5000, value=1000, step=50, label="Max Steps", info="Auto-calculated")
                            train_lr = gr.Number(value=1e-5, label="Learning Rate")
                            
                        with gr.Row():
                            fs_lora_rank = gr.Slider(minimum=4, maximum=64, value=32, step=4, label="LoRA Rank (r)")
                            fs_lora_alpha = gr.Slider(minimum=4, maximum=64, value=16, step=4, label="LoRA Alpha")
                            adv_save_every = gr.Slider(minimum=10, maximum=500, value=50, step=10, label="Save Every N Steps")
                                
                        with gr.Row():
                            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg", scale=3)
                            stop_train_btn = gr.Button("🛑 Stop Training", variant="stop", size="lg", scale=1)
                        
                        with gr.Row():
                            tensorboard_btn = gr.Button("📊 Launch Tensorboard", variant="secondary", size="lg")
                        
                        with gr.Group():
                            gr.Markdown("#### 📦 Export LoRA Model")
                            with gr.Row():
                                export_ckpt_select = gr.Dropdown(
                                    label="Available Checkpoints", 
                                    choices=[], 
                                    info="Select which step/checkpoint to export",
                                    scale=8
                                )
                                export_refresh_ckpts = gr.Button("🔄", scale=1, min_width=50)
                                export_btn = gr.Button("📦 Convert & Export", variant="secondary", scale=4)
                        
                        with gr.Row():
                            clear_results_btn = gr.Button("🗑️ Clear All LoRA Results", variant="stop", size="sm")
                        
                        gr.Markdown("""
### ⚠️ **LoRA Results & Cleanup Information:**

1. **Model Name Dropdown:** This selector is used to browse existing projects for **Exporting LoRA Models** or **Launching Tensorboard**. 
2. **Auto-Overwrite:** Clicking **🚀 Start Training** will **completely wipe** the selected model's results folder before starting a fresh, end-to-end training. To avoid losing intermediate checkpoints, choose a NEW model name.
3. **Clear All LoRA Results:** This button deletes **all** intermediate checkpoints and logs from `modules/s2/results`. 
4. **Exported Models:** None of the above will delete your **final exported models** stored in `models/trained_models` (the ones you use for inference).
""")
                        
                    training_status = gr.Textbox(label="Status Console", lines=10, interactive=False)
                    
                    # Automatic Step 3: Result info (No manual buttons)
                    model_path_display = gr.Markdown("")

            # --- Training Actions ---
            refresh_folders_btn.click(fn=lambda: gr.update(choices=get_dataset_choices()), outputs=[training_output_name])
            
            analyze_btn.click(analyze_dataset, [training_output_name, vram_preset_radio], [dataset_info, train_max_steps])
            
            train_btn.click(
                handle_lora_unified, 
                [training_output_name, model_name_input, train_max_steps, train_lr, vram_preset_radio, fs_lora_rank, fs_lora_alpha, adv_save_every], 
                training_status
            )

            tensorboard_btn.click(
                fn=launch_tensorboard_handler,
                inputs=[model_name_input],
                outputs=[training_status]
            )

            def stop_training_handler():
                global training_process
                if training_process:
                    import os, signal
                    try:
                        # Send CTRL_BREAK_EVENT to let Lightning save last.ckpt and exit
                        print(f"Stopping training gracefully (PID {training_process.pid})...")
                        os.kill(training_process.pid, signal.CTRL_BREAK_EVENT)
                        return "Graceful stop signal sent. Checkpoint saving in progress... (You can use 'Clear VRAM' if needed)"
                    except Exception as e:
                        return f"Error stopping process: {str(e)}"
                return "No training process currently active."

            def update_ckpts(mn):
                ckpts = handle_lora_list_checkpoints(mn)
                if ckpts:
                    return gr.update(choices=ckpts, value=ckpts[0])
                return gr.update(choices=[], value=None)

            model_name_input.change(fn=update_ckpts, inputs=[model_name_input], outputs=[export_ckpt_select])
            export_refresh_ckpts.click(fn=update_ckpts, inputs=[model_name_input], outputs=[export_ckpt_select])

            stop_train_btn.click(fn=stop_training_handler, outputs=[training_status])

            export_btn.click(
                fn=handle_lora_export, 
                inputs=[model_name_input, export_ckpt_select, fs_lora_rank, fs_lora_alpha], 
                outputs=[training_status]
            )

            refresh_models_btn.click(
                fn=lambda: gr.update(choices=get_existing_training_projects()),
                outputs=[model_name_input]
            )

            clear_results_btn.click(fn=handle_clear_results, outputs=[training_status]).then(
                fn=lambda: gr.update(choices=get_existing_training_projects()),
                outputs=[model_name_input]
            )

if __name__ == "__main__":
    # Use "127.0.0.1" for local access or "0.0.0.0" for network access
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CUSTOM_CSS)