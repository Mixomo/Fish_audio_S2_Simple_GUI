import gradio as gr
import subprocess
import os
import time
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
if os.name == 'nt':
    import winsound
else:
    winsound = None


# Audio Chime Path
CHIME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "inference_training_done.wav")

def play_done_chime():
    if not os.path.exists(CHIME_PATH):
        if os.name != 'nt': print("\a", end="")
        return

    try:
        import shutil
        # Try common Linux audio players in order
        players = [
            ["pw-play", CHIME_PATH],
            ["paplay", CHIME_PATH],
            ["aplay", CHIME_PATH],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", CHIME_PATH],
        ]
        for player_cmd in players:
            if shutil.which(player_cmd[0]):
                subprocess.Popen(player_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        
        if os.name != 'nt': print("\a", end="") # Terminal bell fallback
    except Exception as e:
        print(f"Failed to play chime: {e}")


# Main Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_EXEC = os.path.join(ROOT_DIR, "modules", "s2.cpp", "build", "Release", "s2.exe")
if not os.path.exists(CPP_EXEC):
    CPP_EXEC = os.path.join(ROOT_DIR, "modules", "s2.cpp", "build", "s2.exe")
# Linux Fallback
if not os.path.exists(CPP_EXEC):
    CPP_EXEC = os.path.join(ROOT_DIR, "modules", "s2.cpp", "build", "s2")

TOKENIZER_PATH = os.path.join(ROOT_DIR, "modules", "s2.cpp", "tokenizer.json")

s2_process = None
s2_current_model = None

# Fish Python Persistence Cache
fish_python_model = None
fish_python_codec = None
fish_python_decode_one_token = None
fish_python_checkpoint_dir = None

# Model directories (organized per user request)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FISH_MODELS_DIR = os.path.join(MODELS_DIR, "S2")
S2_CPP_MODELS_DIR = os.path.join(MODELS_DIR, "s2.cpp")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")

for d in [OUTPUTS_DIR, MODELS_DIR, FISH_MODELS_DIR, S2_CPP_MODELS_DIR, SAMPLES_DIR, TRAINED_MODELS_DIR, WHISPER_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

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
    "Q6_K [Vulkan - Not Supported under WSL2] (6-8GB VRAM) ~6GB Good Quality": "s2-pro-q6_k.gguf",
    "Q5_K_M [Vulkan - Not Supported under WSL2] (4-6GB VRAM) ~5GB Balanced": "s2-pro-q5_k_m.gguf",
    "Q4_K_M [Vulkan/CPU - Not Supported under WSL2] (3-4GB VRAM) ~4GB Decent": "s2-pro-q4_k_m.gguf",
    "Q3_K [Vulkan/CPU - Not Supported under WSL2] ~3GB Low Quality": "s2-pro-q3_k.gguf",
    "Q2_K [Vulkan/CPU - Not Supported under WSL2] <3GB Very Low Quality": "s2-pro-q2_k.gguf"
}
# GPU backend selection:
#   F16, Q8_0       -> CUDA (-c 0) — native CUDA get_rows support
#   Q6_K..Q4_K_M    -> Vulkan (-v 0) — k-quants need Vulkan backend
#   Q3_K, Q2_K      -> CPU only (no GPU flag)
CUDA_NATIVE_MODELS = {"s2-pro-f16.gguf", "s2-pro-q8_0.gguf"}

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
    trained_dir = os.path.join(MODELS_DIR, "trained_models")
    if not os.path.exists(trained_dir): return base
    subdirs = [d for d in os.listdir(trained_dir) if os.path.isdir(os.path.join(trained_dir, d))]
    return base + sorted(subdirs)

def load_sample(sample_name):
    if not sample_name:
        return gr.update(value=None), gr.update(value="")
    audio_path = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    txt_path = os.path.join(SAMPLES_DIR, f"{sample_name}.txt")
    json_path = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
    
    text = ""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("Text", "")
        except: pass
    
    if not text and os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
    if os.path.exists(audio_path):
        return gr.update(value=audio_path), gr.update(value=text)
    return gr.update(value=None), gr.update(value="")

# --- Helper functions ---

def generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, model_select, progress):
    global fish_python_model, fish_python_codec, fish_python_decode_one_token, fish_python_checkpoint_dir
    from pathlib import Path
    
    target_dir = os.path.join(MODELS_DIR, "trained_models", model_select) if model_select and model_select != "Base Model (Fish S2 Pro)" else FISH_MODELS_DIR
    
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
        # Set torch compile options for Max Autotune (User Requirement)
        if torch.cuda.is_available():
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.fx_graph_cache = True # Speed up repeat recompiles
            torch._inductor.config.max_autotune = True # Max Autotune mode
        
        fish_python_model, fish_python_decode_one_token = init_model(
            checkpoint_path=fish_python_checkpoint_dir,
            device=device,
            precision=precision,
            compile=True, # Enable torch.compile
        )


            
        progress(0.5, desc="Initializing codec...")
        print("Initializing codec...")
        fish_dir = Path(os.path.join(ROOT_DIR, "modules", "s2"))
        codec_cfg = OmegaConf.load(fish_dir / "fish_speech" / "configs" / "modded_dac_vq.yaml")
        fish_python_codec = instantiate(codec_cfg)
        
        codec_checkpoint_path = os.path.join(fish_python_checkpoint_dir, "codec.pth")
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
                    compile=True, # Enable torch.compile

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
                compile=True, # Enable torch.compile

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
        
    return audio_np, sample_rate

def clone_voice(engine, cpp_model_str, trained_model_select, text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, progress=gr.Progress()):
    global s2_process, s2_current_model

    if not text:
        return None, "Please enter some text to synthesize."
    if not ref_audio:
        return None, "Please upload a reference audio."
        
    timestamp = int(time.time())
    out_wav = os.path.join(OUTPUTS_DIR, f"output_{timestamp}.wav")
    
    # Calculate auto tokens based on target text length dynamically
    expected_new_tokens = int(len(text) * 4.5)
    
    if engine == "Fish Speech S2 Pro (CPP)":
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
            # Use physical cores only (not logical/HT) for better GGUF performance
            try:
                import psutil
                cpu_physical = psutil.cpu_count(logical=False) or os.cpu_count() or 4
            except ImportError:
                cpu_physical = os.cpu_count() or 4
            # Use all physical cores for inference threads; batch threads = same
            cpu_total = cpu_physical
            cmd = [
                CPP_EXEC,
                "--model", model_path,
                "--tokenizer", TOKENIZER_PATH,
                "--server",
                "--threads", str(cpu_total),
                "--threads-batch", str(cpu_total),
            ]
            if "CPU ONLY" not in cpp_model_str:
                if filename in CUDA_NATIVE_MODELS:
                    cmd.extend(["-c", "0"])  # CUDA for F16/Q8_0
                else:
                    cmd.extend(["-v", "0"])  # Vulkan for k-quants (Q6_K, Q5_K_M, Q4_K_M)

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
            if os.name == 'nt':
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
            else:
                # Standard Linux CUDA paths
                linux_paths = ["/usr/local/cuda/bin", "/usr/bin"]
                for lp in linux_paths:
                    if os.path.exists(lp):
                        extra_paths.append(lp)


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
            
            # Wait for server ready — use a background thread to drain stdout
            # so readline() never blocks the main thread
            import queue
            import threading

            log_queue = queue.Queue()
            captured_logs = []

            def _drain_stdout(proc, q):
                try:
                    for line in proc.stdout:
                        q.put(line)
                except Exception:
                    pass
                finally:
                    q.put(None)  # sentinel

            drain_thread = threading.Thread(target=_drain_stdout, args=(s2_process, log_queue), daemon=True)
            drain_thread.start()

            start_time = time.time()
            ready = False
            print("--- Starting s2.exe logs ---")

            while time.time() - start_time < 300:
                # Drain all log lines currently available (non-blocking)
                while True:
                    try:
                        line = log_queue.get_nowait()
                    except queue.Empty:
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
                
        progress(0.7, desc="Synthesizing audio...")
        
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
                sf.write(out_wav, audio_data, sr)
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
                    line = log_queue.get_nowait()
                    if line: crash_logs.append(line.strip())
            except Exception: pass
            last_logs = "\n".join(crash_logs[-10:])
            error_msg = f"Crash Detected (Exit code {s2_process.returncode}).\nLogs:\n{last_logs}\n\nOriginal Error: {error_msg}"
            
        return None, f"s2.cpp REST API Error:\n{error_msg}"
            
    elif engine == "Fish Speech S2 Pro (PyTorch)":
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
            audio_np, sr = generate_fish_python(text, ref_audio, ref_text, top_p, top_k, temp, rep_pen, split_by_paragraph, trained_model_select, progress=progress)
            progress(0.9, desc="Saving audio...")
            sf.write(out_wav, audio_np, sr)
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
    progress(0.1, desc=f"Starting {desc}... (Check the console for the progress)")
    import subprocess
    
    # Run from FS_DIR to ensure hydra and relative paths work
    process = subprocess.Popen(
        cmd,
        cwd=FS_DIR if ("python" in cmd.lower() or sys.executable in cmd) else ROOT_DIR,

        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True # Enable shell for string commands on all platforms
    )

    
    logs = []
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if line:
            print(f"[{desc}] {line}")
            logs.append(line)
    
    process.wait()
    if process.returncode != 0:
        return False, f"{desc} failed with return code {process.returncode}.\n\nLast logs:\n" + "\n".join(logs[-10:])
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

    import sys
    cmd = f'"{sys.executable}" tools/vqgan/extract_vq.py "{data_dir}" --config-name modded_dac_vq --checkpoint-path "{codec_path}" --num-workers 1 --batch-size 1'

    success, msg = run_training_step(cmd, "VQ Extraction", progress)
    return msg

def handle_lora_sharding(output_name, progress=gr.Progress()):
    data_dir = os.path.join(TRAINING_DATA_DIR, output_name)
    proto_dir = os.path.join(data_dir, "protos")
    os.makedirs(proto_dir, exist_ok=True)
    
    import sys
    cmd = f'"{sys.executable}" tools/llama/build_dataset.py --input "{data_dir}" --output "{proto_dir}" --text-extension .lab --num-workers 4'

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
    
    import sys
    env_setter = "set" if os.name == 'nt' else "export"
    cmd = (
        f"{env_setter} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f'"{sys.executable}" fish_speech/train.py '

        f"--config-name text2semantic_finetune "
        f"project={model_name} "
        f"+lora@model.model.lora_config=r_32_alpha_16_fast "
        f"trainer.max_steps={max_steps} "
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

def handle_lora_unified(output_name, model_name, max_steps, lr, vram_preset, lora_rank=32, lora_alpha=16, progress=gr.Progress()):
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

    env_setter = "set" if os.name == 'nt' else "export"
    cmd = (
        f"{env_setter} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f'"{sys.executable}" fish_speech/train.py '
        f"--config-name text2semantic_finetune "
        f"project={model_name} "
        f"+lora@model.model.lora_config={lora_config_name} "
        f"trainer.max_steps={int(max_steps)} "
        f"trainer.accumulate_grad_batches={acc_grad} "
        f"data.batch_size={bs} "
        f"model.optimizer.lr={float(lr)} "
        f"trainer.strategy=auto "
        f"trainer.devices=1 "
        f"data.num_workers=0 "
        f"pretrained_ckpt_path=\"{ckpt_dir_abs}\" "
        f"train_dataset.proto_files=[{proto_dir_abs}] "
        f"val_dataset.proto_files=[{proto_dir_abs}] "
        f"callbacks.model_checkpoint.dirpath=\"{ckpt_dir_save}\" "
        f"callbacks.model_checkpoint.every_n_train_steps=50 "
        f"callbacks.model_checkpoint.save_last=True"
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

def handle_lora_export(model_name, ckpt_name=None, fs_lora_rank=32, fs_lora_alpha=16, progress=gr.Progress()):
    if not model_name:
        return "Model name missing."
        
    if not ckpt_name:
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
    output_dir = os.path.join(MODELS_DIR, "trained_models", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge LoRA script
    base_weight = os.path.abspath(FISH_MODELS_DIR).replace("\\", "/") # Path to base model inside models/fish-speech
    ckpt_path_abs = os.path.abspath(ckpt_path).replace("\\", "/")
    output_dir_abs = os.path.abspath(output_dir).replace("\\", "/")
    
    lora_config_name = f"r_{fs_lora_rank}_alpha_{fs_lora_alpha}"
    if fs_lora_rank == 32 and fs_lora_alpha == 16:
        lora_config_name = "r_32_alpha_16_fast"
        
    cmd = (
        f"uv run python tools/llama/merge_lora.py "
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
            
    return f"✅ Export Complete!\n\nYour trained S2 Model is now ready for use at:\n`{output_dir}`"

theme = gr.themes.Ocean(
    neutral_hue=gr.themes.Color(c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af", c50="#f9fafb", c500="#6b7280", c600="hsl(215, 7%, 34%)", c700="hsl(217, 10%, 27%)", c800="hsl(215, 14%, 17%)", c900="hsl(221, 20%, 11%)", c950="hsl(223, 20%, 7%)"),
    spacing_size=gr.themes.Size(lg="6px", md="4px", sm="2px", xl="9px", xs="1px", xxl="10px", xxs="1px"),
    primary_hue="orange",
    secondary_hue="red",
    text_size="lg",
    radius_size="md",
)

custom_css = """
body.dark { background-color: #0b0f19; }
#sample-audio-player { margin-top: 10px; }
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

    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab("Voice Clone", id="tab_voice_clone"):
            gr.Markdown("Clone Voices from Samples. <small>(Use Prep Samples to add samples)</small>")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")
                    vc_sample_dropdown = gr.Dropdown(
                        choices=get_sample_choices(),
                        value=get_sample_choices()[0] if get_sample_choices() else None,
                        label="Select Sample",
                        interactive=True
                    )
                    vc_sample_audio = gr.Audio(label="Sample Preview", type="filepath", interactive=False, elem_id="sample-audio-player")
                    vc_sample_text = gr.Textbox(label="Sample Text", interactive=False, max_lines=10)
                    
                    vc_sample_dropdown.change(
                        fn=load_sample,
                        inputs=[vc_sample_dropdown],
                        outputs=[vc_sample_audio, vc_sample_text]
                    )

                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")
                    target_text = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6
                    )
                    
                    engine_dropdown = gr.Dropdown(
                        choices=["Fish Speech S2 Pro (CPP)", "Fish Speech S2 Pro (PyTorch)"],
                        value="Fish Speech S2 Pro (CPP)",
                        label="Inference Engine"
                    )
                    cpp_model_row = gr.Row(visible=True)
                    with cpp_model_row:
                        cpp_model_dropdown = gr.Dropdown(
                            choices=list(GGUF_MODELS.keys()), 
                            label="GGUF Model Quantization", 
                            value=list(GGUF_MODELS.keys())[1]
                        )
                    trained_model_row = gr.Row(visible=False)
                    with trained_model_row:
                        trained_model_dropdown = gr.Dropdown(
                            choices=get_trained_models(),
                            label="Trained LoRA Model",
                            value="Base Model (Fish S2 Pro)"
                        )
                        
                    def update_engine_ui(engine):
                        is_cpp = (engine == "Fish Speech S2 Pro (CPP)")
                        if is_cpp:
                            return gr.update(visible=True), gr.update(visible=False)
                        else:
                            models = get_trained_models()
                            return gr.update(visible=False), gr.update(visible=True)

                    engine_dropdown.change(
                        fn=update_engine_ui,
                        inputs=engine_dropdown,
                        outputs=[cpp_model_row, trained_model_row]
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        top_p_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Top-P")
                        top_k_slider = gr.Slider(1, 100, value=30, step=1, label="Top-K")
                        temperature_slider = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                        rep_pen_slider = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
                        split_para_check = gr.Checkbox(label="Split by Paragraphs (Recommended for long texts)", value=False)
                        
                    with gr.Row():
                        generate_btn = gr.Button("Generate Audio 🚀", variant="primary", size="lg")
                        
                    with gr.Row():
                        output_audio = gr.Audio(label="Generated Audio", type="filepath")
                    
                    with gr.Row():
                        clone_status = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)
                    
                    generate_btn.click(
                        fn=clone_voice,
                        inputs=[engine_dropdown, cpp_model_dropdown, trained_model_dropdown, target_text, vc_sample_audio, vc_sample_text, top_p_slider, top_k_slider, temperature_slider, rep_pen_slider, split_para_check],
                        outputs=[output_audio, clone_status]
                    )

        with gr.Tab("Prep Samples", id="tab_prep_samples"):
            gr.Markdown("Prepare audio samples for voice cloning.")
            with gr.Row():
                with gr.Column(scale=1) as prep_sidebar:
                    with gr.Group() as audio_samples_group:
                        gr.Markdown("### Audio Samples")
                        prep_sample_dropdown = gr.Dropdown(
                            choices=get_sample_choices(),
                            value=get_sample_choices()[0] if get_sample_choices() else None,
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
                            prep_audio_editor = gr.Audio(label="Audio Editor (Use Trim icon to edit)", type="filepath", interactive=True)
                            
                            gr.Markdown("### Reference Text")
                            transcription_output = gr.Textbox(
                                label="Text",
                                lines=4,
                                max_lines=10,
                                interactive=True,
                                placeholder="Transcription will appear here, or enter/edit text manually..."
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
                                batch_folder_input = gr.Textbox(label="Source Audio Folder", placeholder="path/to/your/audio/files", scale=4)
                                
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
                    
                    def on_prep_sample_select(sample_name):
                        if not sample_name:
                            return None, ""
                        res_audio, res_text = load_sample(sample_name)
                        return res_audio.get("value"), res_text.get("value")
                    
                    # Initial load for Voice Clone tab
                    app.load(
                        fn=lambda: load_sample(get_sample_choices()[0]) if get_sample_choices() else (None, ""),
                        inputs=None,
                        outputs=[vc_sample_audio, vc_sample_text]
                    )
                    
                    # Initial load for Prep Samples tab
                    app.load(
                        fn=lambda: on_prep_sample_select(get_sample_choices()[0]) if get_sample_choices() else (None, ""),
                        inputs=None,
                        outputs=[prep_audio_editor, transcription_output]
                    )

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
                    
                    # Auto-clear UI fields when audio is cleared
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

                    def show_samples_group():
                        return gr.update(visible=True)
                    def hide_samples_group():
                        return gr.update(visible=False)

                    prep_tab_single = prep_tabs.children[0]
                    prep_tab_dataset = prep_tabs.children[1]
                    prep_tab_single.select(fn=show_samples_group, inputs=[], outputs=[audio_samples_group])
                    prep_tab_dataset.select(fn=hide_samples_group, inputs=[], outputs=[audio_samples_group])

        with gr.Tab("Lora Training", id="tab_lora_training"):
            gr.Markdown("### 🏋️ Fish Speech S2 Pro LoRA Training Pipeline")
            
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
                        model_name_input = gr.Textbox(label="Trained Model Name", placeholder="e.g. speaker_v1")
                        
                        with gr.Row():
                            train_max_steps = gr.Slider(50, 5000, value=1000, step=50, label="Max Steps", info="Auto-calculated")
                            train_lr = gr.Number(value=1e-5, label="Learning Rate")
                            
                        with gr.Row():
                            fs_lora_rank = gr.Slider(minimum=4, maximum=64, value=32, step=4, label="LoRA Rank (r)")
                            fs_lora_alpha = gr.Slider(minimum=4, maximum=64, value=16, step=4, label="LoRA Alpha")
                            
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        
                    training_status = gr.Textbox(label="Status Console", lines=10, interactive=False)
                    
                    # Automatic Step 3: Result info (No manual buttons)
                    model_path_display = gr.Markdown("")

            # --- Training Actions ---
            refresh_folders_btn.click(fn=lambda: gr.update(choices=get_dataset_choices()), outputs=[training_output_name])
            analyze_btn.click(analyze_dataset, [training_output_name, vram_preset_radio], [dataset_info, train_max_steps])
            
            auto_train_done_event = train_btn.click(
                handle_lora_unified, 
                [training_output_name, model_name_input, train_max_steps, train_lr, vram_preset_radio, fs_lora_rank, fs_lora_alpha], 
                training_status
            )
            auto_train_done_event.then(
                fn=lambda mn, rank, alpha: handle_lora_export(mn, None, rank, alpha), 
                inputs=[model_name_input, fs_lora_rank, fs_lora_alpha], 
                outputs=[training_status]
            )

if __name__ == "__main__":
    # Use "127.0.0.1" for local access or "0.0.0.0" for network access
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)