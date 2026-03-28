#!/bin/bash

# Fish Speech S2 Pro - Linux / WSL Installer
# Usage: bash install.sh

set -e

# --- Logging helpers ---
Write-Header() {
    echo ""
    echo "======================================================="
    echo "  $1"
    echo "======================================================="
}

Write-Ok() { echo -e "[\e[32mOK\e[0m] $1"; }
Write-Info() { echo -e "[\e[34mINFO\e[0m] $1"; }
Write-Warn() { echo -e "[\e[33mWARNING\e[0m] $1"; }
Write-Err() { echo -e "[\e[31mERROR\e[0m] $1"; }

Write-Header "Fish Speech S2 Pro - Voice Clone & Training - Linux Installer"

# -------------------------------------------------------
# UV Installation
# -------------------------------------------------------
Write-Header "Checking UV"
if ! command -v uv &> /dev/null; then
    Write-Info "UV not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    if [ -f "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
else
    Write-Ok "UV is already installed."
fi

# -------------------------------------------------------
# System Dependencies
# -------------------------------------------------------
Write-Header "Checking System Dependencies"
if command -v apt-get &> /dev/null; then
    Write-Info "Updating apt and installing build-essential, cmake, ninja, libvulkan-dev, ffmpeg, alsa-utils..."
    sudo apt-get update -y
    sudo apt-get install -y build-essential cmake ninja-build libvulkan-dev ffmpeg libsndfile1-dev portaudio19-dev alsa-utils
else

    Write-Warn "Not a Debian-based system (apt-get not found). Please ensure cmake, ninja, vulkan-dev, and ffmpeg are installed."
fi

# -------------------------------------------------------
# Virtual Environment
# -------------------------------------------------------
Write-Header "Creating Virtual Environment (Python 3.10)"
uv venv --python 3.10
source .venv/bin/activate

# -------------------------------------------------------
# GPU Detection
# -------------------------------------------------------
Write-Header "Detecting GPU"

CUDA_EXTRA="cu128"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
CUDA_NIGHTLY=false
CUDA_LABEL="unknown"

if command -v nvidia-smi &> /dev/null; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
    
    if [[ $SM =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        MAJOR=${BASH_REMATCH[1]}
        MINOR=${BASH_REMATCH[2]}

        if [ "$MAJOR" -ge 12 ]; then
            CUDA_EXTRA="cu128"
            CUDA_INDEX="https://download.pytorch.org/whl/nightly/cu128"
            CUDA_NIGHTLY=true
            CUDA_LABEL="Blackwell sm_$SM (cu128 nightly)"
        elif [ "$MAJOR" -ge 8 ]; then
            CUDA_EXTRA="cu128"
            CUDA_INDEX="https://download.pytorch.org/whl/cu128"
            CUDA_NIGHTLY=false
            CUDA_LABEL="Ampere/Ada sm_$SM (cu128 stable)"
        elif [ "$MAJOR" -eq 7 ] && [ "$MINOR" -ge 5 ]; then
            CUDA_EXTRA="cu128"
            CUDA_INDEX="https://download.pytorch.org/whl/cu128"
            CUDA_NIGHTLY=false
            CUDA_LABEL="Turing sm_$SM (cu128 stable)"
        else
            CUDA_EXTRA="cu126"
            CUDA_INDEX="https://download.pytorch.org/whl/cu126"
            CUDA_NIGHTLY=false
            CUDA_LABEL="Volta sm_$SM (cu126 stable)"
        fi
    else
        Write-Warn "Could not parse compute capability '$SM'. Defaulting to cu128 stable."
        CUDA_LABEL="fallback cu128 stable"
    fi
else
    Write-Warn "nvidia-smi not available. Defaulting to cu128 stable (CPU fallback likely)."
    CUDA_LABEL="fallback cu128 stable"
fi

Write-Ok "GPU   : $CUDA_LABEL"
Write-Ok "Extra : $CUDA_EXTRA"
Write-Ok "Index : $CUDA_INDEX"

# -------------------------------------------------------
# Fish Speech installation
# -------------------------------------------------------
Write-Header "Installing Fish Speech [$CUDA_EXTRA]"

# Install torch first to avoid conflicts
Write-Info "Installing torch + CUDA dependencies ($CUDA_INDEX)..."
uv pip install torch torchaudio torchvision --index-url $CUDA_INDEX

if [ "$CUDA_NIGHTLY" = true ]; then
    Write-Info "Upgrading to torch cu128 nightly..."
    uv pip install --pre torch torchvision torchaudio --index-url $CUDA_INDEX
fi

Write-Info "Installing editable fish-speech and preprocess dependencies..."
pushd modules/s2 > /dev/null

# Fix for missing source code (models/datasets folder)
if [ ! -d "fish_speech/models" ] || [ ! -d "fish_speech/datasets" ]; then
    Write-Warn "fish_speech source code (models or datasets) missing! Attempting to restore from official repo..."
    if ! command -v git &> /dev/null; then
        Write-Err "Git not found, cannot restore codebase."
    else
        # Initialize as git repo if not already, and pull the code
        [ ! -d ".git" ] && git init
        if ! git remote | grep origin > /dev/null; then
            git remote add origin https://github.com/fishaudio/fish-speech.git
        fi
        git fetch origin main --depth 1
        # Restore all core source folders
        git checkout origin/main -f fish_speech/ tools
        Write-Ok "Restored all fish_speech and tools source code from official repo."
    fi
fi


uv pip install -e ".[preprocess]" --no-deps
popd > /dev/null


# -------------------------------------------------------
# S2.cpp Compilation
# -------------------------------------------------------
Write-Header "Compiling S2.cpp"

if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
    Write-Ok "CUDA found. Enabling GPU backends (CUDA/Vulkan)."
    pushd modules/s2.cpp > /dev/null
    
    if [ -f "build/s2" ]; then
        Write-Ok "S2.cpp is already compiled. Skipping."
    else
        rm -rf build
        mkdir build
        cd build
        
        # Detect Ninja
        GENERATOR="Unix Makefiles"
        if command -v ninja &> /dev/null; then GENERATOR="Ninja"; fi
        
        Write-Info "Configuring and Building S2.cpp with $GENERATOR..."
        cmake .. -G "$GENERATOR" \
                 -DS2_CUDA=ON \
                 -DS2_VULKAN=ON \
                 -DCMAKE_BUILD_TYPE=Release
        
        if [ "$GENERATOR" = "Ninja" ]; then
            ninja
        else
            make -j$(nproc)
        fi
        Write-Ok "S2.cpp compiled successfully."
    fi
    popd > /dev/null
else
    Write-Warn "CUDA Toolkit not detected. S2.cpp will be compiled for CPU only."
    pushd modules/s2.cpp > /dev/null
    if [ ! -f "build/s2" ]; then
        rm -rf build
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
    fi
    popd > /dev/null
fi

# -------------------------------------------------------
# Extra dependencies
Write-Header "Installing additional dependencies"
# Step 1: Force install modern Protobuf to satisfy Fish Speech generated protos
uv pip install "protobuf>=4.21.0"
# Step 2: Install descript codecs without dependencies to bypass their strict protobuf<3.20 pin
uv pip install descript-audiotools descript-audio-codec --no-deps
# Step 2b: Install all descript-audiotools runtime deps EXCEPT protobuf (which is pinned above)
# These are omitted by --no-deps: argbind pyloudnorm julius ffmpy ipython matplotlib pystoi torch-stoi markdown2 randomname importlib-resources
uv pip install argbind pyloudnorm julius ffmpy ipython matplotlib pystoi torch-stoi markdown2 randomname importlib_resources
# Step 3: Install the remaining dependencies normally

uv pip install "protobuf>=4.21.0" soundfile librosa numpy "pydantic>=2.0" faster-whisper ctranslate2 huggingface_hub hf-xet loralib gradio loguru transformers datasets lightning hydra-core tensorboard natsort einops rich wandb grpcio kui uvicorn pyrootutils resampy einx zstandard pydub pyaudio modelscope opencc-python-reimplemented silero-vad ormsgpack tiktoken cachetools safetensors google-genai deepgram-sdk pyannote.audio flatten_dict




# -------------------------------------------------------
# Verify
# -------------------------------------------------------
Write-Header "Verifying GPU / CUDA"
python -c "import torch; ok=torch.cuda.is_available(); v=torch.__version__; d=torch.cuda.get_device_name(0) if ok else 'N/A'; print(('[OK] CUDA available' if ok else '[WARNING] CUDA NOT available') + ' | torch=' + v + ' | device=' + d)"

Write-Header "Installation complete! Run the app with ./start.sh"
