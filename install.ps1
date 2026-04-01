# Fish Speech S2 Pro Installer
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Cyan
}
function Write-Ok($text)   { Write-Host "[OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "[WARNING] $text" -ForegroundColor Yellow }
function Write-Err($text)  { Write-Host "[ERROR] $text" -ForegroundColor Red }
function Write-Info($text) { Write-Host "[INFO] $text" -ForegroundColor Gray }

Write-Header "Fish Speech S2 Pro - Voice Clone & Training - Installer"

# -------------------------------------------------------
# UV
# -------------------------------------------------------
Write-Header "Checking UV"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "UV not found. Installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:APPDATA\uv\bin;$env:PATH"
} else {
    Write-Ok "UV is already installed."
}

# -------------------------------------------------------
# Ninja (winget)
# -------------------------------------------------------
if (-not (Get-Command ninja -ErrorAction SilentlyContinue) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing Ninja (via winget)"
    try {
        winget install --id Ninja-build.Ninja --exact --silent --accept-source-agreements --accept-package-agreements
        # Refresh Path for current session
        $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    } catch {
        Write-Warn "Winget failed to install Ninja. Please install it manually from https://ninja-build.org/"
    }
}

if (Get-Command ninja -ErrorAction SilentlyContinue) {
    Write-Ok "Ninja is ready. Setting CMAKE_GENERATOR=Ninja for all compilations."
    $env:CMAKE_GENERATOR = "Ninja"
    $env:CMAKE_MAKE_PROGRAM = (Get-Command ninja).Source
} else {
    Write-Warn "Ninja not found. Falling back to default generators."
}

# -------------------------------------------------------
# Virtual Environment
# -------------------------------------------------------
Write-Header "Creating Virtual Environment (Python 3.10)"
uv venv --python 3.10
& .\.venv\Scripts\Activate.ps1

# -------------------------------------------------------
# GPU Detection
# -------------------------------------------------------
Write-Header "Detecting GPU"

$cudaExtra   = "cu128"
$cudaIndex   = "https://download.pytorch.org/whl/cu128"
$cudaNightly = $false
$cudaLabel   = "unknown"

try {
    $raw = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>$null
    if (-not $raw) {
        $raw = & nvidia-smi --query-gpu=compute_cap --format=csv 2>$null | Select-Object -Skip 1
    }
    $sm = ($raw | Select-Object -First 1).Trim()

    if ($sm -match "^(\d+)\.(\d+)$") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]

        if ($major -ge 12) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/nightly/cu128"
            $cudaNightly = $true;  $cudaLabel = "Blackwell sm_$sm (cu128 nightly)"
        } elseif ($major -ge 8) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/cu128"
            $cudaNightly = $false; $cudaLabel = "Ampere/Ada sm_$sm (cu128 stable)"
        } elseif ($major -eq 7 -and $minor -ge 5) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/cu128"
            $cudaNightly = $false; $cudaLabel = "Turing sm_$sm (cu128 stable)"
        } else {
            $cudaExtra = "cu126"; $cudaIndex = "https://download.pytorch.org/whl/cu126"
            $cudaNightly = $false; $cudaLabel = "Volta sm_$sm (cu126 stable)"
        }
    } else {
        Write-Warn "Could not parse compute capability '$sm'. Defaulting to cu128 stable."
        $cudaLabel = "fallback cu128 stable"
    }
} catch {
    Write-Warn "nvidia-smi not available. Defaulting to cu128 stable."
    $cudaLabel = "fallback cu128 stable"
}

Write-Ok "GPU   : $cudaLabel"
Write-Ok "Extra : $cudaExtra"
Write-Ok "Index : $cudaIndex"

# -------------------------------------------------------
# Fish Speech
# -------------------------------------------------------
Write-Header "Installing Fish Speech [$cudaExtra]"

$fishInstalled = uv pip list | Select-String "fish-speech"
if ($fishInstalled) {
    Write-Ok "Fish Speech dependencies are already installed. Skipping."
} else {
    Push-Location "modules\s2"
    try {
        Write-Info "Installing torch + CUDA dependencies ($cudaIndex)..."
        uv pip install torch torchaudio torchvision --index-url $cudaIndex
        uv pip install -e ".[preprocess]" --no-deps
        # Install fish-speech code without dependencies to avoid overwriting PyTorch
        uv pip install -e ".[preprocess]" --no-deps
    } finally {
        Pop-Location
    }
}

# -------------------------------------------------------
# Blackwell only: upgrade to nightly
# -------------------------------------------------------
if ($cudaNightly -and -not $fishInstalled) {
    Write-Header "Upgrading torch to cu128 nightly (Blackwell sm_120)"
    uv pip install --pre torch torchvision torchaudio --index-url $cudaIndex
}

# -------------------------------------------------------
# S2.cpp
# -------------------------------------------------------
Write-Header "Compiling S2.cpp"

# Check for Vulkan SDK (Needed for k-quants on GPU)
if (-not $env:VULKAN_SDK -and -not (Test-Path "C:\VulkanSDK") -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Info "Vulkan SDK not found. Attempting unattended installation via winget..."
    try {
        winget install --id LunarG.VulkanSDK --exact --silent --accept-source-agreements --accept-package-agreements
        # Refresh env path for current session slightly (rough estimate until reboot/new shell)
        if (Test-Path "C:\VulkanSDK") {
            $latest = Get-ChildItem "C:\VulkanSDK" -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($latest) { $env:VULKAN_SDK = $latest.FullName; Write-Ok "Vulkan SDK installed to $($env:VULKAN_SDK)" }
        }
    } catch {
        Write-Warn "Winget failed to install Vulkan SDK. Please install it manually from https://vulkan.lunarg.com/"
    }
}

# -------------------------------------------------------
# Visual Studio 2022 (winget)
# -------------------------------------------------------
$vsBase = "C:\Program Files\Microsoft Visual Studio\2022"
$vcvars = $null
foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
    $candidate = "$vsBase\$edition\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $candidate) { $vcvars = $candidate; break }
}

if (-not $vcvars -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing Visual Studio 2022 Community with C++ Workload (via winget)"
    # Install VS Community with the Desktop C++ workload (Microsoft.VisualStudio.Workload.NativeDesktop)
    winget install --id Microsoft.VisualStudio.2022.Community --override "--passive --config $PSScriptRoot\.vsconfig --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended" --accept-source-agreements --accept-package-agreements
    
    # Re-scan for vcvars64
    foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
        $candidate = "$vsBase\$edition\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $candidate) { $vcvars = $candidate; break }
    }
}

# -------------------------------------------------------
# CUDA Toolkit (winget)
# -------------------------------------------------------
if (-not $env:CUDA_PATH -and -not (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing CUDA Toolkit (via winget)"
    # Install CUDA 12.8 or latest available silently
    winget install --id Nvidia.CUDA --exact --silent --accept-source-agreements --accept-package-agreements
}

# 1. Try to find the path via 'nvcc' if it's in the PATH environment variable
if (-not $cudaToolkitPath -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $nvccPath = (Get-Command nvcc).Source
    if ($nvccPath) {
        # Go up from 'bin\nvcc.exe' to the toolkit root folder
        $cudaToolkitPath = (Get-Item $nvccPath).Directory.Parent.FullName
    }
}

# 2. Fallback to standard Nvidia installation directory
if (-not $cudaToolkitPath) {
    $baseCUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $baseCUDA) {
        $versions = Get-ChildItem $baseCUDA -Directory | Where-Object { $_.Name -match "^v\d+\.\d+$" } | Sort-Object Name -Descending
        if ($versions) { $cudaToolkitPath = $versions[0].FullName }
    }
}

# 3. Check for specific versioned environment variables (e.g., CUDA_PATH_V12_4)
if (-not $cudaToolkitPath) {
    $envVars = Get-ChildItem Env: | Where-Object { $_.Name -match "^CUDA_PATH_V\d+_\d+$" } | Sort-Object Name -Descending
    foreach ($envVar in $envVars) {
        if (Test-Path $envVar.Value) {
            $cudaToolkitPath = $envVar.Value
            break
        }
    }
}

if (-not $cudaToolkitPath -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Warn "CUDA Toolkit not detected."
    Write-Host "  S2.cpp requires:"
    Write-Host "    1. CUDA Toolkit  https://developer.nvidia.com/cuda-downloads"
    Write-Host "    2. Visual Studio 2022 with C++ workload"
    Write-Host "    3. CUDA VS integration: (found in CUDA extras folder)"
    Write-Warn "Skipping S2.cpp CUDA backend. Defaulting to CPU only."
    $cudaArgs = ""
} elseif (-not $vcvars) {
    Write-Warn "Visual Studio 2022 not found. Cannot compile S2.cpp."
    Write-Warn "Install VS2022 with 'Desktop development with C++' workload."
} else {
    Write-Ok "Found vcvars64.bat: $vcvars"
    if ($cudaToolkitPath) { Write-Ok "Using CUDA Toolkit: $cudaToolkitPath" }

    Push-Location "modules\s2.cpp"
    try {
        if ((Test-Path "build\s2.exe") -or (Test-Path "build\Release\s2.exe")) {
            Write-Ok "S2.cpp is already compiled. Audited and skipped."
        } else {
            if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
            New-Item -ItemType Directory "build" | Out-Null
            $buildDir = (Resolve-Path "build").Path

            $useNinja = (Get-Command ninja -ErrorAction SilentlyContinue)
            if ($useNinja) {
                $generator = "Ninja"
                Write-Ok "Generator: Ninja (Priority)"
            } else {
                $generator = "Visual Studio 17 2022"
                Write-Ok "Generator: Visual Studio 17 2022 (Fallback)"
            }

            Write-Info "Configuring..."
            # CUDA backend only supports F16/Q8_0 for get_rows.
            # Vulkan backend supports ALL quantizations including k-quants.
            # Enable both backends so the runtime picks the best one.
            # Performance flags: /Ox (max opt), /arch:AVX2 (SIMD), /fp:fast, /GL /LTCG (Link Time Optimization), /openmp (Multithreading)
            $cudaArgs = "-DS2_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=`"/Ox /arch:AVX2 /fp:fast /GL /DNDEBUG /openmp`" -DCMAKE_EXE_LINKER_FLAGS=`"/LTCG`" -DCMAKE_STATIC_LINKER_FLAGS=`"/LTCG`""
            
            # Check for Vulkan SDK
            $vulkanSdk = $env:VULKAN_SDK
            if (-not $vulkanSdk) {
                # Try common install paths
                $vulkanBase = "C:\VulkanSDK"
                if (Test-Path $vulkanBase) {
                    $versions = Get-ChildItem $vulkanBase -Directory | Sort-Object Name -Descending
                    if ($versions) { $vulkanSdk = $versions[0].FullName }
                }
            }
            if ($vulkanSdk -and (Test-Path $vulkanSdk)) {
                Write-Ok "Vulkan SDK found: $vulkanSdk — enabling Vulkan backend (k-quants GPU support)"
                $cudaArgs += " -DS2_VULKAN=ON"
            } else {
                Write-Warn "Vulkan SDK not found. K-quant models (Q2_K-Q6_K) will only run on CPU."
                Write-Warn "Install Vulkan SDK from https://vulkan.lunarg.com/ for full GPU quant support."
            }
            
            if ($cudaToolkitPath) { $cudaArgs += " -DCUDAToolkit_ROOT=`"$cudaToolkitPath`"" }
            
            cmd /c "`"$vcvars`" && cd `"$buildDir`" && cmake .. -G `"$generator`" $cudaArgs"
            if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

            Write-Info "Building..."
            cmd /c "`"$vcvars`" && cd `"$buildDir`" && cmake --build . --config Release"
            if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

            Write-Ok "S2.cpp compiled successfully."
        }
    } catch {
        Write-Err $_.Exception.Message
        Write-Warn "S2.cpp skipped - some features may be unavailable."
    } finally {
        Pop-Location
    }
}

# -------------------------------------------------------
# Extra deps
# -------------------------------------------------------
Write-Header "Installing additional dependencies"
uv pip install soundfile librosa numpy "pydantic>=2.0" "protobuf>=3.19,<4" faster-whisper ctranslate2 huggingface_hub hf-xet loralib gradio loguru transformers datasets lightning hydra-core tensorboard natsort einops rich wandb grpcio kui uvicorn pyrootutils resampy einx zstandard pydub pyaudio modelscope opencc-python-reimplemented silero-vad ormsgpack tiktoken cachetools descript-audio-codec safetensors google-genai deepgram-sdk pyannote.audio triton-windows

# -------------------------------------------------------
# Verify
# -------------------------------------------------------
Write-Header "Verifying GPU / CUDA"
python -c "import torch; ok=torch.cuda.is_available(); v=torch.__version__; d=torch.cuda.get_device_name(0) if ok else 'N/A'; print(('[OK] CUDA available' if ok else '[WARNING] CUDA NOT available') + ' | torch=' + v + ' | device=' + d)"

Write-Header "Installation complete! Run the app with start.bat"
Read-Host "Press Enter to exit"