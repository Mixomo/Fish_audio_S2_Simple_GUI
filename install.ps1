$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Pinned runtime. Updating one of these values invalidates only the component
# that actually needs to change; the installer does not rebuild the project.
$UvVersion = "0.11.33"
$PythonVersion = "3.11.15"
$TorchVersion = "2.8.0"
$TorchVisionVersion = "0.23.0"
$VulkanVersion = "1.4.341.1"
$VulkanHeaderVersion = "341"
$CMakeVersion = "3.31.8"
$NinjaVersion = "1.13.1"
$TritonVersion = "3.4.0.post21"
$GradioVersion = "5.49.1"
$GradioClientVersion = "1.13.3"
$NumpyVersion = "1.26.4"
$PyArrowVersion = "20.0.0"
$ProtobufVersion = "3.20.3"
$FilelockVersion = "3.32.3"
$FsspecVersion = "2026.7.0"
$MarkupsafeVersion = "3.0.3"
$NetworkxVersion = "3.6.1"
$PillowVersion = "11.3.0"
$TypingExtensionsVersion = "4.16.0"
$FasterWhisperVersion = "1.2.0"
$CTranslate2Version = "4.6.0"
$AvVersion = "18.1.0"
$PsutilVersion = "7.0.0"
$HfXetVersion = "1.1.10"
$SetuptoolsVersion = "80.9.0"

$Runtime = Join-Path $Root ".runtime"
$UvDir = Join-Path $Runtime "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$PythonRoot = Join-Path $Runtime "python"
$Venv = Join-Path $Root ".venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$Tools = Join-Path $Runtime "tools"
$CMakeRoot = Join-Path $Tools "cmake"
$NinjaRoot = Join-Path $Tools "ninja"
$VulkanRoot = Join-Path $Runtime "vulkan"
$VsRoot = Join-Path $Runtime "vs-buildtools"
$Temp = Join-Path $Runtime "temp"
$Downloads = Join-Path $Runtime "downloads"
$State = Join-Path $Runtime "state"
$UvCache = Join-Path $Runtime "uv-cache"

function Header([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Info([string]$Text) { Write-Host "[INFO] $Text" -ForegroundColor Gray }
function Skip([string]$Text) { Write-Host "[SKIP] $Text" -ForegroundColor DarkGray }
function Ok([string]$Text) { Write-Host "[OK] $Text" -ForegroundColor Green }
function Warn([string]$Text) { Write-Host "[WARNING] $Text" -ForegroundColor Yellow }

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
}

function Get-NativeTool([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "$Name is required. Windows 10/11 normally ships it in System32."
    }
    return $cmd.Source
}

$CurlExe = Get-NativeTool "curl.exe"
$TarExe = Get-NativeTool "tar.exe"

function Download-Cached([string]$Url, [string]$Destination) {
    Ensure-Directory (Split-Path -Parent $Destination)
    if ((Test-Path $Destination) -and ((Get-Item $Destination).Length -gt 0)) {
        Skip "Cached download: $(Split-Path -Leaf $Destination)"
        return
    }

    $part = "$Destination.part"
    Info "curl.exe -> $(Split-Path -Leaf $Destination)"
    & $CurlExe --fail --location --retry 4 --retry-delay 2 --retry-all-errors `
        --continue-at - --output $part $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    if (-not (Test-Path $part) -or (Get-Item $part).Length -le 0) { throw "Downloaded file is empty: $Url" }
    Move-Item -Force $part $Destination
}

function Expand-ZipFast([string]$Zip, [string]$Destination) {
    if (Test-Path $Destination) { Remove-Item -Recurse -Force $Destination }
    Ensure-Directory $Destination
    & $TarExe -xf $Zip -C $Destination
    if ($LASTEXITCODE -ne 0) { throw "tar.exe failed to extract: $Zip" }
}

function Invoke-PythonProbe([string]$Code) {
    if (-not (Test-Path $PythonExe)) {
        return [pscustomobject]@{ ExitCode = 9009; StdOut = ""; StdErr = "python.exe not found" }
    }

    # Windows PowerShell 5.1 can expose native stderr as a terminating
    # NativeCommandError when $ErrorActionPreference = "Stop".  Do not use
    # Start-Process here: some Windows environments expose both PATH and Path,
    # which makes Start-Process fail while building its environment dictionary.
    # Invoke Python directly with a local non-terminating error policy instead.
    Ensure-Directory $Temp
    $id = [Guid]::NewGuid().ToString("N")
    $scriptPath = Join-Path $Temp "python-probe-$id.py"
    $stdoutPath = Join-Path $Temp "python-probe-$id.out"
    $stderrPath = Join-Path $Temp "python-probe-$id.err"

    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($scriptPath, $Code, $utf8NoBom)
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $PythonExe $scriptPath 1> $stdoutPath 2> $stderrPath
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorAction
        }

        $stdout = if (Test-Path $stdoutPath) { Get-Content -Raw $stdoutPath } else { "" }
        $stderr = if (Test-Path $stderrPath) { Get-Content -Raw $stderrPath } else { "" }
        return [pscustomobject]@{ ExitCode = $exitCode; StdOut = $stdout; StdErr = $stderr }
    } catch {
        return [pscustomobject]@{ ExitCode = 1; StdOut = ""; StdErr = $_.Exception.Message }
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $scriptPath, $stdoutPath, $stderrPath
    }
}

function Test-PythonCode([string]$Code) {
    return ((Invoke-PythonProbe $Code).ExitCode -eq 0)
}

function Get-TextSha256([string]$Text) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-TreeFingerprint([string]$Path) {
    # Fast metadata fingerprint: enough to detect a replaced patch/source tree
    # without hashing hundreds of megabytes on every installer launch.
    $extensions = @(".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".cmake", ".txt")
    $rows = Get-ChildItem $Path -Recurse -File | Where-Object {
        $_.FullName -notmatch "[\\/]build[\\/]" -and
        ($extensions -contains $_.Extension.ToLowerInvariant() -or $_.Name -eq "CMakeLists.txt")
    } | Sort-Object FullName | ForEach-Object {
        $rel = $_.FullName.Substring($Path.Length).TrimStart('\','/')
        "$rel|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
    }
    return Get-TextSha256 (($rows -join "`n") + "`ncmake=$CMakeVersion`nvulkan=$VulkanVersion")
}

function Read-State([string]$Name) {
    $path = Join-Path $State $Name
    if (Test-Path $path) { return (Get-Content -Raw $path).Trim() }
    return ""
}

function Write-State([string]$Name, [string]$Value) {
    Ensure-Directory $State
    Set-Content -Path (Join-Path $State $Name) -Value $Value -Encoding ASCII
}

foreach ($dir in @(
    $Runtime, $UvDir, $PythonRoot, $Tools, $Temp, $Downloads, $State, $UvCache,
    (Join-Path $Runtime "cache\huggingface"), (Join-Path $Runtime "cache\modelscope"),
    (Join-Path $Runtime "gradio-temp"), (Join-Path $Runtime "triton-cache"),
    (Join-Path $Runtime "torchinductor-cache"), (Join-Path $Runtime "torch-extensions"),
    (Join-Path $Root "models"), (Join-Path $Root "outputs"), (Join-Path $Root "samples"),
    (Join-Path $Root "datasets"), (Join-Path $Root "projects")
)) { Ensure-Directory $dir }

# Project-local caches only. uv/pip do not retain package archives after installation.
$env:UV_PYTHON_INSTALL_DIR = $PythonRoot
$env:UV_PROJECT_ENVIRONMENT = $Venv
$env:UV_CACHE_DIR = $UvCache
$env:UV_NO_CACHE = "1"
$env:UV_LINK_MODE = "copy"
$env:PIP_NO_CACHE_DIR = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TMP = $Temp
$env:TEMP = $Temp
$env:HF_HOME = Join-Path $Runtime "cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_HUB_CACHE = $env:HUGGINGFACE_HUB_CACHE
$env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
$env:MODELSCOPE_CACHE = Join-Path $Runtime "cache\modelscope"
$env:GRADIO_TEMP_DIR = Join-Path $Runtime "gradio-temp"
$env:TRITON_CACHE_DIR = Join-Path $Runtime "triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $Runtime "torchinductor-cache"
$env:TORCH_EXTENSIONS_DIR = Join-Path $Runtime "torch-extensions"
$env:CUDA_MODULE_LOADING = "LAZY"

Header "1/8 - Project-local uv $UvVersion"
$uvReady = $false
if (Test-Path $UvExe) {
    try {
        $installedUv = (& $UvExe --version) -join " "
        $uvReady = ($installedUv -match [regex]::Escape($UvVersion))
    } catch { $uvReady = $false }
}
if (-not $uvReady) {
    $zip = Join-Path $Downloads "uv-$UvVersion-windows-x64.zip"
    Download-Cached "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
    Expand-ZipFast $zip $UvDir
} else {
    Skip "uv $UvVersion is already installed."
}
if (-not (Test-Path $UvExe)) { throw "uv.exe was not found after extraction." }
Ok (& $UvExe --version)

Header "2/8 - Project-local Python $PythonVersion + venv"
$venvReady = $false
if (Test-Path $PythonExe) {
    $versionProbe = Invoke-PythonProbe "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    if ($versionProbe.ExitCode -eq 0) {
        $current = $versionProbe.StdOut.Trim()
        $venvReady = ($current -eq $PythonVersion)
    }
}
if ($venvReady) {
    Skip ".venv already uses Python $PythonVersion; preserving it unchanged."
} else {
    if (Test-Path $Venv) {
        Warn ".venv exists but uses an incompatible/broken Python; recreating only the venv."
        Remove-Item -Recurse -Force $Venv
    }
    & $UvExe python install $PythonVersion --no-cache --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw "uv failed to prepare Python $PythonVersion." }
    & $UvExe venv --python $PythonVersion --no-cache $Venv
    if ($LASTEXITCODE -ne 0) { throw "uv failed to create the project venv." }
}
Ok (& $PythonExe --version)

Header "3/8 - CUDA 12.8 Python runtime"
$torchProbe = @"
import torch, torchvision, torchaudio
assert torch.__version__.split('+')[0] == '$TorchVersion'
assert torchvision.__version__.split('+')[0] == '$TorchVisionVersion'
assert torchaudio.__version__.split('+')[0] == '$TorchVersion'
assert torch.version.cuda == '12.8'
"@
if (Test-PythonCode $torchProbe) {
    Skip "PyTorch $TorchVersion + CUDA 12.8 is already correct."
} else {
    & $UvExe pip install --python $PythonExe --no-cache `
        "torch==$TorchVersion" "torchvision==$TorchVisionVersion" "torchaudio==$TorchVersion" `
        --index-url "https://download.pytorch.org/whl/cu128"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA installation failed." }
}

Header "4/8 - Frozen Fish S2 environment (uv.lock)"
$FishProject = Join-Path $Root "modules\s2"
$FishLock = Join-Path $FishProject "uv.lock"
if (-not (Test-Path $FishLock)) { throw "Required lockfile is missing: $FishLock" }

$lockHash = (Get-FileHash $FishLock -Algorithm SHA256).Hash.ToLowerInvariant()
$pyprojectHash = (Get-FileHash (Join-Path $FishProject "pyproject.toml") -Algorithm SHA256).Hash.ToLowerInvariant()
$RuntimeCompatRevision = "2026-09-15-numpy1264"
$depsFingerprint = Get-TextSha256 ((
    $PythonVersion, $TorchVersion, $TorchVisionVersion, $TritonVersion,
    $GradioVersion, $GradioClientVersion, $NumpyVersion, $PyArrowVersion,
    $ProtobufVersion, $FilelockVersion, $FsspecVersion, $MarkupsafeVersion,
    $NetworkxVersion, $PillowVersion, $TypingExtensionsVersion,
    $FasterWhisperVersion, $CTranslate2Version, $AvVersion, $PsutilVersion,
    $HfXetVersion, $SetuptoolsVersion, $lockHash, $pyprojectHash,
    $RuntimeCompatRevision
) -join "`n")
$depsState = Read-State "python-deps.sha256"

# Fast path: validate the actual final Windows environment first. The state
# marker is only a cache; an old marker must not force a destructive lock sync
# when the installed environment is already healthy.
$depsProbe = @"
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*')
from importlib.metadata import version
import sys
sys.path.insert(0, r'$FishProject')
import numpy, numba, torch, torchvision, torchaudio, gradio, faster_whisper
import lightning, hydra, tensorboard, fish_speech, datasets, pyarrow

expected = {
    'torch': '$TorchVersion',
    'torchvision': '$TorchVisionVersion',
    'torchaudio': '$TorchVersion',
    'numpy': '$NumpyVersion',
    'gradio': '$GradioVersion',
    'gradio-client': '$GradioClientVersion',
    'pyarrow': '$PyArrowVersion',
    'protobuf': '$ProtobufVersion',
    'filelock': '$FilelockVersion',
    'fsspec': '$FsspecVersion',
    'markupsafe': '$MarkupsafeVersion',
    'networkx': '$NetworkxVersion',
    'pillow': '$PillowVersion',
    'typing-extensions': '$TypingExtensionsVersion',
    'faster-whisper': '$FasterWhisperVersion',
    'ctranslate2': '$CTranslate2Version',
    'av': '$AvVersion',
    'psutil': '$PsutilVersion',
    'hf-xet': '$HfXetVersion',
    'setuptools': '$SetuptoolsVersion',
    'triton-windows': '$TritonVersion',
}
for name, wanted in expected.items():
    actual = version(name)
    if name in {'torch', 'torchvision', 'torchaudio'}:
        assert actual.split('+')[0] == wanted, (name, actual, wanted)
    else:
        assert actual == wanted, (name, actual, wanted)

assert torch.version.cuda == '12.8', torch.version.cuda
assert hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'), 'torchvision::nms is not registered'
assert hasattr(pyarrow, 'PyExtensionType')
from datasets.download.streaming_download_manager import xopen
from fish_speech.datasets.semantic import SemanticDataModule
print('Frozen environment OK | numpy', numpy.__version__, '| numba', numba.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| datasets', datasets.__version__, '| pyarrow', pyarrow.__version__)
"@
$DepsReady = $false
$existingProbe = Invoke-PythonProbe $depsProbe
if ($existingProbe.ExitCode -eq 0) {
    $DepsReady = $true
    if ($depsState -eq $depsFingerprint) {
        Skip "Frozen Windows environment is already healthy; uv.lock sync is skipped."
    } else {
        Skip "Healthy final Windows environment detected; refreshing the local state marker and skipping uv.lock sync."
        Write-State "python-deps.sha256" $depsFingerprint
    }
    if ($existingProbe.StdOut) { Write-Host ("[OK] " + $existingProbe.StdOut.Trim()) -ForegroundColor Green }
} elseif ($depsState -eq $depsFingerprint) {
    Write-Host "[INFO] Environment fingerprint matches, but the health check failed; repairing from uv.lock..." -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Existing Python/Fish environment is incomplete or incompatible; repairing from uv.lock..." -ForegroundColor Yellow
}

if (-not $DepsReady) {

# The checked-in lock is authoritative for Fish and training dependencies.
# Force the project sync into the Easy GUI's project-local .venv.
$oldUvProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$env:UV_PROJECT_ENVIRONMENT = $Venv
try {
    & $UvExe sync --project $FishProject --frozen --extra preprocess --inexact --no-cache --no-install-project `
        --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
    if ($LASTEXITCODE -ne 0) { throw "uv sync --frozen failed. uv.lock and pyproject.toml must remain a matched pair." }
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $oldUvProjectEnvironment
}

# IMPORTANT: the upstream Fish lock is cross-platform and may replace the
# Windows CUDA PyTorch family with a registry/CPU torchvision build during sync.
# Re-assert the official, ABI-matched CUDA 12.8 triplet AFTER uv sync.
# PyTorch's supported 2.8.0 matrix is torch 2.8.0 + torchvision 0.23.0 +
# torchaudio 2.8.0 from the same cu128 index.
& $UvExe pip install --python $PythonExe --no-cache --no-deps `
    "torch==$TorchVersion" "torchvision==$TorchVisionVersion" "torchaudio==$TorchVersion" `
    --index-url "https://download.pytorch.org/whl/cu128"
if ($LASTEXITCODE -ne 0) { throw "Post-lock CUDA 12.8 PyTorch ABI alignment failed." }

$torchAbiProbe = @"
import torch, torchvision, torchaudio
assert torch.__version__.split('+')[0] == '$TorchVersion', torch.__version__
assert torchvision.__version__.split('+')[0] == '$TorchVisionVersion', torchvision.__version__
assert torchaudio.__version__.split('+')[0] == '$TorchVersion', torchaudio.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'), 'torchvision::nms is not registered'
print('CUDA ABI OK |', torch.__version__, '|', torchvision.__version__, '|', torchaudio.__version__)
"@
$torchAbiResult = Invoke-PythonProbe $torchAbiProbe
if ($torchAbiResult.ExitCode -ne 0) {
    Write-Host "[ERROR] CUDA PyTorch/torchvision ABI probe failed:" -ForegroundColor Red
    if ($torchAbiResult.StdOut) { Write-Host $torchAbiResult.StdOut }
    if ($torchAbiResult.StdErr) { Write-Host $torchAbiResult.StdErr -ForegroundColor Red }
    throw "PyTorch CUDA ABI alignment failed."
}
if ($torchAbiResult.StdOut) { Write-Host ("[OK] " + $torchAbiResult.StdOut.Trim()) -ForegroundColor Green }

# Small Windows/Easy-GUI compatibility supplement applied AFTER the valid upstream frozen sync.
# PyArrow 20 is intentionally pinned here: upstream Fish currently resolves PyArrow 21,
# while the bundled datasets/training stack still requires the removed PyExtensionType API.
# Keeping the upstream uv.lock intact preserves its complete wheel/source metadata.
# --no-deps is intentional: the lock already installed the dependency graph, and
# resolving these overrides with dependencies would upgrade unrelated packages.
$GuiRuntimeSpec = @(
    "gradio==$GradioVersion",
    "gradio-client==$GradioClientVersion",
    "faster-whisper==$FasterWhisperVersion",
    "ctranslate2==$CTranslate2Version",
    "av==$AvVersion",
    "psutil==$PsutilVersion",
    "hf-xet==$HfXetVersion",
    "protobuf==$ProtobufVersion",
    "setuptools==$SetuptoolsVersion",
    "numpy==$NumpyVersion",
    "pyarrow==$PyArrowVersion",
    "filelock==$FilelockVersion",
    "fsspec==$FsspecVersion",
    "markupsafe==$MarkupsafeVersion",
    "networkx==$NetworkxVersion",
    "pillow==$PillowVersion",
    "typing-extensions==$TypingExtensionsVersion",
    "triton-windows==$TritonVersion"
)
& $UvExe pip install --python $PythonExe --no-cache --no-deps @GuiRuntimeSpec
if ($LASTEXITCODE -ne 0) { throw "Pinned Easy GUI runtime supplement installation failed." }

$finalProbe = Invoke-PythonProbe $depsProbe
if ($finalProbe.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Frozen environment health check failed:" -ForegroundColor Red
    if ($finalProbe.StdOut) { Write-Host $finalProbe.StdOut }
    if ($finalProbe.StdErr) { Write-Host $finalProbe.StdErr -ForegroundColor Red }
    throw "The locked environment is not healthy; see the Python traceback above."
}
if ($finalProbe.StdOut) { Write-Host ("[OK] " + $finalProbe.StdOut.Trim()) -ForegroundColor Green }
Write-State "python-deps.sha256" $depsFingerprint

if (Test-Path $UvCache) {
    Remove-Item -Recurse -Force $UvCache
    Ensure-Directory $UvCache
}
}

# s2.cpp paths are needed before the prebuilt-runtime gate. In older installer
# flow these were initialized later, inside the local-build section.
$S2Root = Join-Path $Root "modules\s2.cpp"
if (-not (Test-Path $S2Root)) { throw "s2.cpp source directory not found: $S2Root" }

# Prefer a published universal s2.cpp runtime over local compilation.
# Keep this as a raw direct asset URL; replace it with another release asset when needed.
# ZIP root must contain s2.exe, ggml-cpu.dll, ggml-cuda.dll, ggml-vulkan.dll
# the CUDA runtime DLLs required by ggml-cuda, and the x64 MSVC CRT/OpenMP
# runtime DLLs required on Windows machines without Visual Studio installed.
$S2PrebuiltUrl = "https://huggingface.co/Blakus/llama-cpp-custom-libs/resolve/main/fish_speech_s2-runtime-windows-x64.zip?download=true"
$PublishedBin = Join-Path $S2Root "build\bin"
$PublishedExe = Join-Path $PublishedBin "s2.exe"

function Test-PublishedS2Runtime {
    if (-not (Test-Path $PublishedExe)) { return $false }
    foreach ($dll in @("ggml-cpu.dll", "ggml-cuda.dll", "ggml-vulkan.dll")) {
        if (-not (Test-Path (Join-Path $PublishedBin $dll))) { return $false }
    }
    $out = Join-Path $Temp "s2-prebuilt-help.out"
    $err = Join-Path $Temp "s2-prebuilt-help.err"
    try {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            Push-Location $PublishedBin
            & $PublishedExe --help 1> $out 2> $err
            $exitCode = $LASTEXITCODE
        } finally {
            Pop-Location
            $ErrorActionPreference = $previousErrorAction
        }
        if ($exitCode -ne 0) { return $false }
        $help = ""
        if (Test-Path $out) { $help += Get-Content -Raw $out }
        if (Test-Path $err) { $help += Get-Content -Raw $err }
        return ($help -match '(?i)--cuda' -and $help -match '(?i)--vulkan')
    } catch { return $false }
    finally { Remove-Item -Force -ErrorAction SilentlyContinue $out, $err }
}

$PrebuiltReady = Test-PublishedS2Runtime
if ($PrebuiltReady) {
    Skip "Existing universal s2.exe is healthy; CMake, Ninja, Vulkan SDK, MSVC and compilation are skipped."
} elseif ($S2PrebuiltUrl -notmatch "REPLACE-WITH-YOUR-REPO") {
    Header "5/8 - Prebuilt universal s2.cpp runtime"
    $runtimeZip = Join-Path $Downloads "s2-runtime-windows-x64.zip"
    Download-Cached $S2PrebuiltUrl $runtimeZip
    if (Test-Path $PublishedBin) { Remove-Item -Recurse -Force $PublishedBin }
    Ensure-Directory $PublishedBin
    Expand-ZipFast $runtimeZip $PublishedBin
    $PrebuiltReady = Test-PublishedS2Runtime
    if (-not $PrebuiltReady) { throw "Downloaded s2.cpp runtime failed validation." }
    Ok "Prebuilt CPU + CUDA + Vulkan s2.exe validated; local compilation is skipped."
}

if (-not $PrebuiltReady) {
Header "5/8 - Local CMake + Ninja"
$CMakeExe = Join-Path $CMakeRoot "bin\cmake.exe"
$cmakeReady = $false
if (Test-Path $CMakeExe) {
    try { $cmakeReady = ((& $CMakeExe --version | Select-Object -First 1) -match [regex]::Escape($CMakeVersion)) } catch {}
}
if (-not $cmakeReady) {
    $zip = Join-Path $Downloads "cmake-$CMakeVersion-windows-x86_64.zip"
    Download-Cached "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/cmake-$CMakeVersion-windows-x86_64.zip" $zip
    $stage = Join-Path $Temp "cmake-stage"
    Expand-ZipFast $zip $stage
    $inner = Get-ChildItem $stage -Directory | Select-Object -First 1
    if (-not $inner) { throw "CMake archive layout is invalid." }
    if (Test-Path $CMakeRoot) { Remove-Item -Recurse -Force $CMakeRoot }
    Move-Item $inner.FullName $CMakeRoot
    Remove-Item -Recurse -Force $stage
} else { Skip "CMake $CMakeVersion is already installed." }

$NinjaExe = Join-Path $NinjaRoot "ninja.exe"
$ninjaReady = $false
if (Test-Path $NinjaExe) {
    try { $ninjaReady = ((& $NinjaExe --version) -match [regex]::Escape($NinjaVersion)) } catch {}
}
if (-not $ninjaReady) {
    $zip = Join-Path $Downloads "ninja-$NinjaVersion-windows-x64.zip"
    Download-Cached "https://github.com/ninja-build/ninja/releases/download/v$NinjaVersion/ninja-win.zip" $zip
    Expand-ZipFast $zip $NinjaRoot
} else { Skip "Ninja $NinjaVersion is already installed." }
Ok "CMake + Ninja are project-local and reusable."

Header "6/8 - Local Vulkan SDK $VulkanVersion"
$VulkanInfo = Join-Path $VulkanRoot "Bin\vulkaninfo.exe"
$VulkanHeader = Join-Path $VulkanRoot "Include\vulkan\vulkan_core.h"
$VulkanLib = Join-Path $VulkanRoot "Lib\vulkan-1.lib"
$Glslc = Join-Path $VulkanRoot "Bin\glslc.exe"
$VulkanStamp = Join-Path $State "vulkan-$VulkanVersion.ready"

# A successful local SDK is immutable for this pinned installer version.  The
# marker avoids rescanning/reinstalling Vulkan on every run.  For migration from
# an older installer, validate the existing files once and create the marker.
$vulkanReady = Test-Path $VulkanStamp
if (-not $vulkanReady) {
    $vulkanReady = (Test-Path $VulkanInfo) -and (Test-Path $VulkanHeader) -and `
                   (Test-Path $VulkanLib) -and (Test-Path $Glslc)
    if ($vulkanReady) {
        try {
            $vulkanReady = Select-String -Path $VulkanHeader `
                -Pattern "#define VK_HEADER_VERSION\s+$VulkanHeaderVersion" -Quiet
        } catch { $vulkanReady = $false }
    }
    if ($vulkanReady) {
        Set-Content -Path $VulkanStamp -Value $VulkanVersion -Encoding ASCII
        Skip "Existing Vulkan SDK $VulkanVersion validated; persistent state marker created."
    }
}

if (-not $vulkanReady) {
    $vkInstaller = Join-Path $Downloads "vulkansdk-windows-X64-$VulkanVersion.exe"
    Download-Cached "https://sdk.lunarg.com/sdk/download/$VulkanVersion/windows/vulkansdk-windows-X64-$VulkanVersion.exe" $vkInstaller
    if (Test-Path $VulkanRoot) { Remove-Item -Recurse -Force $VulkanRoot }
    Info "Installing Vulkan SDK locally (copy_only=1)."
    $p = Start-Process -FilePath $vkInstaller -ArgumentList @(
        "--root", ('"' + $VulkanRoot + '"'), "--accept-licenses", "--default-answer",
        "--confirm-command", "install", "copy_only=1"
    ) -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Vulkan SDK installer failed with exit code $($p.ExitCode)." }
    if (-not ((Test-Path $VulkanHeader) -and (Test-Path $VulkanLib) -and (Test-Path $Glslc))) {
        throw "Vulkan SDK installer completed but required SDK files are missing."
    }
    Set-Content -Path $VulkanStamp -Value $VulkanVersion -Encoding ASCII
} else {
    Skip "Vulkan SDK $VulkanVersion already validated; download/install step skipped entirely."
}
$env:VULKAN_SDK = $VulkanRoot
$env:VK_SDK_PATH = $VulkanRoot
$env:Path = (Join-Path $VulkanRoot "Bin") + ";" + $env:Path
Ok "Vulkan SDK is isolated under .runtime\vulkan."

function Find-VcVars64 {
    # Prefer vswhere's newest compatible Visual Studio instance. This avoids
    # mixing an older vcvars environment with a newer cl.exe already on PATH.
    $vswhereCandidates = @()
    if (${env:ProgramFiles(x86)}) {
        $vswhereCandidates += (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($env:ProgramFiles) {
        $vswhereCandidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    foreach ($vswhere in $vswhereCandidates) {
        if (-not (Test-Path $vswhere)) { continue }
        try {
            $installPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1)
            if ($installPath) {
                $candidate = Join-Path $installPath.Trim() "VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path $candidate) { return $candidate }
            }
        } catch {}
    }

    # Explicit fallbacks. Check every 2022 location before any 2019 location.
    $roots = @()
    if ($env:ProgramFiles)        { $roots += (Join-Path $env:ProgramFiles "Microsoft Visual Studio") }
    if (${env:ProgramFiles(x86)}) { $roots += (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio") }

    foreach ($year in @("2022", "2019")) {
        foreach ($base in $roots) {
            foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
                $candidate = Join-Path $base "$year\$edition\VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }

    # Legacy migration path only; no new private VS installation is created.
    $legacyPrivate = Join-Path $VsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $legacyPrivate) { return $legacyPrivate }
    return $null
}

Header "7/8 - Visual C++ Build Tools"
$VcVars = Find-VcVars64
if ($VcVars) {
    Skip "Existing MSVC x64 toolchain detected: $VcVars"
} else {
    # Prefer Windows Package Manager.  Microsoft documents winget as a
    # supported way to install MSVC/Visual Studio Build Tools.  Do not create a
    # second pseudo-portable VS instance under .runtime: that path is fragile
    # and is what caused bootstrapper exit code 1 on some Windows systems.
    $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if ($winget) {
        Info "MSVC was not found. Installing Visual Studio Build Tools with winget. An elevation prompt may appear."
        $override = "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        & $winget.Source install --exact --id Microsoft.VisualStudio.BuildTools --source winget `
            --accept-source-agreements --accept-package-agreements --disable-interactivity `
            --override $override
        $wingetExit = $LASTEXITCODE
        if ($wingetExit -notin @(0,3010)) {
            Warn "winget returned exit code $wingetExit. Trying the cached Microsoft bootstrapper fallback."
        }
    } else {
        Warn "winget.exe is unavailable; using the Microsoft Build Tools bootstrapper fallback."
        $wingetExit = 1
    }

    $VcVars = Find-VcVars64
    if (-not $VcVars) {
        $vsBootstrap = Join-Path $Downloads "vs_BuildTools.exe"
        Download-Cached "https://aka.ms/vs/17/release/vs_BuildTools.exe" $vsBootstrap
        Info "Installing Visual C++ Build Tools to the standard Visual Studio location. An elevation prompt may appear."

        # Run elevated and let Visual Studio choose its supported installation
        # directory.  Avoid --installPath into .runtime; VS Build Tools is not
        # designed as a portable toolchain and that arrangement can fail with
        # the generic bootstrapper exit code 1.
        $vsArgs = '--passive --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
        try {
            $proc = Start-Process -FilePath $vsBootstrap -ArgumentList $vsArgs -Verb RunAs -Wait -PassThru
            if ($proc.ExitCode -notin @(0,3010)) {
                throw "Visual C++ Build Tools bootstrapper failed with exit code $($proc.ExitCode)."
            }
        } catch {
            throw "Visual C++ Build Tools installation failed. $($_.Exception.Message)"
        }
        $VcVars = Find-VcVars64
    }
}

if (-not $VcVars) {
    throw "MSVC x64 was not found after installation. Open Visual Studio Installer and add 'Desktop development with C++', then run install.ps1 again."
}
Ok "MSVC x64 toolchain ready."

function Find-WindowsSdkBin {
    $roots = New-Object System.Collections.Generic.List[string]
    if (${env:ProgramFiles(x86)}) { $roots.Add((Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin")) }
    if ($env:ProgramFiles)        { $roots.Add((Join-Path $env:ProgramFiles "Windows Kits\10\bin")) }

    foreach ($base in $roots) {
        if (-not (Test-Path $base)) { continue }
        $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
            Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } } -Descending
        foreach ($ver in $versions) {
            $x64 = Join-Path $ver.FullName "x64"
            if ((Test-Path (Join-Path $x64 "rc.exe")) -and (Test-Path (Join-Path $x64 "mt.exe"))) {
                return $x64
            }
        }
    }
    return $null
}

$WindowsSdkBin = Find-WindowsSdkBin
if (-not $WindowsSdkBin) {
    throw "Windows SDK tools rc.exe/mt.exe were not found. Open Visual Studio Installer and add a Windows 10/11 SDK component, then rerun install.ps1."
}
$RcExe = Join-Path $WindowsSdkBin "rc.exe"
$MtExe = Join-Path $WindowsSdkBin "mt.exe"
Skip "Windows SDK tools detected: $WindowsSdkBin"

Header "8/8 - s2.cpp universal CPU + CUDA + Vulkan build"
$Build = Join-Path $Runtime "s2-build-universal"

function Get-CudaVersion([string]$ToolkitRoot) {
    if (-not $ToolkitRoot) { return $null }
    $nvcc = Join-Path $ToolkitRoot "bin\nvcc.exe"
    if (-not (Test-Path $nvcc)) { return $null }
    try {
        $output = & $nvcc --version 2>&1 | Out-String
        if ($output -match 'release\s+(\d+\.\d+)') { return [version]("$($Matches[1]).0") }
    } catch {}
    return $null
}

function Find-CudaToolkit {
    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($name in @("CUDA_PATH", "CUDA_PATH_V13_4", "CUDA_PATH_V13_3", "CUDA_PATH_V13_2", "CUDA_PATH_V13_1", "CUDA_PATH_V13_0", "CUDA_PATH_V12_9", "CUDA_PATH_V12_8", "CUDA_PATH_V12_6", "CUDA_PATH_V12_5", "CUDA_PATH_V12_4")) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, "Machine") }
        if ($value) { $candidates.Add($value) }
    }

    $base = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
            Sort-Object { try { [version]($_.Name.TrimStart('v') + '.0') } catch { [version]'0.0' } } -Descending |
            ForEach-Object { $candidates.Add($_.FullName) }
    }

    $valid = @()
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not $candidate) { continue }
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        $header = Join-Path $candidate "include\cuda_runtime.h"
        $cudart = Join-Path $candidate "lib\x64\cudart.lib"
        if ((Test-Path $nvcc) -and (Test-Path $header) -and (Test-Path $cudart)) {
            $version = Get-CudaVersion $candidate
            if ($version -and $version -ge [version]'12.4.0') {
                $valid += [pscustomobject]@{ Path = $candidate; Version = $version }
            }
        }
    }

    # Prefer the newest CUDA 12.x toolkit when available. The bundled Python
    # runtime is cu128 and CUDA 12.x gives the broadest binary compatibility
    # with the current Windows ggml tree. Fall back to newer major versions.
    return $valid | Sort-Object @{ Expression = { if ($_.Version.Major -eq 12) { 1 } else { 0 } }; Descending = $true }, @{ Expression = { $_.Version }; Descending = $true } | Select-Object -First 1
}

$CudaToolkit = Find-CudaToolkit
if (-not $CudaToolkit) {
    throw @"
CUDA Toolkit 12.4 or newer was not found. The Python CUDA runtime is not enough to compile ggml CUDA.
This Easy GUI intentionally builds the complete s2.cpp runtime (CPU + CUDA + Vulkan) instead of silently downgrading it.
Install an NVIDIA CUDA Toolkit >= 12.4, then rerun install.bat. Existing Python/uv/Vulkan/MSVC components will be reused.
"@
}
$CudaRoot = $CudaToolkit.Path
$CudaVersion = $CudaToolkit.Version
$NvccExe = Join-Path $CudaRoot "bin\nvcc.exe"
Skip "CUDA Toolkit $CudaVersion detected: $CudaRoot"

function Find-S2BuiltExe([string]$BuildRoot) {
    $candidates = @(
        (Join-Path $BuildRoot "s2.exe"),
        (Join-Path $BuildRoot "Release\s2.exe"),
        (Join-Path $BuildRoot "bin\s2.exe"),
        (Join-Path $BuildRoot "bin\Release\s2.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $found = Get-ChildItem -Path $BuildRoot -Filter "s2.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "[\\/]CMakeFiles[\\/]" } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

function Test-S2UniversalCache([string]$BuildRoot) {
    $cache = Join-Path $BuildRoot "CMakeCache.txt"
    if (-not (Test-Path $cache)) { return $false }
    $text = Get-Content -Raw $cache
    return (
        $text -match '(?m)^S2_CUDA:BOOL=ON\s*$' -and
        $text -match '(?m)^S2_VULKAN:BOOL=ON\s*$' -and
        $text -match '(?m)^GGML_CUDA:BOOL=ON\s*$' -and
        $text -match '(?m)^GGML_VULKAN:BOOL=ON\s*$'
    )
}

function Get-S2RuntimeFingerprint([string]$BuildRoot, [string]$ExePath) {
    if (-not $ExePath -or -not (Test-Path $ExePath)) { return "" }
    $files = New-Object System.Collections.Generic.List[System.IO.FileInfo]
    $files.Add((Get-Item $ExePath))
    Get-ChildItem -Path $BuildRoot -Filter "*.dll" -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "[\\/]CMakeFiles[\\/]" } |
        ForEach-Object { $files.Add($_) }
    $rows = $files | Sort-Object Name, FullName | ForEach-Object {
        "$($_.Name)|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
    }
    return Get-TextSha256 ($rows -join "`n")
}

function Publish-S2Runtime([string]$BuildRoot, [string]$ExePath, [string]$Destination) {
    if (-not $ExePath -or -not (Test-Path $ExePath)) { throw "s2.exe was not found after a successful build." }
    Ensure-Directory $Destination

    Copy-Item -LiteralPath $ExePath -Destination (Join-Path $Destination "s2.exe") -Force

    $dlls = Get-ChildItem -Path $BuildRoot -Filter "*.dll" -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "[\\/]CMakeFiles[\\/]" } |
        Group-Object Name
    foreach ($group in $dlls) {
        $dll = $group.Group | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $Destination $dll.Name) -Force
    }

    # Copy the CUDA runtime dependencies normally required by ggml-cuda beside
    # s2.exe. This makes the published C++ runtime independent of PATH ordering
    # while leaving the NVIDIA display driver (nvcuda.dll) system-provided.
    $cudaBin = Join-Path $CudaRoot "bin"
    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "nvJitLink_*.dll", "nvrtc64_*.dll")) {
        Get-ChildItem -Path $cudaBin -Filter $pattern -File -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force }
    }

    $outExe = Join-Path $Destination "s2.exe"
    if (-not (Test-Path $outExe)) { throw "s2.exe publishing failed." }
    return $outExe
}

$sourceFingerprint = Get-TreeFingerprint $S2Root
$universalFingerprint = Get-TextSha256 (($sourceFingerprint + "`nbackend=cpu+cuda+vulkan`ncuda=" + $CudaVersion + "`ncudaRoot=" + $CudaRoot + "`nmsvc=" + $VcVars))
$builtFingerprint = Read-State "s2-universal-build.sha256"
$S2Exe = Find-S2BuiltExe $Build
$needsBuild = $true

if ($S2Exe -and (Test-S2UniversalCache $Build) -and ($builtFingerprint -eq $universalFingerprint)) {
    $needsBuild = $false
    Skip "Universal s2.cpp CPU + CUDA + Vulkan build is current; compilation skipped."
}

if ($needsBuild) {
    Ensure-Directory $Build

    # Keep both the source and build tree at their real project-local paths.
    # Older versions used a temporary drive mapping to shorten the path, which
    # made the build depend on state outside the self-contained project and
    # could leave that mapping behind after an interrupted CUDA build.
    $sdkBin = $WindowsSdkBin
    $cudaCmakeRoot = ($CudaRoot -replace '\\','/')
    $nvccCmake = ($NvccExe -replace '\\','/')
    $shaderGenBuild = (Join-Path $Runtime "vulkan-shaders-gen") -replace '\\','/'
    $cmakeFilesDir = Join-Path $Build "CMakeFiles"
    if (Test-Path $cmakeFilesDir) {
        Get-ChildItem -Path $cmakeFilesDir -Filter "CMakeRCCompiler.cmake" -Recurse -File -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }

    $commonFlags = "/FS /O2 /DNDEBUG"
    $configure = "call `"$VcVars`" && set `"VULKAN_SDK=$VulkanRoot`" && set `"VK_SDK_PATH=$VulkanRoot`" && set `"CUDA_PATH=$CudaRoot`" && set `"PATH=$CMakeRoot\bin;$NinjaRoot;$VulkanRoot\Bin;$CudaRoot\bin;$sdkBin;%PATH%`" && `"$CMakeExe`" -S `"$S2Root`" -B `"$Build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OBJECT_PATH_MAX=180 -DCMAKE_C_FLAGS=`"$commonFlags`" -DCMAKE_CXX_FLAGS=`"$commonFlags`" -DCMAKE_RC_COMPILER=rc.exe -DCMAKE_MT=mt.exe -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON -DS2_CUDA=ON -DS2_VULKAN=ON -DGGML_VULKAN_SHADER_GEN_BUILD_DIR=`"$shaderGenBuild`" -DCUDAToolkit_ROOT=`"$cudaCmakeRoot`" -DCMAKE_CUDA_COMPILER=`"$nvccCmake`""
    cmd.exe /d /s /c $configure
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp universal CMake configure failed." }

    if (-not (Test-S2UniversalCache $Build)) {
        throw "CMake configured, but CPU/CUDA/Vulkan universal backend flags were not all enabled."
    }

    $buildCmd = "call `"$VcVars`" && set `"VULKAN_SDK=$VulkanRoot`" && set `"VK_SDK_PATH=$VulkanRoot`" && set `"CUDA_PATH=$CudaRoot`" && set `"PATH=$CMakeRoot\bin;$NinjaRoot;$VulkanRoot\Bin;$CudaRoot\bin;$sdkBin;%PATH%`" && `"$CMakeExe`" --build `"$Build`" --parallel"
    cmd.exe /d /s /c $buildCmd
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp universal build failed." }

    $S2Exe = Find-S2BuiltExe $Build
    if (-not $S2Exe) { throw "s2.exe was not produced by the universal build." }
    if (-not (Test-S2UniversalCache $Build)) { throw "Universal backend validation failed after build." }
    Write-State "s2-universal-build.sha256" $universalFingerprint
    Ok "s2.cpp universal CPU + CUDA + Vulkan compilation completed."
}

# Publishing is independent from compilation. A failed copy never forces a
# costly CUDA/Vulkan rebuild on the next installer run.
$S2Exe = Find-S2BuiltExe $Build
if (-not $S2Exe) { throw "No completed universal s2.cpp binary is available to publish." }
if (-not (Test-S2UniversalCache $Build)) { throw "Existing s2.cpp build is not a validated CPU + CUDA + Vulkan build." }
$runtimeFingerprint = Get-S2RuntimeFingerprint $Build $S2Exe
$publishedFingerprint = Read-State "s2-universal-runtime.sha256"
$publishedBackendsPresent = (
    (Test-Path $PublishedExe) -and
    (Test-Path (Join-Path $PublishedBin "ggml-cpu.dll")) -and
    (Test-Path (Join-Path $PublishedBin "ggml-cuda.dll")) -and
    (Test-Path (Join-Path $PublishedBin "ggml-vulkan.dll"))
)

if ($publishedBackendsPresent -and $publishedFingerprint -eq $runtimeFingerprint) {
    Skip "Published universal s2.cpp runtime is already current."
} else {
    $PublishedExe = Publish-S2Runtime $Build $S2Exe $PublishedBin
    Write-State "s2-universal-runtime.sha256" $runtimeFingerprint
    Ok "Universal s2.cpp runtime published to modules\s2.cpp\build\bin."
}

# Validate CLI surface and persist an explicit backend manifest. No model is
# loaded, so this takes only a moment and does not test inference.
$smokeOut = Join-Path $Temp "s2-smoke.out"
$smokeErr = Join-Path $Temp "s2-smoke.err"
try {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        Push-Location $PublishedBin
        & $PublishedExe --help 1> $smokeOut 2> $smokeErr
        $smokeExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $previousErrorAction
    }
    $helpOut = if (Test-Path $smokeOut) { Get-Content -Raw $smokeOut } else { "" }
    $helpErr = if (Test-Path $smokeErr) { Get-Content -Raw $smokeErr } else { "" }
    $helpText = $helpOut + "`n" + $helpErr
    if ($smokeExitCode -ne 0) { throw "Published s2.exe could not start (exit code $smokeExitCode)." }
    if ($helpText -notmatch '(?i)--cuda' -or $helpText -notmatch '(?i)--vulkan') {
        throw "Published s2.exe started, but its CLI does not expose both CUDA and Vulkan backends."
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $smokeOut, $smokeErr
}

$exeHash = (Get-FileHash -Algorithm SHA256 $PublishedExe).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    schema = 1
    backend = "universal"
    cpu = $true
    cuda = $true
    vulkan = $true
    quantization = @("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K", "Q2_K")
    cuda_toolkit = $CudaVersion.ToString()
    vulkan_sdk = $VulkanVersion
    sha256 = $exeHash
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $PublishedBin "s2-runtime.json")
Ok "s2.cpp universal runtime validated: CPU + CUDA + Vulkan + GGUF quant variants."

}

Header "Fast verification"
& $PythonExe -c "import torch, gradio; print('torch=',torch.__version__,'cuda=',torch.version.cuda,'available=',torch.cuda.is_available()); print('gradio=',gradio.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python runtime verification failed." }
Write-Host "`nInstallation is ready." -ForegroundColor Green
Write-Host "Python/Fish dependencies were validated against the checked-in uv.lock and Windows compatibility pins. A healthy existing/prebuilt universal s2.exe bypasses the entire local C++ toolchain and compilation path." -ForegroundColor Green
Write-Host "Run start.bat. Models are downloaded on demand with hf-xet." -ForegroundColor Green
