# Fish Speech S2 Pro Installer
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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

function Get-VsWherePath {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Test-MsvcToolchain($vcvars) {
    if (-not $vcvars -or -not (Test-Path $vcvars)) {
        return $false
    }

    cmd /d /s /c "call `"$vcvars`" >nul 2>&1 && where cl.exe >nul 2>&1"
    return ($LASTEXITCODE -eq 0)
}

function Find-MsvcToolchain {
    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $installPath = & $vswhere -latest -products * -version "[17.0,18.0)" `
            -requires Microsoft.VisualStudio.Workload.NativeDesktop Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($installPath) {
            $vcvars = Join-Path ($installPath | Select-Object -First 1) "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-MsvcToolchain $vcvars) {
                return $vcvars
            }
        }
    }

    $vsBase = "C:\Program Files\Microsoft Visual Studio\2022"
    foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
        $vcvars = Join-Path $vsBase "$edition\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-MsvcToolchain $vcvars) {
            return $vcvars
        }
    }
    return $null
}

function Get-S2Executable {
    $s2Root = Join-Path $PSScriptRoot "modules\s2.cpp"
    $candidates = @(
        (Join-Path $s2Root "build\bin\Release\s2.exe"),
        (Join-Path $s2Root "build\Release\s2.exe"),
        (Join-Path $s2Root "build\bin\s2.exe"),
        (Join-Path $s2Root "build\s2.exe"),
        (Join-Path $s2Root "s2.exe")
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Test-S2BuildNeeded($s2Executable) {
    if (-not $s2Executable -or -not (Test-Path $s2Executable)) {
        return $true
    }

    $sourceRoot = Join-Path $PSScriptRoot "modules\s2.cpp"
    $latestSource = Get-ChildItem $sourceRoot -Recurse -File |
        Where-Object {
            $_.FullName -notlike "$sourceRoot\build\*" -and
            ($_.Extension -in @(".c", ".cpp", ".cu", ".h", ".hpp") -or $_.Name -eq "CMakeLists.txt")
        } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1

    return ($latestSource -and $latestSource.LastWriteTimeUtc -gt (Get-Item $s2Executable).LastWriteTimeUtc)
}

function Stop-MissingMsvc {
    Write-Err "Visual Studio 2022 with the C++ toolchain was not detected."
    Write-Host "Install Visual Studio 2022 Community from:"
    Write-Host "  https://aka.ms/vs/17/release/vs_community.exe"
    Write-Host "Select the workload:"
    Write-Host "  Desktop development with C++"
    Write-Host "Then rerun install.bat."
    throw "Required Visual Studio 2022 C++ toolchain is unavailable."
}

function Refresh-SystemPath {
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:PATH = (($userPath, $machinePath, $env:PATH) | Where-Object { $_ }) -join ";"
}

function Install-Cuda129ForLegacyGpu {
    $cudaVersion = "12.9.1"
    $installerName = "cuda_${cudaVersion}_windows_network.exe"
    $installerUrl = "https://developer.download.nvidia.com/compute/cuda/$cudaVersion/network_installers/$installerName"
    $installerPath = Join-Path $env:TEMP $installerName

    Write-Header "Installing CUDA Toolkit $cudaVersion for Pascal/Volta"
    Write-Info "Downloading the official NVIDIA network installer..."
    Write-Info "The toolkit installation is several gigabytes and can take a while."

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

        $signature = Get-AuthenticodeSignature $installerPath
        if ($signature.Status -ne "Valid" -or
            $signature.SignerCertificate.Subject -notmatch "NVIDIA") {
            throw "The downloaded CUDA installer does not have a valid NVIDIA digital signature."
        }

        Write-Info "Running the CUDA installer silently. Approve the Windows administrator prompt if shown."
        $process = Start-Process -FilePath $installerPath -ArgumentList @("-s", "-n") `
            -Verb RunAs -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) {
            throw "CUDA $cudaVersion installer failed with exit code $($process.ExitCode)."
        }
        if ($process.ExitCode -eq 3010) {
            Write-Warn "CUDA installed successfully, but Windows recommends a reboot."
        }
    } catch {
        throw "Automatic CUDA $cudaVersion installation failed: $($_.Exception.Message)"
    } finally {
        if (Test-Path $installerPath) {
            Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
        }
    }

    Refresh-SystemPath
}

function Find-CMake {
    $command = Get-Command cmake -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "$env:ProgramFiles\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Get-CMakeVersion($cmakePath) {
    if (-not $cmakePath) {
        return $null
    }
    $firstLine = & $cmakePath --version | Select-Object -First 1
    if ($firstLine -match "cmake version (\d+\.\d+\.\d+)") {
        return [version]$Matches[1]
    }
    return $null
}

function Ensure-CMake {
    $minimumVersion = [version]"3.24.0"
    $cmakePath = Find-CMake
    $cmakeVersion = Get-CMakeVersion $cmakePath

    if ($cmakeVersion -and $cmakeVersion -ge $minimumVersion) {
        $env:PATH = (Split-Path $cmakePath) + ";" + $env:PATH
        Write-Ok "CMake $cmakeVersion is ready: $cmakePath"
        return
    }

    if ($cmakeVersion) {
        Write-Warn "CMake $cmakeVersion is too old. Version 3.24 or newer is required."
    } else {
        Write-Info "CMake 3.24 or newer was not found."
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Header "Installing CMake (via winget)"
        winget install --id Kitware.CMake --exact --silent --force `
            --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Winget failed to install CMake (exit code $LASTEXITCODE)."
        }
        Refresh-SystemPath
        $cmakePath = Find-CMake
        $cmakeVersion = Get-CMakeVersion $cmakePath
    }

    if (-not $cmakeVersion -or $cmakeVersion -lt $minimumVersion) {
        throw "CMake 3.24+ is required. Install it from https://cmake.org/download/ and rerun install.bat."
    }

    $env:PATH = (Split-Path $cmakePath) + ";" + $env:PATH
    Write-Ok "CMake $cmakeVersion is ready: $cmakePath"
}

function Refresh-UvPath {
    $paths = @(
        "$env:USERPROFILE\.local\bin",
        "$env:USERPROFILE\.cargo\bin",
        "$env:APPDATA\uv\bin",
        "$env:LOCALAPPDATA\uv\bin",
        "$env:LOCALAPPDATA\Programs\uv"
    )
    $env:PATH = (($paths + ($env:PATH -split ";")) | Where-Object { $_ } | Select-Object -Unique) -join ";"
}

function Ensure-Uv {
    Refresh-UvPath
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        Write-Info (& uv --version)
        return
    }

    Write-Info "UV not found. Installing..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install uv. Please install it manually from https://astral.sh/uv/"
    }

    Refresh-UvPath
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv was installed but is still not visible in PATH for this session. Open a new terminal and run install.bat again."
    }

    Write-Ok "$(& uv --version) is ready."
}

Write-Header "Fish Speech S2 Pro - Voice Clone & Training - Installer"

# -------------------------------------------------------
# UV
# -------------------------------------------------------
Write-Header "Checking UV"
Ensure-Uv

# -------------------------------------------------------
# CMake
# -------------------------------------------------------
Write-Header "Checking CMake"
Ensure-CMake

# -------------------------------------------------------
# Ninja (winget)
# -------------------------------------------------------
if (-not (Get-Command ninja -ErrorAction SilentlyContinue) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing Ninja (via winget)"
    try {
        winget install --id Ninja-build.Ninja --exact --silent --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Winget exited with code $LASTEXITCODE."
        }
        # Refresh Path for current session
        Refresh-SystemPath
        Refresh-UvPath
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
$gpuComputeMajor = $null
$gpuComputeMinor = $null

try {
    $raw = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>$null
    if (-not $raw) {
        $raw = & nvidia-smi --query-gpu=compute_cap --format=csv 2>$null | Select-Object -Skip 1
    }
    $sm = ($raw | Select-Object -First 1).Trim()

    if ($sm -match "^(\d+)\.(\d+)$") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        $gpuComputeMajor = $major
        $gpuComputeMinor = $minor

        if ($major -ge 12) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/nightly/cu128"
            $cudaNightly = $true;  $cudaLabel = "Blackwell sm_$sm (cu128 nightly)"
        } elseif ($major -ge 8) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/cu128"
            $cudaNightly = $false; $cudaLabel = "Ampere/Ada sm_$sm (cu128 stable)"
        } elseif ($major -eq 7 -and $minor -ge 5) {
            $cudaExtra = "cu128"; $cudaIndex = "https://download.pytorch.org/whl/cu128"
            $cudaNightly = $false; $cudaLabel = "Turing sm_$sm (cu128 stable)"
        } elseif ($major -eq 6) {
            $cudaExtra = "cu126"; $cudaIndex = "https://download.pytorch.org/whl/cu126"
            $cudaNightly = $false; $cudaLabel = "Pascal sm_$sm (legacy CUDA 12.x)"
        } else {
            $cudaExtra = "cu126"; $cudaIndex = "https://download.pytorch.org/whl/cu126"
            $cudaNightly = $false; $cudaLabel = "Volta/legacy sm_$sm (CUDA 12.x)"
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
# Visual Studio 2022 C++ toolchain
# -------------------------------------------------------
$vcvars = Find-MsvcToolchain

if (-not $vcvars -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing Visual Studio 2022 Community with C++ Workload (via winget)"
    Write-Info "This is a multi-gigabyte installation and can take a while."
    winget install --id Microsoft.VisualStudio.2022.Community --exact --force `
        --override "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended" `
        --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Automatic Visual Studio installation failed with exit code $LASTEXITCODE."
    }

    $vcvars = Find-MsvcToolchain
}

if (-not $vcvars) {
    Stop-MissingMsvc
}
Write-Ok "Verified Visual Studio 2022 C++ compiler via: $vcvars"

# -------------------------------------------------------
# CUDA Toolkit (winget)
# -------------------------------------------------------
$requiresCuda12 = ($gpuComputeMajor -ne $null -and $gpuComputeMajor -lt 7) -or
                  ($gpuComputeMajor -eq 7 -and $gpuComputeMinor -lt 5)

# Prefer installed toolkits that can target the detected GPU. CUDA 13 removed
# offline compilation and library support for pre-Turing architectures.
$baseCUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $baseCUDA) {
    $versions = Get-ChildItem $baseCUDA -Directory |
        Where-Object { $_.Name -match "^v(\d+)\.(\d+)$" } |
        ForEach-Object {
            $versionMatch = [regex]::Match($_.Name, "^v(\d+)\.(\d+)$")
            [PSCustomObject]@{
                Path = $_.FullName
                Version = [version]("$($versionMatch.Groups[1].Value).$($versionMatch.Groups[2].Value).0")
            }
        } |
        Where-Object {
            $_.Version -ge [version]"12.4.0" -and
            (-not $requiresCuda12 -or $_.Version.Major -eq 12)
        } |
        Sort-Object Version -Descending
    if ($versions) {
        $cudaToolkitPath = $versions[0].Path
    }
}

# Fallback to nvcc from PATH if it is compatible with the detected GPU.
if (-not $cudaToolkitPath -and (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    $nvccPath = (Get-Command nvcc).Source
    if ($nvccPath) {
        $candidateToolkit = (Get-Item $nvccPath).Directory.Parent.FullName
        $candidateOutput = & $nvccPath --version
        $candidateRelease = $candidateOutput | Select-String "release (\d+\.\d+)"
        if ($candidateRelease -and $candidateRelease.Matches.Count -gt 0) {
            $candidateVersion = [version]($candidateRelease.Matches[0].Groups[1].Value + ".0")
            if ($candidateVersion -ge [version]"12.4.0" -and
                (-not $requiresCuda12 -or $candidateVersion.Major -eq 12)) {
                $cudaToolkitPath = $candidateToolkit
            }
        }
    }
}

# Check versioned environment variables (e.g., CUDA_PATH_V12_4).
if (-not $cudaToolkitPath) {
    $envVars = Get-ChildItem Env: | Where-Object { $_.Name -match "^CUDA_PATH_V\d+_\d+$" } | Sort-Object Name -Descending
    foreach ($envVar in $envVars) {
        if ($envVar.Name -match "^CUDA_PATH_V(\d+)_(\d+)$") {
            $envVersion = [version]("$($Matches[1]).$($Matches[2]).0")
        } else {
            continue
        }
        if ($envVersion -ge [version]"12.4.0" -and
            (-not $requiresCuda12 -or $envVersion.Major -eq 12) -and
            (Test-Path $envVar.Value)) {
            $cudaToolkitPath = $envVar.Value
            break
        }
    }
}

if (-not $cudaToolkitPath -and $requiresCuda12) {
    Install-Cuda129ForLegacyGpu
    $expectedCuda129 = Join-Path $baseCUDA "v12.9"
    if (Test-Path (Join-Path $expectedCuda129 "bin\nvcc.exe")) {
        $cudaToolkitPath = $expectedCuda129
    }
}

if (-not $cudaToolkitPath -and -not $requiresCuda12 -and
    (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Header "Installing current CUDA Toolkit (via winget)"
    winget install --id Nvidia.CUDA --exact --silent `
        --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Automatic CUDA installation failed with exit code $LASTEXITCODE."
    }

    Refresh-SystemPath
    if (Test-Path $baseCUDA) {
        $latestCuda = Get-ChildItem $baseCUDA -Directory |
            Where-Object { $_.Name -match "^v\d+\.\d+$" } |
            Sort-Object {
                $match = [regex]::Match($_.Name, "^v(\d+)\.(\d+)$")
                [version]("$($match.Groups[1].Value).$($match.Groups[2].Value).0")
            } -Descending |
            Select-Object -First 1
        if ($latestCuda -and (Test-Path (Join-Path $latestCuda.FullName "bin\nvcc.exe"))) {
            $cudaToolkitPath = $latestCuda.FullName
        }
    }
}

$cudaVersion = $null
if ($cudaToolkitPath -and (Test-Path (Join-Path $cudaToolkitPath "bin\nvcc.exe"))) {
    $nvccVersionOutput = & (Join-Path $cudaToolkitPath "bin\nvcc.exe") --version
    $releaseLine = $nvccVersionOutput | Select-String "release (\d+\.\d+)"
    if ($releaseLine -and $releaseLine.Matches.Count -gt 0) {
        $cudaVersion = [version]($releaseLine.Matches[0].Groups[1].Value + ".0")
    }
}

if ($cudaToolkitPath -and (
    -not $cudaVersion -or
    $cudaVersion -lt [version]"12.4.0" -or
    ($requiresCuda12 -and $cudaVersion.Major -ne 12)
)) {
    Write-Warn "The selected CUDA Toolkit is not compatible with this GPU and bundled ggml backend."
    Write-Warn "Pascal/Volta require CUDA 12.4-12.9; Turing and newer require CUDA 12.4 or newer."
    $cudaToolkitPath = $null
}

if ($requiresCuda12 -and -not $cudaToolkitPath) {
    throw "CUDA 12.9.1 was not installed correctly. Pascal/Volta cannot be built with CUDA 13."
}

if (-not $cudaToolkitPath) {
    Write-Warn "CUDA Toolkit not detected."
    Write-Host "  S2.cpp requires:"
    Write-Host "    1. CUDA Toolkit  https://developer.nvidia.com/cuda-downloads"
    Write-Host "    2. Visual Studio 2022 with C++ workload"
    Write-Host "    3. CUDA VS integration: (found in CUDA extras folder)"
    Write-Warn "Skipping S2.cpp CUDA backend. Defaulting to CPU only."
    $cudaArgs = ""
} else {
    Write-Ok "Using CUDA Toolkit: $cudaToolkitPath"
}

Push-Location "modules\s2.cpp"
try {
    $s2Executable = Get-S2Executable
    $needsBuild = Test-S2BuildNeeded $s2Executable
    if (-not $needsBuild) {
        Write-Ok "S2.cpp is already compiled: $s2Executable"
    } else {
        if ($s2Executable) {
            Write-Info "S2.cpp sources changed. Rebuilding incrementally..."
        }
        if (-not (Test-Path "build")) {
            New-Item -ItemType Directory "build" | Out-Null
        }
        $buildDir = (Resolve-Path "build").Path

        $cmakeCache = Join-Path $buildDir "CMakeCache.txt"
        if (Test-Path $cmakeCache) {
            $generatorArgs = ""
            Write-Ok "Generator: existing CMake build directory"
        } else {
            $useNinja = (Get-Command ninja -ErrorAction SilentlyContinue)
            if ($useNinja) {
                $generatorArgs = '-G "Ninja"'
                Write-Ok "Generator: Ninja (Priority)"
            } else {
                $generatorArgs = '-G "Visual Studio 17 2022"'
                Write-Ok "Generator: Visual Studio 17 2022 (Fallback)"
            }
        }

        Write-Info "Configuring..."
        # Performance flags: /Ox (max opt), /arch:AVX2 (SIMD), /fp:fast,
        # /GL /LTCG (Link Time Optimization), /openmp (Multithreading)
        $buildArgs = "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON -DGGML_LTO=ON -DCMAKE_CXX_FLAGS=`"/Ox /arch:AVX2 /fp:fast /GL /DNDEBUG /openmp`" -DCMAKE_EXE_LINKER_FLAGS=`"/LTCG`" -DCMAKE_STATIC_LINKER_FLAGS=`"/LTCG`""
        if ($cudaToolkitPath) {
            $buildArgs += " -DS2_CUDA=ON -DCUDAToolkit_ROOT=`"$cudaToolkitPath`""
        }
            
        # Check for Vulkan SDK
        $vulkanSdk = $env:VULKAN_SDK
        if (-not $vulkanSdk) {
            $vulkanBase = "C:\VulkanSDK"
            if (Test-Path $vulkanBase) {
                $versions = Get-ChildItem $vulkanBase -Directory | Sort-Object Name -Descending
                if ($versions) { $vulkanSdk = $versions[0].FullName }
            }
        }
        if ($vulkanSdk -and (Test-Path $vulkanSdk)) {
            Write-Ok "Vulkan SDK found: $vulkanSdk - enabling Vulkan backend (k-quants GPU support)"
            $buildArgs += " -DS2_VULKAN=ON"
        } else {
            Write-Warn "Vulkan SDK not found. K-quant models (Q2_K-Q6_K) will only run on CPU."
            Write-Warn "Install Vulkan SDK from https://vulkan.lunarg.com/ for full GPU quant support."
        }
            
        cmd /d /s /c "call `"$vcvars`" && cd /d `"$buildDir`" && cmake .. $generatorArgs $buildArgs"
        if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

        Write-Info "Building..."
        cmd /d /s /c "call `"$vcvars`" && cd /d `"$buildDir`" && cmake --build . --config Release"
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

        $s2Executable = Get-S2Executable
        if (-not $s2Executable) {
            $expected = Join-Path $PSScriptRoot "modules\s2.cpp\build\bin\Release\s2.exe"
            throw "The build completed but s2.exe was not created. Expected output includes: $expected"
        }
        Write-Ok "S2.cpp compiled successfully."
    }
    Write-Ok "s2.exe location: $s2Executable"
} catch {
    Write-Err $_.Exception.Message
    throw
} finally {
    Pop-Location
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
