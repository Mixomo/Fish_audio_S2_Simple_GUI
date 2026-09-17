@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Fish Audio S2 Easy GUI

set "UV_EXE=%CD%\.runtime\uv\uv.exe"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "VULKAN_SDK=%CD%\.runtime\vulkan"
set "VK_SDK_PATH=%VULKAN_SDK%"
set "UV_PYTHON_INSTALL_DIR=%CD%\.runtime\python"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"
set "UV_NO_CACHE=1"
set "UV_LINK_MODE=copy"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CUDA_MODULE_LOADING=LAZY"
set "HF_HOME=%CD%\.runtime\cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "MODELSCOPE_CACHE=%CD%\.runtime\cache\modelscope"
set "GRADIO_TEMP_DIR=%CD%\.runtime\gradio-temp"
set "TRITON_CACHE_DIR=%CD%\.runtime\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.runtime\torchinductor-cache"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"
set "PATH=%VULKAN_SDK%\Bin;%CD%\.venv\Lib\site-packages\torch\lib;%PATH%"

if not exist "%PY_EXE%" (
  echo [ERROR] Local environment not found. Run install.bat first.
  pause
  exit /b 1
)

echo.
echo [INFO] PyTorch first inference after each GUI launch may take a few minutes.
echo [INFO] This is normal: PyTorch/Triton compiles optimized GPU kernels on first use.
echo [INFO] Subsequent PyTorch generations while that runtime remains loaded will be much faster.
echo [INFO] Unloading models or switching away from PyTorch deliberately frees model/compile state,
echo [INFO] so returning to PyTorch can require the warm-up again.
echo.

"%PY_EXE%" app.py
if errorlevel 1 pause
