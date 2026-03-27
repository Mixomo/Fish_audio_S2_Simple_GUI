# 🎙️ Fish Speech S2 Pro - Voice Clone & Training GUI

A comprehensive, all-in-one Graphical User Interface (GUI) for **Fish Speech S2 Pro**. This project streamlines the process of voice cloning, dataset preparation, and LoRA training, providing a robust and optimized experience on Windows with full GPU acceleration.

## Key Features

### 🔊 High-Performance Voice Cloning
*   **Dual-Engine Support**: Choose between **C++ (s2.cpp) Backend** (optimized for GGUF models) or **PyTorch Backend**.
*   **GGUF Quantization Support**: Native support for **F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K, and Q2_K** models.
*   **Automated GPU Orchestration**: 
    *   **CUDA Core Engine**: Used for high-precision models (F16, Q8_0) on supported NVIDIA GPUs.
    *   **Vulkan Engine**: Leveraged for K-quantized models (`Q_K` types) to ensure stability and compatibility.
    *   **CPU Fallback**: Automatic fallback for lower quantization levels or systems without dedicated GPUs.
*   **Intelligent Text Processing**: Features automatic paragraph splitting for long texts to ensure smooth and high-quality synthesis.

### 🛠️ Advanced Dataset Preparation
*   **Single Editor**: Drag-and-drop interface for individual audio editing. Trim, normalize, and transcribe audio on the fly.
*   **Batch Processor**: Automatically process entire folders of audio. 
    *   Normalizes volume and converts to mono.
    *   Uses **Faster-Whisper** for rapid, accurate batched transcriptions (creates `.lab` files).
    *   Generates `metadata.csv` required for training automatically.

### 🏋️ LoRA Training Pipeline
*   **Unified Workflow**: A simplified, 4-step pipeline that handles dataset preparation, VQ code extraction, sharding, and actual LoRA training.
*   **VRAM Optimization**: Hardware presets for **24GB** and **32GB+ VRAM** to auto-tune batch sizes and gradient accumulation.
*   **Auto-Tune Max Steps**: Automatically calculates the optimal number of training steps based on your dataset size.
*   **One-Click Export**: Automatically exports trained LoRA weights for immediate use in the inference tab.

### 🌟 One click install
*   **Automated Infrastructure**: The installer automatically detects and installs missing dependencies like **Visual Studio 2022 C++ Tools, CUDA Toolkit, and Vulkan SDK** via `winget`.
*   **GPU Architecture aware**: Auto-detects GPU architecture and installs the appropriate version of CUDA. Ampere (RTX 30 series), Ada (RTX 40 series) & Blackwell (RTX 50 series) Support. 


## 🚀 Quick Start (Windows)

### 1. Installation
Simply run the batch installer:
```cmd
install.bat
```
The script will set up a virtual environment through uv and install all necessary Python libraries, CUDA - PyTorch specific version for your GPU architecture, and compile s2.cpp for CUDA, Vulkan & CPU.

### 2. Launch
Start the application:
```cmd
start.bat
```
Inspired by [FranckyB](https://github.com/FranckyB) [Voice Clone Studio](https://github.com/FranckyB/Voice-Clone-Studio)

Based on [Fish Speech S2 PRO](https://huggingface.co/fishaudio/s2-pro) by [Fish Audio](https://github.com/fishaudio)