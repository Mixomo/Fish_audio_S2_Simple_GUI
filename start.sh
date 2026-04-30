#!/bin/bash

# Performance Optimizations for OpenMP (Threading Affinity)
# This ensures threads stay on the fastest cores and prevents skipping.
export OMP_PROC_BIND=TRUE
export OMP_PLACES=CORES
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

# Fish Speech S2 Pro - Linux / WSL Start Script
# Usage: bash start.sh

echo "Starting Fish Speech S2 Pro GUI..."
cd "$(dirname "$0")" || exit 1

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' was not found. Run bash install.sh first, then open a new terminal."
    exit 1
fi

# Detect Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[ERROR] Virtual environment (.venv) not found. Please run bash install.sh first."
    exit 1
fi

# Run the app
uv run app.py
