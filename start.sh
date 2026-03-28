#!/bin/bash

# Fish Speech S2 Pro - Linux / WSL Start Script
# Usage: bash start.sh

echo "Starting Fish Speech S2 Pro GUI..."

# Detect Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[ERROR] Virtual environment (.venv) not found. Please run bash install.sh first."
    exit 1
fi

# Run the app
uv run app.py
