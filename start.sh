# Performance Optimizations for OpenMP (Threading Affinity)
# This ensures threads stay on the fastest cores and prevents skipping.
export OMP_PROC_BIND=TRUE
export OMP_PLACES=CORES
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "Starting Fish Speech S2 Pro GUI..."

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
