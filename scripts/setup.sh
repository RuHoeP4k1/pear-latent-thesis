#!/usr/bin/env bash
# Setup for Linux (lab workstation). Run from the repository root: bash scripts/setup.sh
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python and packages from uv.lock ..."
uv sync

if [ ! -f config/local.toml ]; then
    cp config/local.example.toml config/local.toml
    echo "Created config/local.toml - edit the data paths for this machine."
fi

echo "GPU check:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
else
    echo "nvidia-smi not found."
fi
uv run python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| CUDA build:', torch.version.cuda)"

uv run marimo check notebooks
uv run pytest -q
command -v tmux >/dev/null 2>&1 || echo "Warning: tmux is not installed; long jobs will stop if SSH disconnects."
