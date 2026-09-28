#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu (or native Linux).
set -euo pipefail

echo "== Checking GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || {
  echo "nvidia-smi failed. Update your Windows NVIDIA driver (do NOT install a Linux driver in WSL)."; exit 1; }

echo "== System packages =="
sudo apt update && sudo apt install -y python3-venv python3-pip curl

echo "== Python environment (~/vllm-env) =="
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install -U pip
pip install -U vllm
pip install -r "$(dirname "$0")/../requirements.txt"

echo
echo "Done. Next: install CLM (README, Step 2), then run ./scripts/01_start_embedder.sh"
