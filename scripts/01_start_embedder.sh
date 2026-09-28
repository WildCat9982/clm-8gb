#!/usr/bin/env bash
# Terminal 1: vLLM embedding server that clm-serve talks to (port 8090).
# Defaults are tuned for an 8 GB GPU. Override with env vars, e.g.:
#   MODEL=Qwen/Qwen3-8B GPU_UTIL=0.9 ./scripts/01_start_embedder.sh   (16 GB+ GPU)
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B-AWQ}"
MAX_LEN="${MAX_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.8}"

echo "Starting vLLM embedder: $MODEL (max_len=$MAX_LEN, gpu_util=$GPU_UTIL) on :8090"
exec vllm serve "$MODEL" \
  --served-model-name qwen3-8b \
  --runner pooling \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enforce-eager \
  --port 8090
