#!/usr/bin/env bash
# Terminal 2: CLM server (port 8700). Waits until the embedder is ready first.
set -euo pipefail
echo "Waiting for the vLLM embedder on :8090 ..."
until curl -sf http://127.0.0.1:8090/v1/models > /dev/null; do sleep 2; done
echo "Embedder is up. Starting clm-serve on :8700"
exec clm-serve
