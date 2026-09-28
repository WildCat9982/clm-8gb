#!/usr/bin/env bash
# Terminal 3: confirm both services are up before running experiments.
check() {
  if curl -s -o /dev/null --max-time 3 "$1"; then echo "  OK    $2"; else echo "  DOWN  $2"; fi
}
echo "Service check:"
check http://127.0.0.1:8090/v1/models "vLLM embedder  (:8090)"
check http://127.0.0.1:8700/          "clm-serve      (:8700)"

echo
echo "Embedding dimension (CLM expects 4096):"
curl -s http://127.0.0.1:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b", "input": "hello"}' \
  | python -c "import sys, json; print('  ', len(json.load(sys.stdin)['data'][0]['embedding']))" \
  || echo "   could not get an embedding"
