#!/usr/bin/env bash
# Sobe o professor de desenvolvimento no PC.
# Só aceita conexões desta máquina (127.0.0.1), na porta 8080.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELO=$(ls models/qwen35-4b/*.gguf | head -n 1)
echo "Modelo: $MODELO"

exec llama-server \
  --model "$MODELO" \
  --alias professor-dev \
  --host 127.0.0.1 \
  --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 16384 \
  --parallel 2 \
  --jinja
