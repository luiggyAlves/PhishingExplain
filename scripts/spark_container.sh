#!/usr/bin/env bash
# Terminal dentro da imagem do projeto na Spark, para depurar à mão (o notebook não precisa disto).
# Mesmas opções que src/servidor.py usa: usuário do host, cache do Hugging Face em ~/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ~/.cache/huggingface .home
docker run -it --rm --gpus all --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e USER=phish -e HOME=/workspace/phishexplain/.home \
  -v "$PWD":/workspace/phishexplain -w /workspace/phishexplain \
  -v ~/.cache/huggingface:/hf -e HF_HOME=/hf -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PHISH_PERFIL=spark \
  phishexplain-unsloth:0.1 bash
