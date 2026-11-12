#!/usr/bin/env bash
# 6.1: constrói as imagens na DGX Spark. Rode na raiz do repositório, na Spark (aarch64).
# 1) unsloth-dgx-spark: o Dockerfile oficial da Unsloth (compila triton e xformers para o GB10; demora).
# 2) phishexplain-unsloth:0.1: a imagem acima + dependências do projeto + Jupyter.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(uname -m)" = "aarch64" ] || { echo "Este script é para a DGX Spark (aarch64)."; exit 1; }
docker image inspect unsloth-dgx-spark >/dev/null 2>&1 || \
  docker build -t unsloth-dgx-spark -f docker/Dockerfile.unsloth docker/
docker build -t phishexplain-unsloth:0.1 -f docker/Dockerfile .
echo "Pronto. Suba o container com ./scripts/spark_container.sh"
