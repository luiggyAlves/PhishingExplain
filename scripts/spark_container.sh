#!/usr/bin/env bash
# 6.2: sobe o container do projeto na Spark com o Jupyter na porta 8888 (acesse por túnel SSH:
#   ssh -L 8888:localhost:8888 <usuario>@<spark>  e abra http://localhost:8888).
# Use "./scripts/spark_container.sh bash" para um terminal em vez do Jupyter.
# O socket do Docker é montado para o notebook subir e parar o vLLM num container irmão; isso dá ao
# container controle total do Docker da Spark, então só use esta imagem com código seu.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ~/.cache/huggingface outputs reports

COMANDO=${1:-jupyter}
[ "$COMANDO" = "jupyter" ] && COMANDO="jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --ServerApp.token=${JUPYTER_TOKEN:-phishexplain}"

docker run -it --rm --gpus all --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/phishexplain \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e PHISH_PERFIL=spark \
  -e PHISH_HOST_REPO="$PWD" \
  -e PHISH_HOST_HF="$HOME/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -w /workspace/phishexplain --name phishexplain \
  phishexplain-unsloth:0.1 $COMANDO
