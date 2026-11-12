# Imagem do projeto na DGX Spark (6.1): a imagem da Unsloth (docker/Dockerfile.unsloth, cópia do Dockerfile_DGX_Spark
# oficial) mais as dependências do projeto. O notebook roda no host e usa esta imagem para cada passo (src/servidor.py).
# O torch da NGC fica travado: se alguma dependência pedir outro torch, o pip falha em vez de trocá-lo.
# Construa com ./scripts/spark_construir.sh
FROM unsloth-dgx-spark

COPY requirements.txt requirements-treino.txt /tmp/
RUN pip list --format=freeze | grep -iE '^(torch|torchvision)==' > /tmp/ngc_torch.txt && \
    pip install --no-cache-dir -c /tmp/ngc_torch.txt -r /tmp/requirements.txt -r /tmp/requirements-treino.txt \
        matplotlib && \
    rm /tmp/ngc_torch.txt

WORKDIR /workspace/phishexplain
CMD ["/bin/bash"]
