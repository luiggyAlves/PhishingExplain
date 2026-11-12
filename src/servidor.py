# 6.4: sobe e para o vLLM num container irmão (pelo socket do Docker), com API compatível com OpenAI.
# Usado pelo notebook da Spark para servir o professor, os alunos base e os alunos com adaptador LoRA.
# Só um modelo por vez: o vLLM e o treino disputam a mesma memória unificada.

import os
import time
import urllib.request


VLLM_IMAGEM = os.environ.get("PHISH_VLLM_IMAGEM", "nvcr.io/nvidia/vllm:26.09-py3")  # precisa de vLLM >= 0.17
NOME = "phishexplain-vllm"
PORTA = 8000
PROFESSOR = ["--reasoning-parser", "qwen3", "--language-model-only", "--max-model-len", "32768",
             "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.8", "--max-num-seqs", "32"]
ALUNO = ["--language-model-only", "--max-model-len", "8192", "--gpu-memory-utilization", "0.6", "--max-num-seqs", "64"]


def parar() -> None:
    import docker

    try:
        docker.from_env().containers.get(NOME).remove(force=True)
        print("vLLM parado")
    except docker.errors.NotFound:
        pass


def subir(modelo: str, flags: list[str], lora: dict[str, str] | None = None, nome: str | None = None, espera: int = 3600) -> str:
    """Sobe o vLLM servindo `modelo` com o nome `nome` (padrão: o próprio modelo; no plano B do professor, o nome de
    CFG.professor_modelo); `lora` = {nome: pasta relativa à raiz, como "outputs/qwen35-4b"}. Devolve a URL da API."""
    import docker

    parar()
    host_repo, host_hf = os.environ["PHISH_HOST_REPO"], os.environ["PHISH_HOST_HF"]
    extra = ["--enable-lora", "--max-lora-rank", "16", "--lora-modules",
             *[f"{n}=/repo/{p}" for n, p in lora.items()]] if lora else []
    c = docker.from_env().containers.run(
        VLLM_IMAGEM, ["vllm", "serve", modelo, "--served-model-name", nome or modelo, "--port", str(PORTA), *flags, *extra],
        name=NOME, detach=True, network_mode="host", ipc_mode="host", shm_size="16g",
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        volumes={host_hf: {"bind": "/root/.cache/huggingface", "mode": "rw"}, host_repo: {"bind": "/repo", "mode": "ro"}},
        environment={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    inicio = time.time()
    while time.time() - inicio < espera:
        c.reload()
        if c.status == "exited":
            raise RuntimeError("vLLM caiu:\n" + c.logs(tail=60).decode())
        try:
            urllib.request.urlopen(f"http://localhost:{PORTA}/health", timeout=5)
            print(f"vLLM no ar com {modelo} em {time.time() - inicio:.0f} s")
            return f"http://localhost:{PORTA}/v1"
        except OSError:
            time.sleep(10)
    raise TimeoutError("vLLM não respondeu a tempo:\n" + c.logs(tail=60).decode())
