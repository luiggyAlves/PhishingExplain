# Orquestração na DGX Spark, rodando no HOST (o notebook importa este módulo sem dependências: só biblioteca padrão).
# - python(...): roda um passo do projeto dentro da imagem phishexplain-unsloth:0.1.
# - subir(...) / parar(): vLLM num container próprio, um modelo por vez (vLLM e treino disputam a mesma memória).
# Os containers rodam com o usuário do host, então nada no repositório nem no cache do Hugging Face fica com dono root.

import os
import subprocess
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
IMAGEM = "phishexplain-unsloth:0.1"
VLLM_IMAGEM = os.environ.get("PHISH_VLLM_IMAGEM", "nvcr.io/nvidia/vllm:26.09-py3")  # precisa de vLLM >= 0.17
HF = Path(os.environ.get("PHISH_HF", Path.home() / ".cache" / "huggingface"))
NOME = "phishexplain-vllm"
PORTA = 8000
PROFESSOR = ["--reasoning-parser", "qwen3", "--language-model-only", "--max-model-len", "32768",
             "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.8", "--max-num-seqs", "32"]
ALUNO = ["--language-model-only", "--max-model-len", "8192", "--gpu-memory-utilization", "0.6", "--max-num-seqs", "64"]
JUIZ = ["--max-model-len", "16384", "--gpu-memory-utilization", "0.8", "--max-num-seqs", "32",
        "--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"]


def executar(cmd: list[str]) -> None:
    """Roda um comando mostrando a saída linha a linha (no Jupyter, a saída de subprocessos não aparece sozinha)."""
    print("$", " ".join(cmd[cmd.index(IMAGEM) + 1:] if IMAGEM in cmd else cmd), flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for linha in p.stdout:
            print(linha, end="", flush=True)
    if p.returncode:
        raise RuntimeError(f"falhou (código {p.returncode}): {' '.join(cmd)}")


def comum() -> list[str]:
    """Opções de todo container: GPU, rede do host, usuário do host, cache do Hugging Face e token."""
    HF.mkdir(parents=True, exist_ok=True)
    return ["--rm", "--gpus", "all", "--ipc=host", "--net=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "USER=phish",  # sem USER, o getpass do torch quebra (uid sem /etc/passwd)
            "-v", f"{HF}:/hf", "-e", "HF_HOME=/hf",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}"]


def python(*args: str, **env) -> None:
    """`python <args>` dentro da imagem do projeto, com o repositório montado e o perfil spark."""
    (RAIZ / ".home").mkdir(exist_ok=True)  # HOME persistente dos containers (caches do triton e da Unsloth)
    variaveis = {"PHISH_PERFIL": "spark", "HOME": "/workspace/phishexplain/.home", "TQDM_MININTERVAL": "60"} | env
    executar(["docker", "run", *comum(), "-v", f"{RAIZ}:/workspace/phishexplain", "-w", "/workspace/phishexplain",
              *[x for k, v in variaveis.items() for x in ("-e", f"{k}={v}")], IMAGEM, "python", *args])


def modulo(nome: str, *args: str, **env) -> None:
    python("-m", nome, *args, **env)


def parar() -> None:
    if NOME in subprocess.run(["docker", "rm", "-f", NOME], capture_output=True, text=True).stdout:  # vazio se não existia
        print("vLLM parado")


def subir(modelo: str, flags: list[str], lora: dict[str, str] | None = None, nome: str | None = None, espera: int = 7200) -> str:
    """Sobe o vLLM servindo `modelo` com o nome `nome` (padrão: o próprio modelo); `lora` = {nome: pasta relativa à raiz,
    como "outputs/qwen35-4b"}. Espera ficar pronto (a primeira vez inclui o download) e devolve a URL da API."""
    parar()
    extra = ["--enable-lora", "--max-lora-rank", "16", "--lora-modules", *[f"{n}=/repo/{p}" for n, p in lora.items()]] if lora else []
    r = subprocess.run(["docker", "run", "-d", "--name", NOME, *comum()[1:], "-e", "HOME=/tmp", "-v", f"{RAIZ}:/repo:ro",
                        "--entrypoint", "vllm", VLLM_IMAGEM, "serve", modelo, "--served-model-name", nome or modelo,
                        "--port", str(PORTA), *flags, *extra], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    inicio = time.time()
    while time.time() - inicio < espera:
        if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", NOME], capture_output=True, text=True).stdout.strip() != "true":
            raise RuntimeError("vLLM caiu:\n" + logs())
        try:
            urllib.request.urlopen(f"http://localhost:{PORTA}/health", timeout=5)
            print(f"vLLM no ar com {modelo} em {time.time() - inicio:.0f} s")
            return f"http://localhost:{PORTA}/v1"
        except OSError:
            time.sleep(15)
    raise TimeoutError("vLLM não respondeu a tempo:\n" + logs())


def logs(n: int = 60) -> str:
    r = subprocess.run(["docker", "logs", "--tail", str(n), NOME], capture_output=True, text=True)
    return r.stdout + r.stderr
