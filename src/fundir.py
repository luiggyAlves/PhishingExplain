# Funde o adaptador LoRA de um aluno ao modelo base e grava um modelo completo em bf16 (outputs/<aluno>/fundido).
# Plano B do notebook, se o vLLM não aceitar o adaptador do aluno multimodal, e ponto de partida do GGUF (Fase 9).
# Os arquivos que não são pesos (config, tokenizador, processador, template) vêm do repositório base, então o vLLM
# serve o fundido do mesmo jeito que serve o base.
# Uso: python -m src.fundir --aluno qwen35-4b

import argparse
import shutil
from pathlib import Path

from src.config import CFG, SAIDAS

PESOS = ["*.safetensors", "*.index.json", "*.bin", "*.pth", "*.pt", "*.gguf", "original/*", "consolidated*"]


def fundir(aluno: str) -> Path:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    destino = SAIDAS / aluno / "fundido"
    if (destino / "model.safetensors.index.json").exists() or (destino / "model.safetensors").exists():
        print(destino, "já existe")
        return destino
    repo = CFG.alunos[aluno]
    base = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16)
    modelo = PeftModel.from_pretrained(base, SAIDAS / aluno).merge_and_unload()
    shutil.copytree(snapshot_download(repo, ignore_patterns=PESOS), destino, dirs_exist_ok=True)
    modelo.save_pretrained(destino)
    print(f"{repo} + adaptador -> {destino}")
    return destino


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aluno", required=True, choices=list(CFG.alunos))
    fundir(ap.parse_args().aluno)
