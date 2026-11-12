# 6.2: confere o ambiente da Spark (ou do PC). Cada linha mostra ok ou FALHA.

import platform

from src.config import CFG


def checar() -> bool:
    import torch
    import transformers

    gpu = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if gpu else (0, 0)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16) if gpu else None
    itens = {
        f"perfil {CFG.nome}": True,
        f"arquitetura {platform.machine()}": CFG.nome != "spark" or platform.machine() == "aarch64",
        f"torch {torch.__version__}, CUDA {torch.version.cuda}": gpu and (CFG.nome != "spark" or str(torch.version.cuda).startswith("13")),
        f"GPU {torch.cuda.get_device_name() if gpu else '-'}": gpu,
        f"compute capability {cap[0]}.{cap[1]}": gpu and (CFG.nome != "spark" or cap == (12, 1)),
        "bf16": gpu and torch.cuda.is_bf16_supported(),
        f"transformers {transformers.__version__} (>= 5.2)": tuple(map(int, transformers.__version__.split(".")[:2])) >= (5, 2),
        "multiplicação de matrizes na GPU": gpu and bool(torch.isfinite(a @ a).all()),
    }
    for nome, ok in itens.items():
        print(f"{'ok   ' if ok else 'FALHA'} {nome}")
    return all(itens.values())


if __name__ == "__main__":
    raise SystemExit(0 if checar() else 1)
