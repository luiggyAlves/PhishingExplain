# Roda uma tarefa do professor sobre muitos e-mails em paralelo, gravando cada resultado assim que fica pronto.
# Se a execução cair, rodar de novo pula os ids que já estão em algum dos dois arquivos.

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config import CFG


def ler(arquivo: Path) -> list[dict]:
    return [json.loads(l) for l in arquivo.open(encoding="utf-8")] if arquivo.exists() else []


def rodar(itens: list[dict], tarefa, aprovados: Path, reprovados: Path) -> None:
    """`tarefa(item)` devolve um dict; se ele tiver "erros" não vazios, vai para `reprovados`."""
    feitos = {r["id"] for arq in (aprovados, reprovados) for r in ler(arq)}
    pendentes = [i for i in itens if i["id"] not in feitos]
    print(f"{len(itens)} itens, {len(itens) - len(pendentes)} já feitos, {len(pendentes)} pendentes")
    with ThreadPoolExecutor(CFG.professor_paralelo) as ex, \
         aprovados.open("a", encoding="utf-8") as ok, reprovados.open("a", encoding="utf-8") as ruim:
        futuros = {ex.submit(tarefa, i): i["id"] for i in pendentes}
        for f in tqdm(as_completed(futuros), total=len(futuros)):
            try:
                r = {"id": futuros[f]} | f.result()
            except Exception as erro:  # falha de rede ou do servidor: não grava, fica para a próxima execução
                print(f"{futuros[f]}: {erro!r}")
                continue
            destino = ruim if r.get("erros") else ok
            destino.write(json.dumps(r, ensure_ascii=False) + "\n")
            destino.flush()
