# Anotação pelo professor: para cada e-mail em português, pede indicadores, classificação e explicação,
# valida (2.3) e, se falhar, devolve os erros ao professor e pede de novo.
# Uso: python -m src.teacher.anotar           (treino: nativos, traduções e sintéticos fora dos grupos de teste)
#      python -m src.teacher.anotar --teste   (os 399 do teste, só como referência da avaliação, em arquivo separado)

import argparse
import json

import pandas as pd

from src.config import CFG, DADOS_INTERIM, SEMENTE
from src.dados import emails_pt, teste
from src.schema import Resposta, formatar_email
from src.teacher.cliente import perguntar
from src.teacher.lote import ler, rodar
from src.teacher.prompts import mensagens
from src.teacher.traduzir import como_itens
from src.teacher.validar import validar

TENTATIVAS = 3
ARQUIVOS = {do_teste: (DADOS_INTERIM / f"anotacoes{s}.jsonl", DADOS_INTERIM / f"anotacoes{s}_reprovadas.jsonl")
            for do_teste, s in ((False, ""), (True, "_teste"))}


def anotar(e: dict) -> dict:
    email = formatar_email(e["remetente"], e["assunto"], e["corpo"])
    conversa, segundos = mensagens(email, e["rotulo"]), 0.0
    for tentativa in range(1, TENTATIVAS + 1):
        r = perguntar(conversa, esquema=Resposta)
        segundos += r.segundos
        erros = ["resposta cortada"] * r.cortada + validar(r.texto, email, e["rotulo"])
        if not erros:
            break
        conversa = conversa + [
            {"role": "assistant", "content": r.texto},
            {"role": "user", "content": "A resposta tem estes problemas:\n- " + "\n- ".join(erros)
             + "\nResponda de novo com o JSON corrigido. Os trechos devem ser cópias exatas do e-mail."},
        ]
    try:
        resposta = json.loads(r.texto)
    except ValueError:
        resposta = r.texto
    return {"rotulo": e["rotulo"], "resposta": resposta, "erros": erros, "tentativas": tentativa,
            "segundos": round(segundos, 1)}


def selecionar(do_teste: bool) -> pd.DataFrame:
    pt, split = emails_pt(), teste()
    if do_teste:  # nativos do teste + traduções das sementes do teste
        return pt[pt.id_original.isin(split.id)]
    livres = pt[pt.rotulo.isin(["phishing", "legitimo"]) & ~pt.grupo.isin(split.grupo)]
    if not (n := CFG.limite_exemplos):
        return livres
    estratos = livres.groupby(["origem", "rotulo"])  # no PC: amostra equilibrada por origem e rótulo
    return livres.sample(frac=1, random_state=SEMENTE).groupby(["origem", "rotulo"]).head(max(1, n // estratos.ngroups))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", action="store_true", help="anota o teste, só como referência para a avaliação")
    do_teste = ap.parse_args().teste
    aprovadas, reprovadas = ARQUIVOS[do_teste]
    emails = selecionar(do_teste)[["id", "rotulo", "remetente", "assunto", "corpo"]]
    rodar(como_itens(emails), anotar, aprovadas, reprovadas)

    ok, ruins = ler(aprovadas), ler(reprovadas)
    print(f"\nAprovadas: {len(ok)} | reprovadas: {len(ruins)} | taxa de aprovação: {len(ok) / max(1, len(ok) + len(ruins)):.0%}")
    print("Tentativas das aprovadas:", pd.Series([r["tentativas"] for r in ok]).value_counts().sort_index().to_dict())
    print(f"Tempo médio por e-mail: {pd.Series([r['segundos'] for r in ok + ruins]).mean():.1f} s")
    for r in ruins:
        print(f"  {r['id']}: {r['erros']}")
