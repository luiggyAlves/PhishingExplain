# 4.3: exporta treino, validação e teste em formato de conversa (data/processed/*.jsonl).
# Treino e validação: system = prompt do aluno, user = e-mail, assistant = resposta aprovada do professor.
# Teste: sem assistant; a anotação do professor vai em "referencia", só para a métrica de cobertura.

import json

import pandas as pd

from src.config import DADOS_PROCESSED, RAIZ
from src.dados import ANOTACOES, ANOTACOES_TESTE, emails_pt, teste
from src.prompts_aluno import mensagens
from src.schema import Resposta, formatar_email
from src.teacher.lote import ler


def gravar(nome: str, linhas: list[dict]) -> None:
    DADOS_PROCESSED.mkdir(parents=True, exist_ok=True)
    with (DADOS_PROCESSED / f"{nome}.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas)
    print(f"{nome}.jsonl: {len(linhas)} conversas")


def conversa(e, resposta: dict | None = None) -> list[dict]:
    msgs = mensagens(formatar_email(e.remetente, e.assunto, e.corpo))
    return msgs + ([{"role": "assistant", "content": Resposta.model_validate(resposta).model_dump_json()}] if resposta else [])


def emails_do_teste() -> pd.DataFrame:
    """Os e-mails do teste como o modelo os vê: nativos em português e as traduções das sementes."""
    split = teste()
    t = emails_pt().merge(split[["id", "recorte"]].rename(columns={"id": "id_original"}), on="id_original")
    if faltando := len(split) - len(t):
        print(f"AVISO: {faltando} e-mails do teste ainda sem tradução ficam fora do test.jsonl")
    return t


if __name__ == "__main__":
    pt = emails_pt().set_index("id")
    anot = {r["id"]: r["resposta"] for r in ler(ANOTACOES)}
    for nome, arquivo in (("train", "treino"), ("val", "validacao")):
        ids = pd.read_csv(RAIZ / "splits" / f"{arquivo}.csv").id
        gravar(nome, [{"id": i, "rotulo": pt.loc[i, "rotulo"], "messages": conversa(pt.loc[i], anot[i])} for i in ids])

    ref = {r["id"]: r["resposta"] for r in ler(ANOTACOES_TESTE)}
    gravar("test", [{"id": e.id_original, "id_modelo": e.id, "recorte": e.recorte, "rotulo": e.rotulo, "grupo": e.grupo,
                     "fonte": e.fonte, "messages": conversa(e), "referencia": ref.get(e.id)}
                    for e in emails_do_teste().itertuples()])
