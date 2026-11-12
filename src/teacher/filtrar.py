# Filtros dos sintéticos (3.3): descarta quase-duplicatas (contra toda a tabela mestre, teste incluído, e entre si),
# grava os aprovados, repete a tabela de marcadores e monta a amostra para revisão manual.
# Rode depois de gerar.py e de novo depois da anotação, para a revisão já trazer as respostas do professor.

import json

import pandas as pd
from datasketch import MinHashLSH

from src.config import DADOS_INTERIM, SEMENTE
from src.dados import ANOTACOES, SINTETICOS, emails_pt, mestre
from src.data.agrupar import LIMIAR, NUM_PERM, assinatura
from src.teacher.gerar import GERADOS
from src.teacher.lote import ler

REVISAO = DADOS_INTERIM / "revisao_sinteticos.md"
N_REVISAO = 50


def texto(df: pd.DataFrame) -> pd.Series:
    return df.assunto.fillna("") + "\n" + df.corpo


def filtrar() -> pd.DataFrame:
    sint = pd.DataFrame(ler(GERADOS)).drop(columns=["erros", "segundos", "descricao"])
    df = mestre()
    lsh = MinHashLSH(threshold=LIMIAR, num_perm=NUM_PERM)
    for i, t in enumerate(texto(df)):
        lsh.insert(f"m{i}", assinatura(t))
    manter = []
    for i, t in enumerate(texto(sint)):
        m = assinatura(t)
        manter.append(not lsh.query(m))  # nada parecido na tabela mestre nem nos sintéticos já aceitos
        if manter[-1]:
            lsh.insert(f"s{i}", m)
    print(f"Sintéticos: {len(sint)} gerados, {len(sint) - sum(manter)} descartados por quase-duplicata")
    return sint[manter].assign(fonte="sintetico", grupo=lambda d: d.id.str.replace("sintetico:", "sint:"),
                               autoria="llm", idioma="pt", traduzido=False)


def revisao(sint: pd.DataFrame) -> str:
    anot = {r["id"]: r["resposta"] for r in ler(ANOTACOES)}
    blocos = [
        f"## {e.id}: {e.rotulo} / {e.cenario}{' (parecido com golpe)' if e.parecido else ''}\n\n"
        f"Eixos: {e.tipo_remetente}; {e.tom}; {e.qualidade}; {e.tamanho}; {e.link_anexo}; marcadores: {e.marcadores}\n\n"
        f"```text\nDe: {e.remetente}\nAssunto: {e.assunto}\n\n{e.corpo}\n```\n\n"
        + (f"```json\n{json.dumps(anot[e.id], ensure_ascii=False, indent=1)}\n```\n" if e.id in anot else "_(ainda sem anotação)_\n")
        for e in sint.sample(min(N_REVISAO, len(sint)), random_state=SEMENTE).itertuples()
    ]
    return "# Revisão manual dos sintéticos\n\nPara cada um: o e-mail é realista? O rótulo está certo? A anotação cita bem?\n\n" + "\n".join(blocos)


if __name__ == "__main__":
    sint = filtrar()
    sint.to_parquet(SINTETICOS, index=False)
    print(sint.groupby(["rotulo", "parecido"]).size().rename("sintéticos aprovados").to_string())

    pt = emails_pt()
    marcado = pt.corpo.str.contains(r"\[EMAIL\]|\[LINK\]")
    print("\nE-mails em português com [EMAIL] ou [LINK], por fonte e rótulo (proporção):")
    print(marcado.groupby([pt.fonte, pt.rotulo]).agg(["sum", "mean"]).round(2).to_string())

    REVISAO.write_text(revisao(sint), encoding="utf-8")
    print(f"\nAmostra para revisão em {REVISAO}")
