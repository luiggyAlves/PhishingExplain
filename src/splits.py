# 4.1: treino e validação (90/10) por grupo, estratificados por origem, fonte e rótulo.
# Só entram e-mails com anotação aprovada, fora dos grupos de teste e sem spam. Os ids vão para splits/ (no git).

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RAIZ, SEMENTE
from src.dados import ANOTACOES, emails_pt, teste
from src.teacher.lote import ler

FRACAO_VALIDACAO = 0.1
COLUNAS = ["id", "id_original", "grupo", "fonte", "origem", "rotulo"]


def separar() -> tuple[pd.DataFrame, pd.DataFrame]:
    aprovadas = {r["id"] for r in ler(ANOTACOES)}
    pt, split = emails_pt(), teste()
    pool = pt[pt.id.isin(aprovadas) & pt.rotulo.isin(["phishing", "legitimo"]) & ~pt.grupo.isin(split.grupo)].reset_index(drop=True)
    estrato = pool.origem + "|" + pool.fonte + "|" + pool.rotulo
    dobras = StratifiedGroupKFold(round(1 / FRACAO_VALIDACAO), shuffle=True, random_state=SEMENTE)
    _, val = next(dobras.split(pool, estrato, pool.grupo))
    return pool.drop(index=val), pool.loc[val]


if __name__ == "__main__":
    treino, validacao = separar()
    assert not set(treino.grupo) & set(validacao.grupo), "grupo nos dois conjuntos"
    for nome, d in (("treino", treino), ("validacao", validacao)):
        d[COLUNAS].sort_values("id").to_csv(RAIZ / "splits" / f"{nome}.csv", index=False)
    juntos = pd.concat([treino.assign(conjunto="treino"), validacao.assign(conjunto="validacao")])
    print(pd.crosstab([juntos.origem, juntos.fonte, juntos.rotulo], juntos.conjunto, margins=True, margins_name="total"))
