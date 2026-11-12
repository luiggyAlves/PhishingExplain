# 5.2: baseline clássico, só classificação. TF-IDF de palavras e caracteres + regressão logística, treinado no
# train.jsonl e aplicado ao test.jsonl. Grava as previsões no mesmo formato dos LLMs para o avaliador.

import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from src.config import DADOS_PROCESSED, SAIDAS

DESTINO = SAIDAS / "previsoes" / "baseline_tfidf.jsonl"


def textos(nome: str) -> pd.DataFrame:
    df = pd.read_json(DADOS_PROCESSED / f"{nome}.jsonl", lines=True)
    return df.assign(texto=df.messages.map(lambda m: m[1]["content"]))


if __name__ == "__main__":
    treino, teste = textos("train"), textos("test")
    modelo = make_pipeline(
        make_union(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                   TfidfVectorizer(sublinear_tf=True, min_df=2, analyzer="char_wb", ngram_range=(3, 5))),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(treino.texto, treino.rotulo)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    with DESTINO.open("w", encoding="utf-8") as f:
        for i, c in zip(teste.id, modelo.predict(teste.texto)):
            f.write(json.dumps({"id": i, "texto": json.dumps({"classificacao": c})}) + "\n")
    print(f"{len(teste)} previsões em {DESTINO.relative_to(SAIDAS.parent)} (treino com {len(treino)} e-mails)")
