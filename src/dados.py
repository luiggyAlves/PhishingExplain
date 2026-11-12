# Fonte única dos e-mails usados da Fase 2 em diante. Grupo, fonte e rótulo vêm sempre da tabela mestre atual
# (traduções e anotações guardam só o id), então refazer a Fase 1 não deixa nada desatualizado.

import pandas as pd

from src.config import CFG, DADOS_INTERIM, RAIZ, SEMENTE
from src.teacher.lote import ler

MESTRE = DADOS_INTERIM / "emails_agrupados.parquet"
TRADUCOES = DADOS_INTERIM / "traducoes.jsonl"
SINTETICOS = DADOS_INTERIM / "sinteticos_filtrados.parquet"
ANOTACOES = DADOS_INTERIM / "anotacoes.jsonl"            # treino (só aprovadas)
ANOTACOES_TESTE = DADOS_INTERIM / "anotacoes_teste.jsonl"  # referência da avaliação, nunca no treino
TESTE = RAIZ / "splits" / "teste.csv"


def mestre() -> pd.DataFrame:
    return pd.read_parquet(MESTRE)


def teste() -> pd.DataFrame:
    return pd.read_csv(TESTE)


def amostrar(df: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    """No PC, uma amostra de CFG.limite_exemplos equilibrada pelas chaves; na Spark, tudo."""
    if not (n := CFG.limite_exemplos):
        return df
    return df.sample(frac=1, random_state=SEMENTE).groupby(chaves).head(max(1, n // df.groupby(chaves).ngroups))


def emails_pt() -> pd.DataFrame:
    """E-mails em português: nativos da tabela mestre, traduções aprovadas (<id>@pt) e sintéticos filtrados."""
    df = mestre()
    nativos = df[df.idioma == "pt"].assign(id_original=lambda d: d.id, origem="nativo")
    t = pd.DataFrame(ler(TRADUCOES), columns=["id", "id_original", "assunto", "corpo"])
    traduzidos = (df.rename(columns={"id": "id_original"}).drop(columns=["assunto", "corpo"]).merge(t, on="id_original")
                  .assign(idioma="pt", traduzido=True, origem="traduzido"))
    sinteticos = pd.read_parquet(SINTETICOS).assign(id_original=lambda d: d.id, origem="sintetico") if SINTETICOS.exists() else None
    return pd.concat([nativos, traduzidos, sinteticos], ignore_index=True)
