# 5.1: avaliador. Lê previsões em JSONL ({"id", "texto", "segundos", "tokens_saida"}) e calcula, por recorte do teste
# e no total: precisão, recall, F1 e taxa de falso positivo (com IC 95% por bootstrap), JSON válido no contrato,
# ancoragem dos trechos, cobertura de tipos contra a anotação de referência, tokens e tempo.
# Previsão sem classificação legível conta como erro (o pior caso), para não premiar quem não segue o formato.
# Uso: python -m src.evaluate outputs/previsoes/<nome>.jsonl [...]

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import DADOS_PROCESSED, RELATORIOS, SEMENTE
from src.schema import Resposta
from src.teacher.validar import normal

N_BOOTSTRAP = 1000
CLASSE = re.compile(r'"classificacao"\s*:\s*"(phishing|legitimo)"')


def limpar(texto: str) -> str:
    """Tira raciocínio (<think>...</think>) e cercas de código que alguns modelos põem em volta do JSON."""
    texto = re.sub(r"<think>.*?</think>", "", texto or "", flags=re.S).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", texto).strip()


def ler_previsao(p: dict, email: str) -> dict:
    texto = limpar(p.get("texto", ""))
    try:
        dados = json.loads(texto)
    except ValueError:
        dados = {}
    try:
        Resposta.model_validate_json(texto)
        valido = True
    except ValueError:
        valido = False
    trechos = [i.get("trecho", "") for i in dados.get("indicadores", []) if isinstance(i, dict)] if isinstance(dados, dict) else []
    tipos = {i.get("tipo") for i in dados.get("indicadores", []) if isinstance(i, dict)} if isinstance(dados, dict) else set()
    achou = CLASSE.search(texto)
    return {"previsto": achou.group(1) if achou else None, "json_valido": valido, "tipos": tipos,
            "ancorados": sum(normal(t) in normal(email) for t in trechos if isinstance(t, str)), "trechos": len(trechos),
            "segundos": p.get("segundos", np.nan), "tokens_saida": p.get("tokens_saida", np.nan)}


def f1_tipos(a: set, b: set) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def classificacao(df: pd.DataFrame) -> dict:
    real = (df.rotulo == "phishing").to_numpy()
    prev = np.where(df.previsto.isna(), ~real, df.previsto == "phishing")  # sem classificação = erro
    vp, fp, fn, vn = (real & prev).sum(), (~real & prev).sum(), (real & ~prev).sum(), (~real & ~prev).sum()
    precisao = vp / (vp + fp) if vp + fp else np.nan
    recall = vp / (vp + fn) if vp + fn else np.nan
    return {"precisao": precisao, "recall": recall, "f1": 2 * vp / (2 * vp + fp + fn) if vp else 0.0,
            "taxa_falso_positivo": fp / (fp + vn) if fp + vn else np.nan, "acuracia": (vp + vn) / len(df)}


def com_ic(df: pd.DataFrame) -> dict:
    """Métricas de classificação com intervalo de 95% por bootstrap (reamostragem dos e-mails)."""
    rng = np.random.default_rng(SEMENTE)
    base = classificacao(df)
    amostras = pd.DataFrame([classificacao(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(N_BOOTSTRAP)])
    return {k: v for m, v0 in base.items()
            for k, v in ((m, v0), (f"{m}_ic", f"{amostras[m].quantile(0.025):.3f}–{amostras[m].quantile(0.975):.3f}"))}


def resumo(df: pd.DataFrame) -> dict:
    cobertura = [f1_tipos(r.tipos, r.tipos_ref) for r in df.itertuples() if r.tipos_ref is not None and r.tipos]
    return {"n": len(df)} | com_ic(df) | {
        "json_valido": df.json_valido.mean(),
        "ancoragem": df.ancorados.sum() / df.trechos.sum() if df.trechos.sum() else np.nan,
        "cobertura_tipos": np.mean(cobertura) if cobertura else np.nan,
        "tokens_saida": df.tokens_saida.mean(), "segundos": df.segundos.mean()}


def carregar(arquivo: Path) -> pd.DataFrame:
    teste = pd.read_json(DADOS_PROCESSED / "test.jsonl", lines=True).set_index("id")
    prev = {p["id"]: p for p in map(json.loads, Path(arquivo).open(encoding="utf-8"))}
    linhas = []
    for i, t in teste.iterrows():
        email = t.messages[1]["content"]
        ref = t.referencia if isinstance(t.referencia, dict) else None
        linhas.append({"id": i, "recorte": t.recorte, "rotulo": t.rotulo, "fonte": t.fonte, "grupo": t.grupo,
                       "tipos_ref": {x["tipo"] for x in ref["indicadores"]} if ref else None}
                      | ler_previsao(prev.get(i, {}), email))
    faltando = len(teste) - len(set(prev) & set(teste.index))
    if faltando:
        print(f"AVISO: {Path(arquivo).name}: {faltando} e-mails do teste sem previsão (contam como erro)")
    return pd.DataFrame(linhas)


def avaliar(arquivo: Path) -> pd.DataFrame:
    """Uma linha por recorte (pt_nativo, traduzido, total)."""
    df = carregar(arquivo)
    partes = [(r, df[df.recorte == r]) for r in sorted(df.recorte.unique())] + [("total", df)]
    return pd.DataFrame([{"modelo": Path(arquivo).stem, "recorte": r} | resumo(d) for r, d in partes])


def campanhas_rnp(arquivo: Path) -> pd.DataFrame:
    """Recall por campanha (grupo) nos golpes reais do RNP."""
    df = carregar(arquivo).query("fonte == 'pt_rnp'")
    return (df.assign(acerto=df.previsto == "phishing").groupby("grupo").acerto.agg(["size", "mean"])
            .rename(columns={"size": "emails", "mean": "recall"}).sort_values("emails", ascending=False))


def comparar(a: Path, b: Path, recorte: str | None = None) -> dict:
    """Teste de McNemar exato (pareado nos mesmos e-mails): a diferença de acertos entre dois modelos é significativa?"""
    da, db = (carregar(x).set_index("id") for x in (a, b))
    if recorte:
        da, db = da[da.recorte == recorte], db[db.recorte == recorte]
    certo = lambda d: d.previsto == d.rotulo
    so_a, so_b = int((certo(da) & ~certo(db)).sum()), int((~certo(da) & certo(db)).sum())
    p = binomtest(so_a, so_a + so_b).pvalue if so_a + so_b else 1.0
    return {"a": Path(a).stem, "b": Path(b).stem, "recorte": recorte or "total", "acuracia_a": certo(da).mean(),
            "acuracia_b": certo(db).mean(), "so_a_acerta": so_a, "so_b_acerta": so_b, "p_mcnemar": p}


if __name__ == "__main__":
    tabela = pd.concat([avaliar(Path(a)) for a in sys.argv[1:]], ignore_index=True)
    RELATORIOS.mkdir(exist_ok=True)
    tabela.to_csv(RELATORIOS / "metricas.csv", index=False)
    pd.set_option("display.width", 200)
    print(tabela[["modelo", "recorte", "n", "precisao", "recall", "f1", "f1_ic", "taxa_falso_positivo", "json_valido",
                  "ancoragem", "cobertura_tipos", "segundos"]].round(3).to_string(index=False))
