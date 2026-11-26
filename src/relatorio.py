# 8.1 e 8.3: junta tudo em reports/resultados.md: métricas por recorte de todos os modelos, testes de McNemar
# das perguntas de pesquisa, recall por campanha no RNP, notas do juiz, erros por fonte e tabela de marcadores.
# Convenção dos arquivos em outputs/previsoes/: <aluno>_zeroshot, <aluno>_lora, professor, baseline_tfidf.

from pathlib import Path

import pandas as pd

from src.config import RAIZ, RELATORIOS, SAIDAS
from src.dados import emails_pt
from src.evaluate import avaliar, campanhas_rnp, carregar, comparar
from src.juiz import tabela as notas_juiz

PREVISOES = SAIDAS / "previsoes"
COLUNAS = ["modelo", "recorte", "n", "precisao", "recall", "f1", "f1_ic", "taxa_falso_positivo", "taxa_falso_positivo_ic",
           "json_valido", "ancoragem", "cobertura_tipos", "tokens_saida", "segundos"]


def arquivos() -> dict[str, Path]:
    return {a.stem: a for a in sorted(PREVISOES.glob("*.jsonl")) if not a.stem.endswith("_falhas")}


def perguntas(arq: dict) -> pd.DataFrame:
    """Pares das perguntas de pesquisa: S5-nano x Qwen3.5-4B, Gemma x Qwen, e LoRA x zero-shot de cada aluno."""
    pares = [(f"s5-nano_{m}", f"qwen35-4b_{m}") for m in ("zeroshot", "lora")]
    pares += [(f"gemma4-e2b_{m}", f"qwen35-4b_{m}") for m in ("zeroshot", "lora")]
    pares += [(f"{a}_lora", f"{a}_zeroshot") for a in ("gemma4-e2b", "qwen35-4b", "s5-nano", "qwen35-08b")]
    return pd.DataFrame([comparar(arq[a], arq[b], r) for a, b in pares if a in arq and b in arq
                         for r in (None, "pt_nativo", "traduzido")])


def erros(arq: dict) -> pd.DataFrame:
    """Falsos positivos e negativos (e respostas ilegíveis) por modelo e fonte."""
    linhas = []
    for nome, a in arq.items():
        d = carregar(a)
        tipo = pd.Series("acerto", index=d.index)
        tipo[d.previsto.isna()] = "ilegivel"
        tipo[(d.rotulo == "legitimo") & (d.previsto == "phishing")] = "falso_positivo"
        tipo[(d.rotulo == "phishing") & (d.previsto == "legitimo")] = "falso_negativo"
        linhas.append(pd.crosstab([d.recorte, d.fonte], tipo).assign(modelo=nome))
    return pd.concat(linhas).reset_index().set_index(["modelo", "recorte", "fonte"]).fillna(0).astype(int) if linhas else pd.DataFrame()


def marcadores() -> pd.DataFrame:
    """Proporção de e-mails com [EMAIL] ou [LINK] no treino, por fonte e rótulo (atalho que o aluno poderia aprender)."""
    treino = pd.read_csv(RAIZ / "splits" / "treino.csv").id
    pt = emails_pt()
    pt = pt[pt.id.isin(treino)]
    return pt.corpo.str.contains(r"\[EMAIL\]|\[LINK\]").groupby([pt.origem, pt.fonte, pt.rotulo]).agg(["size", "mean"]).round(3)


def md(df: pd.DataFrame) -> str:
    if not len(df):
        return "_(sem dados)_"
    return (df if isinstance(df.index, pd.RangeIndex) else df.reset_index()).to_markdown(index=False, floatfmt=".3f")


def grafico(metricas: pd.DataFrame) -> None:
    """F1 com IC de 95% por modelo, um painel por recorte (reports/f1.png). No pt_nativo só há golpes: F1 ~ recall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = metricas.reset_index()
    recortes, modelos = list(m.recorte.unique()), sorted(m.modelo.unique())
    fig, eixos = plt.subplots(1, len(recortes), figsize=(4 * len(recortes), 0.45 * len(modelos) + 1.2), sharey=True)
    for ax, recorte in zip(eixos, recortes):
        d = m[m.recorte == recorte].set_index("modelo").reindex(modelos)
        ic = d.f1_ic.str.split("–", expand=True).astype(float)
        erro = [(d.f1 - ic[0]).clip(lower=0), (ic[1] - d.f1).clip(lower=0)]
        ax.barh(modelos, d.f1, height=0.6, color="#257de0", xerr=erro, error_kw={"ecolor": "#666666", "lw": 1, "capsize": 2})
        for y, (v, fim) in enumerate(zip(d.f1, ic[1].fillna(d.f1))):  # o rótulo fica depois do fim do IC
            ax.text(fim + 0.02, y, f"{v:.2f}", va="center", fontsize=8, color="#191919")
        ax.set(title=recorte, xlim=(0, 1.2), xticks=[0, 0.25, 0.5, 0.75, 1], xlabel="F1 (phishing)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
    eixos[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(RELATORIOS / "f1.png", dpi=150)
    plt.close(fig)


def main() -> dict[str, pd.DataFrame]:
    arq = arquivos()
    tabelas = {  # arquivo CSV: (título, tabela)
        "metricas": ("Métricas por recorte", pd.concat([avaliar(a) for a in arq.values()], ignore_index=True)[COLUNAS].set_index(["modelo", "recorte"])),
        "perguntas": ("Perguntas de pesquisa (McNemar pareado; p < 0,05 = diferença significativa)", perguntas(arq)),
        "campanhas_rnp": ("Recall por campanha no RNP", pd.concat({n: campanhas_rnp(a) for n, a in arq.items()}, names=["modelo"]).recall.unstack(0)
                                       if arq else pd.DataFrame()),
        "juiz": ("Notas do juiz (1 a 5)", notas_juiz() if (SAIDAS / "juiz" / "notas.jsonl").exists() else pd.DataFrame()),
        "erros": ("Erros por fonte", erros(arq)),
        "marcadores": ("Marcadores no treino", marcadores()),
    }
    RELATORIOS.mkdir(exist_ok=True)
    texto = "# Resultados do PhishExplain\n\n![F1 por modelo e recorte](f1.png)\n\n" + "\n\n".join(f"## {t}\n\n{md(d)}" for t, d in tabelas.values())
    (RELATORIOS / "resultados.md").write_text(texto + "\n", encoding="utf-8")
    for arquivo, (_, d) in tabelas.items():
        d.to_csv(RELATORIOS / f"{arquivo}.csv")
    if arq:
        grafico(tabelas["metricas"][1])
    print(f"Relatório em {RELATORIOS / 'resultados.md'}")
    return dict(tabelas.values())


if __name__ == "__main__":
    main()
