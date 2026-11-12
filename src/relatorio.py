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
    return df.to_markdown(floatfmt=".3f") if len(df) else "_(sem dados)_"


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
    texto = "# Resultados do PhishExplain\n\n" + "\n\n".join(f"## {t}\n\n{md(d)}" for t, d in tabelas.values())
    (RELATORIOS / "resultados.md").write_text(texto + "\n", encoding="utf-8")
    for arquivo, (_, d) in tabelas.items():
        d.to_csv(RELATORIOS / f"{arquivo}.csv")
    print(f"Relatório em {RELATORIOS / 'resultados.md'}")
    return dict(tabelas.values())


if __name__ == "__main__":
    main()
