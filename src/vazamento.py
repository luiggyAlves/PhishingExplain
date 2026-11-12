# 4.2: checagem de vazamento entre treino/validação e teste, no texto que o modelo vê.
# MinHash acha quase-cópias de forma; embeddings multilíngues acham e-mails com o mesmo sentido.
# Os pares mais parecidos vão para reports/vazamento.csv para revisão manual.

import pandas as pd
from datasketch import MinHashLSH

from src.config import EMBEDDINGS_MODELO, RAIZ, RELATORIOS
from src.dados import emails_pt
from src.data.agrupar import LIMIAR, NUM_PERM, assinatura
from src.exportar import emails_do_teste
from src.schema import formatar_email

LIMIAR_EMBEDDING = 0.97  # o e5 concentra os cossenos perto de 0,9; acima disso o par vai para revisão manual (o CSV vem ordenado)


def textos(df: pd.DataFrame) -> list[str]:
    return [formatar_email(e.remetente, e.assunto, e.corpo) for e in df.itertuples()]


def main() -> None:
    from sentence_transformers import SentenceTransformer  # importado aqui: só este passo precisa do torch

    pt = emails_pt().set_index("id")
    ids = pd.concat([pd.read_csv(RAIZ / "splits" / f"{n}.csv") for n in ("treino", "validacao")]).id
    treino, teste = pt.loc[ids].reset_index(), emails_do_teste()
    t_treino, t_teste = textos(treino), textos(teste)

    lsh = MinHashLSH(threshold=LIMIAR, num_perm=NUM_PERM)
    for i, t in enumerate(t_treino):
        lsh.insert(i, assinatura(t))
    minhash = [len(lsh.query(assinatura(t))) for t in t_teste]

    modelo = SentenceTransformer(EMBEDDINGS_MODELO)
    vetor = lambda ts: modelo.encode([f"query: {t}" for t in ts], normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    sim = vetor(t_teste) @ vetor(t_treino).T
    vizinho = sim.argmax(axis=1)

    rel = pd.DataFrame({"id_teste": teste.id_original, "recorte": teste.recorte, "pares_minhash": minhash,
                        "id_treino": treino.id.values[vizinho], "similaridade": sim.max(axis=1).round(4),
                        "texto_teste": [t[:300] for t in t_teste], "texto_treino": [t_treino[j][:300] for j in vizinho]})
    rel = rel.sort_values("similaridade", ascending=False)
    rel.to_csv(RELATORIOS / "vazamento.csv", index=False)
    print(f"Teste: {len(teste)} | treino+validação: {len(treino)}")
    print(f"Com quase-cópia por MinHash: {(rel.pares_minhash > 0).sum()}")
    print(f"Similaridade de embedding >= {LIMIAR_EMBEDDING}: {(rel.similaridade >= LIMIAR_EMBEDDING).sum()} (revisar à mão)")
    print(rel.groupby("recorte").similaridade.describe().round(3).to_string())


if __name__ == "__main__":
    main()
