# Nessa etapa vamos separar o teste, para que ele fique reservado e não seja mexido durante o treino, evitando vazamento de dados.
# Todos os emails usados no teste serão de autoria humana, nenhum sintético de LLM,usaremos tanto os emails nativos da RNP (que são nossa métrica principal para detecção real), quanto os traduzidos, mas ainda escritos originalmente por humanos
# Todo o split ocorre por grupo, justamente para endereçarmos a questão de honestidade das métricas que relatamos em agrupar.py

import pandas as pd 
from src.config import DADOS_INTERIM, RAIZ, SEMENTE 

GRUPOS_POR_ROTULO = 150 
DESTINO = RAIZ / "splits" / "teste.csv"

def main() -> None:
    df = pd.read_parquet(DADOS_INTERIM / "emails_agrupados.parquet")

    # Teste já congelado: os ids não mudam, só o grupo acompanha a tabela mestre (ids que saíram da tabela saem do teste)
    if DESTINO.exists():
        teste = pd.read_csv(DESTINO)
        atual = teste.merge(df[["id", "grupo"]], on="id", suffixes=("_antigo", ""))
        print(f"Teste já congelado: {len(atual)} de {len(teste)} ids seguem na tabela mestre, "
              f"{(atual.grupo != atual.grupo_antigo).sum()} mudaram de grupo")
        atual[teste.columns].to_csv(DESTINO, index=False)
        reservados = df[df["grupo"].isin(atual["grupo"]) & ~df["id"].isin(atual["id"])]
        print(pd.crosstab(atual["recorte"], atual["rotulo"], margins=True, margins_name="total"))
        print(f"\nFora do treino por serem dos mesmos grupos: {len(reservados):,} e-mails")
        return
    humanos = df[(df["autoria"] == "humano") & df["rotulo"].isin(["phishing", "legitimo"])]

    # recorte 1: emails humanos escritos em português
    nativos = humanos[humanos["idioma"]=="pt"].assign(recorte="pt_nativo")

    # recorte 2: emails humanos escritos em ingles, sorteado por rótulos, que serão posteriormente traduzidos
    candidatas = humanos[
        (humanos["fonte"] == "pf_semente") &
        (humanos["idioma"] == "en") & 
        ~humanos["grupo"].isin(nativos["grupo"])
    ]

    sorteados = candidatas.drop_duplicates("grupo").groupby("rotulo").sample(n=GRUPOS_POR_ROTULO, random_state=SEMENTE)
    
    traduzir = candidatas[candidatas["grupo"].isin(sorteados["grupo"])].assign(recorte="traduzido")

    teste = pd.concat([nativos, traduzir])[["id", "grupo", "fonte", "rotulo", "recorte"]]
    DESTINO.parent.mkdir(exist_ok=True)
    teste.sort_values("id").to_csv(DESTINO, index=False)

    reservados = df[df["grupo"].isin(teste["grupo"]) & ~df["id"].isin(teste["id"])]

    print(f"Teste congelado em {DESTINO.relative_to(RAIZ)}: {len(teste)} e-mails\n")
    print(pd.crosstab(teste["recorte"], teste["rotulo"], margins=True, margins_name="total"))
    print(f"\nFora do treino por serem dos mesmos grupos: {len(reservados):,} e-mails")
    print(reservados["fonte"].value_counts().to_string())


if __name__ == "__main__":
    main()