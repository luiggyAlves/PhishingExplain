# Agora já temos os dados, mas precisamos entender eles, fazer a análise exploratória desses dados, respondendo algumas perguntas sobre os dados: 
    # 1. Qual coluna tem o texto? Qual coluna tem o assunto e qual tem o rótulo? 
    # 2. Quais valores o rótulo assume? 
    # 3. Em que idiomas cada dataset está ?
    # 4. Há muitos nulos ou duplicados? E como iremos tratar eles? 
# Essa etapa será apenas para inspecionar, iremos tratar esses dados em outro script 

import json 
from pathlib import Path 
import pandas as pd
from src.config import DADOS_RAW, SEMENTE

MAX_CATEGORIAS = 25 # tratamos como categorias colunas que tiverem até 25 valores distintos 
TAMANHO_AMOSTRA = 300 # para ficar mais fácil de ler nessa etapa de exploração, iremos mostrar apenas os 300 primeiros caracteres de cada texto de exemplo

def ler(caminho: Path) -> pd.DataFrame:
    """Abre um arquivo bruto como tabela, conforme a extensão."""
    if caminho.suffix == ".parquet":
        return pd.read_parquet(caminho)
    if caminho.suffix == ".csv":
        return pd.read_csv(caminho)
    if caminho.suffix == ".json":
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        # Alguns JSONs guardam a lista de registros dentro de uma chave
        if isinstance(dados, dict):
            listas = [v for v in dados.values() if isinstance(v, list)]
            dados = listas[0] if listas else [dados]
        return pd.DataFrame(dados)
    raise ValueError(f"Formato não suportado: {caminho.suffix}")


def eh_texto(serie: pd.Series) -> bool:
    """Diz se a maioria dos valores da coluna é texto."""
    return serie.map(lambda v: isinstance(v, str)).mean() > 0.5


def resumir(df: pd.DataFrame, nome: str) -> None:
    print("=" * 80)
    print(f"{nome}: {len(df):,} linhas, {df.shape[1]} colunas")
    print("=" * 80)

    duplicadas = df.astype(str).duplicated().sum()
    print(f"Linhas duplicadas (todas as colunas iguais): {duplicadas:,}")

    for coluna in df.columns:
        serie = df[coluna]
        nulos = serie.isna().sum()
        distintos = serie.astype(str).nunique()
        print(f"\n- {coluna!r}: {nulos:,} nulos, {distintos:,} valores distintos")

        if distintos <= MAX_CATEGORIAS:
            for valor, n in serie.astype(str).value_counts().items():
                print(f"    {valor[:60]!r}: {n:,}")
        elif eh_texto(serie):
            tamanhos = serie.dropna().astype(str).str.len()
            vazios = (serie.dropna().astype(str).str.strip() == "").sum()
            print(
                f"    tamanho em caracteres: mediana {tamanhos.median():,.0f}, "
                f"mínimo {tamanhos.min():,}, máximo {tamanhos.max():,}; vazios: {vazios:,}"
            )

    print("\nExemplos:")
    amostra = df.sample(n=min(2, len(df)), random_state=SEMENTE)
    for _, linha in amostra.iterrows():
        print("-" * 40)
        for coluna, valor in linha.items():
            texto = str(valor).replace("\n", " ")
            print(f"  {coluna}: {texto[:TAMANHO_AMOSTRA]}")
    print()


def main() -> None:
    arquivos = sorted(
        p for p in DADOS_RAW.rglob("*")
        if p.is_file() and p.suffix in {".csv", ".json", ".parquet"}
    )
    for caminho in arquivos:
        nome = str(caminho.relative_to(DADOS_RAW))
        try:
            df = ler(caminho)
        except Exception as erro:
            print(f"{nome}: FALHOU ao ler: {erro}\n")
            continue
        resumir(df, nome)


if __name__ == "__main__":
    main()

# Decisões tomadas após inspecionar os dados
    # 1) Manteremos apenas português e inglês, tem outros idiomas no Phishing, mas configuram apenas ~3% da base de dados, então deixaremos de fora
    # 2) Vamos tirar o dataset do zefang-liu, pois ele mistura spam com phishing e possivelmente se sobrepõe ao PhishFuzzer (usam a mesma fonte do kaggle). Talvez possamos colocá-lo, em caso de necessidade, testando deduplicação com o PhishFuzzer
    # 3) Spam ficará de fora dessa primeira versão, inclusive os do PhishFuzzer