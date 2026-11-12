# Decisões tomadas após inspecionar os dados
    # 1) Manteremos apenas português e inglês, tem outros idiomas no Phishing, mas configuram apenas ~3% da base de dados, então deixaremos de fora
    # 2) Vamos tirar o dataset do zefang-liu, pois ele mistura spam com phishing e possivelmente se sobrepõe ao PhishFuzzer (usam a mesma fonte do kaggle). Talvez possamos colocá-lo, em caso de necessidade, testando deduplicação com o PhishFuzzer
    # 3) Spam ficará de fora dessa primeira versão, inclusive os do PhishFuzzer

# Normalização dos dados: temos diversas fontes e cada uma chama suas colunas de um jeito (body, corpo, type) etc. Nessa etapa iremos colocar todas no mesmo modelo, no mesmo esquema de dados, para que depois consigamos limpá-las e transformá-las em uma fonte consistente

import pandas as pd 

from src.config import DADOS_INTERIM, DADOS_RAW, RAIZ 
from src.data.inspecionar import ler 

COLUNAS = [
    "id", "fonte", "grupo", "autoria", "idioma", "traduzido", "remetente", "assunto", "corpo", "urls", "anexos", "rotulo", "rotulo_original", "motivacao",
]

# vamos epgar os rotulos brutos das fontes e normaliá-los todos da mesma forma
ROTULOS = {
    "Phishing": "phishing", 
    "Valid":"legitimo",
    "Legítimo":"legitimo",
    "1":"phishing",
    "0":"legitimo",
    "Spam":"spam"
}

# Tradução dos rotulos: pega os rótulos atuais e aplica o dicionário ROTULOS que criamos

def padronizar_rotulos(serie: pd.Series) -> pd.Series:
    originais = serie.astype(str).str.strip()
    desconhecidos = set(originais) - set(ROTULOS)
    if desconhecidos:
        raise ValueError(f"Rótulos sem tradução: {desconhecidos}")
    return originais.map(ROTULOS) # Para cada elemento da serie originais, vai aplicar o dicionario ROTULO

# Normalizando nulo, texto solto ou lista em um só formato: uma lista de str, serve para tratar direito na hora de remover ou modificar 

def como_lista(valor) -> list[str]:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return []
    if isinstance(valor,str):
        return [valor] if valor.strip() else []
    return [str(v) for v in valor if v ]

def phishfuzzer(arquivo: str, fonte: str, autoria: str) -> pd.DataFrame:
    bruto = ler(DADOS_RAW / "phishfuzzer" / arquivo)
    return pd.DataFrame({
        "id": fonte + ":" + bruto["No."].astype(str),
        "fonte": fonte,
        "grupo": "pf:" + bruto["Original_ID"].astype(str),
        "autoria": autoria,
        "idioma": bruto["Language"],
        "traduzido": False,
        "remetente": bruto["Sender"],
        "assunto": bruto["Subject"],
        "corpo": bruto["Body"],
        "urls": bruto["URL"].map(como_lista),
        "anexos": bruto["File"].map(como_lista),
        "rotulo": padronizar_rotulos(bruto["Type"]),
        "rotulo_original": bruto["Type"].astype(str),
        "motivacao": bruto["Motivation"],
    })

def ptbr(
    arquivo: str, fonte: str, autoria: str, coluna_corpo: str, usar_assunto: bool = True
) -> pd.DataFrame:
    bruto = ler(DADOS_RAW / "ptbr_estilometria" / arquivo)
    ids = fonte + ":" + bruto.index.astype(str)
    return pd.DataFrame({
        "id": ids,
        "fonte": fonte,
        "grupo": ids,
        "autoria": autoria,
        "idioma": "pt",
        "traduzido": False,
        "remetente": None,
        "assunto": bruto["Assunto"] if usar_assunto else None,
        "corpo": bruto[coluna_corpo],
        "urls": [[] for _ in range(len(bruto))],
        "anexos": [[] for _ in range(len(bruto))],
        "rotulo": padronizar_rotulos(bruto["Label"]),
        "rotulo_original": bruto["Label"].astype(str),
        "motivacao": None,
    })

def checar(condicao:bool, mensagem:str)->None:
    if not condicao:
        raise ValueError(mensagem)

# Serve para conferir se o esquema está seguindo as regras (dentro dos rótulos, quantidade certa de colunas etc. Antes de salvar de vez)
def validar(df: pd.DataFrame)->None:
    checar(list(df.columns) ==COLUNAS, "colunas fora do esquema")
    checar(df["id"].is_unique, "há ids repetidos")
    checar(df["rotulo"].isin(["phishing", "legitimo", "spam"]).all(), "rótulo inválido")
    checar(df["autoria"].isin(["humano", "llm"]).all(), "autoria inválida")
    checar(df["corpo"].notna().all(), "há corpos nulos")

# Estamos transformando em um vetor de dataframes, usando as funções que criamos anteriormente, todos os datasets, como formatamos eles para terem um formato único, poderemos concatená-los e fazer uma grande base de dados, que será limpa depois
def main() -> None:
    partes = [
        phishfuzzer("original_seed.json", "pf_semente", "humano"),
        phishfuzzer("entity_rephrased.json", "pf_variante", "llm"),
        ptbr("dataset_fraudes_rnp.csv", "pt_rnp", "humano", "Texto_Limpo"),
        ptbr("dataset_d.csv", "pt_d", "llm", "Texto_Limpo"),
        ptbr("dataset_b.csv", "pt_b", "llm", "Corpo", usar_assunto=False),
        ptbr("dataset_opara.csv", "pt_opara", "llm", "Corpo"),
    ]

    emails = pd.concat(partes, ignore_index=True)[COLUNAS]

# Depois de concatenar, deu algum problema na normalização?
    validar(emails)

# Se tudo deu certo, salvamos como parquet a base de dados
    DADOS_INTERIM.mkdir(parents=True, exist_ok=True)
    destino = DADOS_INTERIM / "emails.parquet"
    emails.to_parquet(destino, index=False)

    print(f"{len(emails):,} e-mails salvos em {destino.relative_to(RAIZ)}\n")
    print(pd.crosstab(emails["fonte"], emails["rotulo"], margins=True, margins_name="total"))


if __name__ == "__main__":
    main()

