# Na parte da normalização,juntamos todas as bases de dados em uma só base de dados consistente e padronizada quanto a rótulos e atributos. Nessa etapa vamos limpar a base de dados em dois novos termos: 
    # 1) normalizar o texto, filtrar idiomas e registros malformados, anonimizar tudo e remover duplicatas exatas (emails que tem exatamente os mesmos caracteres)
    # 2) Remover as quase duplicatas (tipo emails que só mudam o remetente ou o protocolo) e juntá-las num só grupo
# Tudo que for removido irá para uma base de dados chamada emails_descartados, para termos a proveniencia das remoções 
# Aqui iremos tratar os primeiros casos, que são os erros de formatação e as duplicatas iguais, em agrupar.py estarão sendo tratadas 

import re 
import unicodedata
import pandas as pd 
from src.config import DADOS_INTERIM

IDIOMAS = {"pt", "en"}
MIN_CORPO=30
MAX_CORPO=20_000
MAX_CAMPO_CURTO = 300

# Aqui estamos usando regex para tratar alguns fatores, como espaços, caracteres invisiveis, links removidos etc. O objetivo aqui é padronizar todos esses aspectos dos emails
INVISIVEIS = re.compile("[\u00ad\u034f\u200b-\u200f\u2060\ufeff]")
ESPACOS = re.compile(r"[ \t\u00a0]+")
EMAIL_FICTICIO = re.compile(
    r"\b[\w.+-]+@(?:email\.com(?:\.br)?|email\.pt|empresa\.com\.br)\b", re.IGNORECASE
)
LINK_REMOVIDO = re.compile(r"\[link removed\]", re.IGNORECASE)

# Vamos pegar um campo de texto e nromaliza-lo usando regex, sem alterar seu significado e conteúdo
def limpar_texto(texto):
    if not isinstance(texto, str): # Se o texto não for uma string nem faz sentido continuar, o formato já está errado
        return None
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = INVISIVEIS.sub("", texto)
    texto = ESPACOS.sub(" ", texto)  # colapsa espaços repetidos em um só (com "" os e-mails perdiam todos os espaços)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = texto.strip()
    return texto or None

# Vamos pegar os marcadores de anonimização, sobretudo dos da RNP e tranformar em um só tipo, porque atualmente tem vários distintos
def padronizar_anonimizacao(texto):
    if not isinstance(texto, str): 
        return None
    texto = EMAIL_FICTICIO.sub("[EMAIL]", texto)
    return LINK_REMOVIDO.sub("[LINK]", texto)

# Vamos tirar as linhas qe não queremos, guardá-las explicando o porque elas foram tiradas e retornamos as linhas corretas, é importante guardar isso para rastreabilidade e proveniência
def separar(df, mascara, motivo, descartes):
    removidas = df[mascara].assign(motivo=motivo)
    descartes.append(removidas)
    print(f" {motivo}: {len(removidas):,}")
    return df[~mascara]

def main() -> None: 
    # lendo o dataset
    df=pd.read_parquet(DADOS_INTERIM / "emails.parquet")
    descartes = []

    # removendo texto unicode, quebras de linhas, caracteres invisiveis e espaços
    for coluna in ["remetente", "assunto", "corpo"]:
        df[coluna] = df[coluna].map(limpar_texto)

    # padronizando os marcadores de anomização
    for coluna in ["assunto", "corpo"]:
        df[coluna] = df[coluna].map(padronizar_anonimizacao)

    # retirando o que não é portugues e ingles 
    df = separar(df, ~df["idioma"].isin(IDIOMAS), "idioma fora de pt/en", descartes)

    # retirando registros malformados
    tamanho = df["corpo"].fillna("").str.len()
    df = separar(df, tamanho < MIN_CORPO, "corpo vazio ou curto demais", descartes)
    tamanho = df["corpo"].str.len()
    df = separar(df, tamanho>MAX_CORPO, "corpo longo demais", descartes)
    longo = (df["assunto"].fillna("").str.len() > MAX_CAMPO_CURTO) | (df["remetente"].fillna("").str.len() > MAX_CAMPO_CURTO)
    df = separar(df, longo, "assunto ou remetente malformado", descartes)

    # removendo duplicatas exatas, ignorando maiúsculas e minúsculas
    chave = (df["assunto"].fillna("") + "\n" + df["corpo"]).str.lower().str.split().str.join(" ")
    df = df.assign(chave=chave)
    conflito = df.groupby("chave")["rotulo"].transform("nunique") > 1
    df = separar(df, conflito, "duplicata com rótulos diferentes", descartes)

    df = df.sort_values("autoria", key=lambda s: s.ne("humano"), kind="stable")
    repetida = df.duplicated("chave", keep="first")
    df = separar(df, repetida, "duplicata exata", descartes)
    df = df.sort_index().drop(columns="chave")

    # salvando em parquet 

    df.to_parquet(DADOS_INTERIM / "emails_limpos.parquet", index=False)
    descartados = pd.concat(descartes, ignore_index=True).drop(columns="chave", errors="ignore")
    descartados.to_parquet(DADOS_INTERIM / "emails_descartados.parquet", index=False)

    print(f"\nSaída: {len(df):,} e-mails ({len(descartados):,} descartados)\n")
    print(pd.crosstab(df["fonte"], df["rotulo"], margins=True, margins_name="total"))

    # Checagem de atalho: quem ainda carrega marcadores de anonimização?
    marcado = df["corpo"].str.contains(r"\[EMAIL\]|\[LINK\]", regex=True)
    print("\nE-mails com [EMAIL] ou [LINK], por fonte e rótulo:")
    print(pd.crosstab([df["fonte"], df["rotulo"]], marcado.rename("tem marcador")))


if __name__ == "__main__":
    main()