# Tradução EN -> PT-BR pelo professor. Cada tradução vira uma linha nova (<id>@pt) com o mesmo grupo do original.
# Uso: python -m src.teacher.traduzir           (sementes livres, para o treino)
#      python -m src.teacher.traduzir --teste   (sementes do recorte "traduzido" do teste)

import argparse
import json
import re

import pandas as pd
from pydantic import BaseModel

from src.config import CFG, DADOS_INTERIM, RAIZ, SEMENTE
from src.schema import cortar
from src.teacher.cliente import perguntar
from src.teacher.lote import rodar

APROVADAS = DADOS_INTERIM / "traducoes.jsonl"
REPROVADAS = DADOS_INTERIM / "traducoes_reprovadas.jsonl"
AMOSTRAGEM = {"temperature": 0.2, "presence_penalty": 0.0}  # tradução fiel: pouca variação e sem punir repetição de termos
PRESERVAR = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+|\[EMAIL\]|\[LINK\]")

SISTEMA = """Você traduz e-mails do inglês para o português do Brasil.
Você recebe um JSON com "assunto" e "corpo" e responde só com um JSON com os mesmos campos, traduzidos.

Regras:
- Escreva em português brasileiro natural, como um brasileiro escreveria esse e-mail.
- Preserve sem alterar: URLs, endereços de e-mail, números, valores, datas, códigos, nomes de pessoas, marcas e empresas, e os marcadores [EMAIL] e [LINK].
- Preserve o tom (formal, informal, alarmista) e as quebras de linha.
- Quando o original tem erros de escrita ou frases estranhas, reproduza erros parecidos em português: eles são pistas de golpe.
- Não resuma, não acrescente nem explique nada. Assunto vazio continua vazio."""


class Traducao(BaseModel):
    assunto: str
    corpo: str


def traduzir(e: dict) -> dict:
    original = {"assunto": e["assunto"] if isinstance(e["assunto"], str) else "", "corpo": cortar(e["corpo"])}
    r = perguntar(
        [{"role": "system", "content": SISTEMA}, {"role": "user", "content": json.dumps(original, ensure_ascii=False)}],
        max_tokens=4096, esquema=Traducao, **AMOSTRAGEM,
    )
    try:
        t = Traducao.model_validate_json(r.texto)
        faltando = [x for x in PRESERVAR.findall(original["assunto"] + "\n" + original["corpo"])
                    if x not in t.assunto + "\n" + t.corpo]
        erros = ["resposta cortada"] * r.cortada + [f"não preservou {x!r}" for x in dict.fromkeys(faltando)]
    except ValueError as erro:
        return {"erros": [f"JSON inválido: {erro}"], "texto": r.texto}
    return {"id_original": e["id_original"], "fonte": e["fonte"], "grupo": e["grupo"], "autoria": e["autoria"],
            "rotulo": e["rotulo"], "remetente": e["remetente"], "assunto": t.assunto or None, "corpo": t.corpo,
            "idioma": "pt", "traduzido": True, "erros": erros, "segundos": round(r.segundos, 1)}


def selecionar(teste: bool) -> pd.DataFrame:
    df = pd.read_parquet(DADOS_INTERIM / "emails_agrupados.parquet")
    split = pd.read_csv(RAIZ / "splits" / "teste.csv")
    if teste:
        return df[df.id.isin(split.id[split.recorte == "traduzido"])]
    livres = df[(df.fonte == "pf_semente") & (df.idioma == "en") & df.rotulo.isin(["phishing", "legitimo"])
                & ~df.grupo.isin(split.grupo)]
    n = CFG.limite_exemplos
    return livres.groupby("rotulo").sample(n // 2, random_state=SEMENTE) if n else livres


def como_itens(df: pd.DataFrame) -> list[dict]:
    """Linhas da tabela como dicts, com None no lugar de NaN (o JSON não aceita NaN)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", action="store_true", help="traduz o recorte 'traduzido' do teste")
    originais = selecionar(ap.parse_args().teste)[["id", "fonte", "grupo", "autoria", "rotulo", "remetente", "assunto", "corpo"]]
    itens = [e | {"id": f"{e['id']}@pt", "id_original": e["id"]} for e in como_itens(originais)]
    rodar(itens, traduzir, APROVADAS, REPROVADAS)
