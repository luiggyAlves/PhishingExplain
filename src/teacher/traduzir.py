# Tradução EN -> PT-BR pelo professor. Cada tradução vira uma linha <id>@pt; grupo, fonte e rótulo vêm do original (src/dados.py).
# Uso: python -m src.teacher.traduzir           (treino: sementes livres + 1 variante por grupo livre)
#      python -m src.teacher.traduzir --teste   (sementes do recorte "traduzido" do teste)

import argparse
import json
import re

import pandas as pd
from pydantic import BaseModel

from src.config import CFG, DADOS_INTERIM, SEMENTE
from src.dados import TRADUCOES, mestre, teste
from src.schema import cortar
from src.teacher.cliente import perguntar
from src.teacher.lote import rodar

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
    original = {"assunto": e["assunto"] or "", "corpo": cortar(e["corpo"])}
    r = perguntar(
        [{"role": "system", "content": SISTEMA}, {"role": "user", "content": json.dumps(original, ensure_ascii=False)}],
        max_tokens=4096, esquema=Traducao, **AMOSTRAGEM,
    )
    try:
        t = Traducao.model_validate_json(r.texto)
    except ValueError as erro:
        return {"id_original": e["id_original"], "erros": [f"JSON inválido: {erro}"], "texto": r.texto}
    faltando = [x for x in PRESERVAR.findall(original["assunto"] + "\n" + original["corpo"]) if x not in t.assunto + "\n" + t.corpo]
    erros = ["resposta cortada"] * r.cortada + [f"não preservou {x!r}" for x in dict.fromkeys(faltando)]
    return {"id_original": e["id_original"], "assunto": t.assunto or None, "corpo": t.corpo, "erros": erros,
            "segundos": round(r.segundos, 1)}


def selecionar(do_teste: bool) -> pd.DataFrame:
    df, split = mestre(), teste()
    if do_teste:
        return df[df.id.isin(split.id[split.recorte == "traduzido"])]
    livres = df[(df.idioma == "en") & df.rotulo.isin(["phishing", "legitimo"]) & ~df.grupo.isin(split.grupo)]
    sementes = livres[livres.fonte == "pf_semente"]
    variantes = livres[livres.fonte == "pf_variante"].groupby("grupo").sample(1, random_state=SEMENTE)  # 1 por grupo
    escolhidos = pd.concat([sementes, variantes])
    n = CFG.limite_exemplos
    return escolhidos.groupby("rotulo").sample(n // 2, random_state=SEMENTE) if n else escolhidos


def como_itens(df: pd.DataFrame) -> list[dict]:
    """Linhas da tabela como dicts, com None no lugar de NaN (o JSON não aceita NaN)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", action="store_true", help="traduz o recorte 'traduzido' do teste")
    originais = selecionar(ap.parse_args().teste)[["id", "assunto", "corpo"]]
    itens = [e | {"id": f"{e['id']}@pt", "id_original": e["id"]} for e in como_itens(originais)]
    rodar(itens, traduzir, TRADUCOES, REPROVADAS)
