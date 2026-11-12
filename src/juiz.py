# 8.2: juiz de qualidade das explicações. Um modelo de outra família (CFG: JUIZ_MODELO) dá notas de 1 a 5
# sem saber qual modelo escreveu a resposta. Gera também 30 casos para o usuário calibrar o juiz à mão.
# Uso: python -m src.juiz --url http://localhost:8000/v1 outputs/previsoes/a.jsonl outputs/previsoes/b.jsonl ...

import argparse
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import DADOS_PROCESSED, JUIZ_MODELO, RELATORIOS, SAIDAS, SEMENTE
from src.evaluate import limpar
from src.teacher.lote import ler, rodar

NOTAS = SAIDAS / "juiz" / "notas.jsonl"
CRITERIOS = ["correcao", "especificidade", "clareza", "orientacao"]
N_CALIBRACAO = 30

SISTEMA = """Você avalia explicações de um detector de phishing escritas para pessoas leigas.
Você recebe o e-mail, a classificação correta e a resposta do detector. Dê notas de 1 (péssimo) a 5 (excelente):
- correcao: a classificação está certa e as afirmações batem com o e-mail, sem inventar nada.
- especificidade: aponta evidências concretas deste e-mail, não frases genéricas que serviriam para qualquer e-mail.
- clareza: um leigo entende sem jargão técnico.
- orientacao: diz claramente o que a pessoa deve fazer.
Responda só com JSON: as quatro notas e um "comentario" de uma frase."""


class Notas(BaseModel):
    correcao: int = Field(ge=1, le=5)
    especificidade: int = Field(ge=1, le=5)
    clareza: int = Field(ge=1, le=5)
    orientacao: int = Field(ge=1, le=5)
    comentario: str


def avaliador(cliente: OpenAI, modelo: str):
    def avaliar(c: dict) -> dict:
        conteudo = f"Classificação correta: {c['rotulo']}\n\n{c['email']}\n\nResposta do detector:\n{c['resposta']}"
        r = cliente.chat.completions.create(
            model=modelo, temperature=0.0, max_tokens=300,
            messages=[{"role": "system", "content": SISTEMA}, {"role": "user", "content": conteudo}],
            response_format={"type": "json_schema", "json_schema": {"name": "Notas", "schema": Notas.model_json_schema()}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # juízes com raciocínio gastariam os tokens pensando
        )
        return Notas.model_validate_json(r.choices[0].message.content).model_dump() | {"modelo": c["modelo"]}
    return avaliar


def casos(arquivos: list[Path]) -> list[dict]:
    """Uma resposta por (modelo, e-mail), só quando há explicação para julgar."""
    teste = {l["id"]: l for l in map(json.loads, (DADOS_PROCESSED / "test.jsonl").open(encoding="utf-8"))}
    saida = []
    for arq in arquivos:
        for p in ler(arq):
            texto = limpar(p.get("texto", ""))
            if p["id"] in teste and '"explicacao"' in texto:
                t = teste[p["id"]]
                saida.append({"id": f"{arq.stem}|{p['id']}", "modelo": arq.stem, "rotulo": t["rotulo"],
                              "email": t["messages"][1]["content"], "resposta": texto})
    return saida


def tabela() -> pd.DataFrame:
    notas = pd.DataFrame(ler(NOTAS))
    return notas.groupby("modelo")[CRITERIOS].mean().round(2).assign(n=notas.groupby("modelo").size()) if len(notas) else pd.DataFrame()


def calibracao(lista: list[dict]) -> None:
    """30 casos sorteados, com as notas do juiz, para o usuário dar as próprias notas e comparar."""
    notas = {n["id"]: n for n in ler(NOTAS)}
    amostra = pd.DataFrame([c for c in lista if c["id"] in notas]).sample(frac=1, random_state=SEMENTE).head(N_CALIBRACAO)
    blocos = [f"## Caso {k + 1}\n\nClassificação correta: {c.rotulo}\n\n```text\n{c.email}\n```\n\n```json\n{c.resposta}\n```\n\n"
              f"Juiz: {', '.join(f'{x} {notas[c.id][x]}' for x in CRITERIOS)} ({notas[c.id]['comentario']})\n\n"
              f"Suas notas: correcao _ especificidade _ clareza _ orientacao _\n"
              for k, c in enumerate(amostra.itertuples())]
    (RELATORIOS / "calibracao_juiz.md").write_text("# Calibração do juiz\n\nO modelo de cada caso está escondido de propósito.\n\n"
                                                   + "\n".join(blocos), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("previsoes", nargs="+", type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--modelo", default=JUIZ_MODELO)
    args = ap.parse_args()
    NOTAS.parent.mkdir(parents=True, exist_ok=True)
    lista = casos(args.previsoes)
    rodar(lista, avaliador(OpenAI(base_url=args.url, api_key="sem-chave", timeout=600), args.modelo),
          NOTAS, NOTAS.with_name("falhas.jsonl"))
    calibracao(lista)
    print(tabela().to_string())
