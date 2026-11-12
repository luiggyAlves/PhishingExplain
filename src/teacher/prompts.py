# Prompt do professor: recebe o e-mail e o rótulo correto e devolve as evidências e a explicação no formato do contrato (src/schema.py)

import json

import pandas as pd

from src.config import DADOS_INTERIM, RAIZ, SEMENTE
from src.schema import LEGITIMIDADE, PHISHING, Resposta, formatar_email
from src.teacher.cliente import perguntar
from src.teacher.validar import validar

lista = lambda sinais: "\n".join(f"- {k}: {v}" for k, v in sinais.items())

SISTEMA = f"""Você é especialista em segurança de e-mail e escreve explicações para pessoas leigas.
Você recebe um e-mail e a classificação correta dele (phishing ou legítimo). Mostre as evidências do próprio e-mail que sustentam essa classificação e explique-as.

Responda só com JSON, com os campos nesta ordem:
1. "indicadores": de 1 a 6 sinais, cada um com "tipo" (um nome das listas abaixo) e "trecho" (cópia literal de 3 a 300 caracteres de uma parte do e-mail, incluindo as linhas "De:" e "Assunto:"). Copie o trecho exatamente como está: não corrija, não traduza, não resuma.
2. "classificacao": "phishing" ou "legitimo".
3. "explicacao": de 2 a 4 frases curtas, em português do Brasil, para um leigo: por que o e-mail é ou não é golpe e o que a pessoa deve fazer.

Sinais de phishing:
{lista(PHISHING)}

Sinais de legitimidade:
{lista(LEGITIMIDADE)}

Regras:
- Pelo menos um indicador deve ser do lado da classificação. Um e-mail legítimo pode ter sinais de phishing (como urgência) e vice-versa; cite-os só se ajudarem a explicar.
- Cite só o que está escrito. A ausência de um sinal (por exemplo, "não pede senha") não é trecho: mencione-a na explicação.
- Nunca cite um trecho que seja só [EMAIL] ou [LINK]; são marcadores de anonimização.
- Escreva a explicação como se você mesmo tivesse analisado o e-mail. Nunca diga que a classificação foi informada, fornecida ou dada."""


def mensagens(email: str, rotulo: str) -> list[dict]:
    """Conversa para anotar um e-mail já formatado com formatar_email."""
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f"Classificação correta: {rotulo}\n\n<email>\n{email}\n</email>"},
    ]


if __name__ == "__main__":
    df = pd.read_parquet(DADOS_INTERIM / "emails_agrupados.parquet")
    reservados = set(pd.read_csv(RAIZ / "splits" / "teste.csv").grupo)
    livres = df[(df.fonte == "pf_semente") & ~df.grupo.isin(reservados)]
    amostra = livres.groupby("rotulo").sample(2, random_state=SEMENTE).query("rotulo != 'spam'")

    validas = 0
    for _, e in amostra.iterrows():
        email = formatar_email(e.remetente, e.assunto, e.corpo)
        r = perguntar(mensagens(email, e.rotulo), esquema=Resposta)
        print(f"\n=== {e.id} ({e.rotulo}) {r.segundos:.1f} s, {r.tokens_entrada} + {r.tokens_saida} tokens"
              f"{' CORTADA' if r.cortada else ''}")
        print(r.texto)
        try:
            Resposta.model_validate_json(r.texto)
            validas += 1
            ordem = list(json.loads(r.texto)) == list(Resposta.model_fields)
            print(f"válida | ordem do contrato: {ordem} | erros dos validadores: {validar(r.texto, email, e.rotulo)}")
        except ValueError as erro:
            print("INVÁLIDA:", erro)
    print(f"\n{validas}/{len(amostra)} respostas passam no contrato")
