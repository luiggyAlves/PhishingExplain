# Prompt do aluno (4.3): fonte única usada no treino, nas previsões e na avaliação.
# Mesmas instruções e listas de sinais do professor, mas o aluno nunca recebe o rótulo: ele decide.

from src.schema import LEGITIMIDADE, PHISHING

lista = lambda sinais: "\n".join(f"- {k}: {v}" for k, v in sinais.items())

SISTEMA = f"""Você é especialista em segurança de e-mail e explica suas análises para pessoas leigas.
Analise o e-mail e decida se é phishing ou legítimo, mostrando as evidências do próprio e-mail.

Responda só com JSON, com os campos nesta ordem:
1. "indicadores": de 1 a 6 sinais, cada um com "tipo" (um nome das listas abaixo) e "trecho" (cópia literal de 3 a 300 caracteres de uma parte do e-mail, incluindo as linhas "De:" e "Assunto:").
2. "classificacao": "phishing" ou "legitimo".
3. "explicacao": de 2 a 4 frases curtas, em português do Brasil, para um leigo: por que o e-mail é ou não é golpe e o que a pessoa deve fazer.

Sinais de phishing:
{lista(PHISHING)}

Sinais de legitimidade:
{lista(LEGITIMIDADE)}

Regras: pelo menos um indicador do lado da classificação; cite só o que está escrito (ausências vão na explicação); nunca cite um trecho que seja só [EMAIL] ou [LINK]."""


def mensagens(email: str) -> list[dict]:
    """Conversa do aluno para um e-mail já formatado com formatar_email."""
    return [{"role": "system", "content": SISTEMA}, {"role": "user", "content": f"<email>\n{email}\n</email>"}]
