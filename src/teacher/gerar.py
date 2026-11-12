# Geração de e-mails sintéticos pelo professor (3.2), a partir dos cenários sorteados em cenarios.py.
# Nenhum e-mail real entra no prompt, então não há como copiar o teste. Depois vem filtrar.py (3.3) e a anotação.

from pydantic import BaseModel

from src.config import CFG, DADOS_INTERIM, SINTETICOS_META
from src.data.limpar import MIN_CORPO, limpar_texto
from src.teacher.cenarios import sortear
from src.teacher.cliente import perguntar
from src.teacher.lote import rodar

GERADOS = DADOS_INTERIM / "sinteticos.jsonl"
REPROVADOS = DADOS_INTERIM / "sinteticos_reprovados.jsonl"

SISTEMA = """Você escreve e-mails realistas em português do Brasil para um conjunto de dados de pesquisa que treina detectores de phishing.
Você recebe o tipo de e-mail e suas características e responde só com um JSON com "remetente", "assunto" e "corpo".

Regras:
- "remetente" no formato Nome <endereco@dominio>. Use empresas, órgãos e marcas brasileiras conhecidas quando o cenário pedir, mas invente pessoas, números de documentos, valores e domínios.
- Escreva como um e-mail de verdade: nada de títulos, comentários, avisos de que é um exemplo nem a palavra "phishing".
- Se for um golpe, ele deve ser convincente; se for legítimo, ele deve ser coerente com uma empresa ou pessoa honesta.
- Quando pedido, use o marcador [LINK] no lugar do endereço de um link e [EMAIL] no lugar de um endereço de e-mail citado no corpo. Sem pedido, não use marcadores."""


class Email(BaseModel):
    remetente: str
    assunto: str
    corpo: str


def pedido(p: dict) -> str:
    tipo = "golpe (phishing)" if p["rotulo"] == "phishing" else "e-mail legítimo"
    marcadores = "não use marcadores" if p["marcadores"] == "nenhum" else f"use {p['marcadores']}"
    return (f"Escreva um {tipo}: {p['descricao']}.\n- Remetente: {p['tipo_remetente']}\n- Tom: {p['tom']}\n"
            f"- Qualidade: {p['qualidade']}\n- Tamanho: {p['tamanho']}\n- Link ou anexo: {p['link_anexo']}\n- Marcadores: {marcadores}")


def gerar(p: dict) -> dict:
    r = perguntar([{"role": "system", "content": SISTEMA}, {"role": "user", "content": pedido(p)}], max_tokens=2048, esquema=Email)
    try:
        e = Email.model_validate_json(r.texto)
    except ValueError as erro:
        return {"erros": [f"JSON inválido: {erro}"], "texto": r.texto}
    campos = {k: limpar_texto(v) for k, v in e.model_dump().items()}
    corpo, pedidos = campos["corpo"] or "", {m for m in ("[LINK]", "[EMAIL]") if m in p["marcadores"]}
    erros = (["resposta cortada"] * r.cortada + ["corpo curto demais"] * (len(corpo) < MIN_CORPO)
             + [f"faltou {m}" for m in pedidos if m not in corpo]
             + [f"marcador não pedido {m}" for m in {"[LINK]", "[EMAIL]"} - pedidos if m in corpo])
    return p | campos | {"erros": erros, "segundos": round(r.segundos, 1)}


if __name__ == "__main__":
    n = CFG.limite_exemplos or SINTETICOS_META
    rodar([{"id": f"sintetico:{i}"} | sortear(i) for i in range(n)], gerar, GERADOS, REPROVADOS)
