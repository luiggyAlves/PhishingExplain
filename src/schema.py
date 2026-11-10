# Aqui vamos definir o formato de saída do professor, o JSON schema que todo mundo vai usar, o objetivo aqui é forçar todo mundo, inclusive o professor, a escrever dentro do formato, que já saberemos então será mais fácil de interpretar

"""Contrato de saída: fonte única do JSON que professor e alunos produzem."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Aqui são os dicionários que o professor irá usar tanto para avaliar se um possível email é phishing ou não. Ele vai "avaliar" em termos de phishing os sinais como urgencia e ameaça e avaliar a legitimidade em termos de dominio oficial, informações confiaveis etc. É importante que isso aqui esteja bem delimitado para que possamos comparar os modelos e para que o LLM não halucine em causas inexistentes de phishing, que ele mesmo inventou

PHISHING = {
    "urgencia": "prazo curto ou pressa para agir",
    "ameaca": "bloqueio, multa, suspensão ou outra punição",
    "personificacao": "usa o nome de banco, órgão, empresa ou chefe sem sinal de autenticidade",
    "remetente_inconsistente": "o domínio do remetente não bate com quem diz ser",
    "link_suspeito": "domínio parecido com o oficial, encurtador ou texto do link diferente do destino",
    "pedido_de_dados": "pede senha, token, CPF, cartão ou código de verificação",
    "pagamento_incomum": "Pix, boleto ou transferência inesperados",
    "anexo_suspeito": "anexo executável, compactado ou não esperado",
    "oferta_irreal": "prêmio, sorteio ou dinheiro fácil",
    "saudacao_generica": "trata a pessoa de forma genérica, como 'Prezado cliente'",
    "erros_de_linguagem": "erros de escrita ou tradução estranha",
}
LEGITIMIDADE = {
    "dominio_oficial": "remetente ou link no domínio oficial de quem envia",
    "contexto_esperado": "refere-se a algo que a pessoa fez ou contratou, como um pedido",
    "personalizacao": "traz dados específicos da pessoa, como nome ou final do cartão",
    "orientacao_segura": "orienta a usar canais oficiais e não pede dados",
}

# Serve para atrelar a uma parte do email o indicador de qual criterio dos dicionários estão sendo considerados
class Indicador(BaseModel):
    tipo: Literal[tuple(PHISHING | LEGITIMIDADE)]
    trecho: str = Field(min_length=3, max_length=300)


class Resposta(BaseModel):
    # O modelo vai retornar como resposta os indicadores (de 1 a 6), a classificação e a explicação daquela classificação
    indicadores: list[Indicador] = Field(min_length=1, max_length=6)
    classificacao: Literal["phishing", "legitimo"]
    explicacao: str = Field(min_length=40, max_length=800)

    @field_validator("explicacao") # faz a função rodar sempre que explicacao for preenchida, já que é um validador
    @classmethod
    def frases(cls, v: str) -> str:
        n = len(re.split(r"(?<=[.!?])\s+", v.strip()))
        if not 2 <= n <= 4:
            raise ValueError(f"a explicação deve ter de 2 a 4 frases, tem {n}")
        return v

    @model_validator(mode="after")
    # Confere se o modelo tem pelo menos 1 indicador do lado da sua classificação, os indicadores devem estar submissos ao contexto, pois um email legitimo pode ter urgencia mas no contexto não ser phishing
    def lado_certo(self):
        lados = {i.tipo in LEGITIMIDADE for i in self.indicadores}
        if (self.classificacao == "legitimo") not in lados:
            raise ValueError(f"'{self.classificacao}' precisa de ao menos um sinal do seu lado")
        return self


def formatar_email(remetente, assunto, corpo) -> str:
    """O e-mail como o modelo o vê; os trechos citados vêm deste texto."""
    cab = [f"{r}: {v}" for r, v in (("De", remetente), ("Assunto", assunto)) if isinstance(v, str) and v]
    return "\n".join(cab) + "\n\n" + corpo if cab else corpo