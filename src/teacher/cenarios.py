# Matriz de cenários dos e-mails sintéticos (3.1). Cada sintético sorteia rótulo, cenário e eixos de variação.
# Os eixos são sorteados sem olhar o rótulo, então erros de escrita, links e marcadores aparecem na mesma
# proporção nas duas classes e não viram atalho para o aluno.

import random

from src.config import SEMENTE

PHISHING = {
    "pix_devolucao": "pedido de devolução de um Pix supostamente recebido por engano",
    "boleto_falso": "boleto falso de uma conta, compra ou serviço",
    "cobranca_divida": "cobrança de dívida com ameaça de negativação ou processo",
    "receita_federal": "pendência na Receita Federal ou CPF irregular",
    "govbr": "atualização obrigatória de cadastro no gov.br",
    "inss": "benefício do INSS bloqueado ou valor a receber",
    "correios_taxa": "encomenda retida nos Correios com taxa alfandegária a pagar",
    "bloqueio_conta": "bloqueio de conta bancária ou cartão por suposta atividade suspeita",
    "token": "atualização de token, chave de segurança ou dispositivo do banco",
    "detran_multa": "multa do Detran, pontos na CNH ou licenciamento atrasado",
    "serasa": "nome negativado no Serasa ou acordo de dívida com grande desconto",
    "falso_chefe": "chefe ou diretor pedindo transferência urgente ou compra de cartões-presente",
    "vaga_emprego": "vaga de emprego que pede dados pessoais ou uma taxa de inscrição",
    "premio_sorteio": "prêmio, sorteio ou promoção imperdível que exige ação imediata",
}
LEGITIMO_PARECIDO = {  # legítimos que um leigo poderia confundir com golpe
    "aviso_compra_cartao": "aviso de compra no cartão, sem link e sem pedir dados, orientando a usar o app oficial",
    "nota_fiscal_pedido": "nota fiscal ou confirmação de um pedido que a pessoa fez",
    "alerta_login": "alerta de novo login feito pela própria pessoa, orientando a usar o app ou site oficial",
    "rh_intranet": "comunicado do RH com link para a intranet da empresa",
    "newsletter_loja": "newsletter de uma loja onde a pessoa é cliente",
    "revisao_documento": "colega de trabalho pedindo revisão de um documento",
    "boleto_mensal": "boleto mensal de um serviço contratado (internet, escola, condomínio)",
}
LEGITIMO_COMUM = {
    "reuniao": "convite ou mudança de horário de reunião",
    "consulta_medica": "confirmação de consulta ou exame",
    "suporte_chamado": "resposta do suporte a um chamado aberto pela pessoa",
    "condominio": "comunicado do condomínio aos moradores",
    "escola_faculdade": "aviso da escola ou faculdade sobre calendário ou matrícula",
    "conversa_pessoal": "mensagem de um amigo ou parente",
    "viagem": "confirmação de reserva de hotel ou passagem comprada pela pessoa",
}
PARECIDOS = 0.4  # fração dos legítimos que imitam situações de golpe

REMETENTE = {
    "phishing": ["domínio parecido com o oficial", "conta gratuita genérica (gmail, hotmail, outlook)",
                 "nome de exibição oficial com endereço de outro domínio"],
    "legitimo": ["domínio oficial de quem envia", "conta gratuita pessoal (gmail, hotmail, outlook)"],
}
EIXOS = {
    "tom": ["formal", "informal"],
    "qualidade": ["texto impecável", "texto com alguns erros de escrita"],
    "tamanho": ["curto (3 a 5 frases)", "médio (2 ou 3 parágrafos)", "longo (4 a 6 parágrafos)"],
    "link_anexo": ["sem link nem anexo", "com um link", "com menção a um anexo"],
    "marcadores": ["nenhum", "nenhum", "nenhum", "[LINK]", "[EMAIL]", "[LINK] e [EMAIL]"],  # ≈ 50% com marcador
}


def sortear(i: int) -> dict:
    """Parâmetros do sintético número i (sempre os mesmos para o mesmo i). Pares são phishing, ímpares legítimos."""
    rng = random.Random(SEMENTE * 1_000_003 + i)
    rotulo = "legitimo" if i % 2 else "phishing"
    parecido = rotulo == "legitimo" and rng.random() < PARECIDOS
    cenarios = PHISHING if rotulo == "phishing" else LEGITIMO_PARECIDO if parecido else LEGITIMO_COMUM
    cenario = rng.choice(sorted(cenarios))
    eixos = {k: rng.choice(v) for k, v in EIXOS.items()}
    if "LINK" in eixos["marcadores"]:
        eixos["link_anexo"] = "com um link"
    return {"rotulo": rotulo, "cenario": cenario, "descricao": cenarios[cenario], "parecido": parecido,
            "tipo_remetente": rng.choice(REMETENTE[rotulo])} | eixos
