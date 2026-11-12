import json

import pytest

from src.teacher.validar import validar

EMAIL = "De: suporte@banco-seguro.xyz\nAssunto: Conta   BLOQUEADA\n\nClique em [LINK] para liberar sua conta hoje."
OK = {
    "indicadores": [{"tipo": "ameaca", "trecho": "conta bloqueada"}],
    "classificacao": "phishing",
    "explicacao": "O remetente não é do banco. A ameaça de bloqueio serve para apressar você.",
}


def erros(mudanca=None, rotulo="phishing"):
    return validar(json.dumps(OK | (mudanca or {})), EMAIL, rotulo)


def test_aprovada():
    assert erros() == []


def test_trecho_com_espacos_e_maiusculas_diferentes():
    assert erros({"indicadores": [{"tipo": "link_suspeito", "trecho": "CLIQUE em  [LINK] para\nliberar"}]}) == []


@pytest.mark.parametrize("mudanca, rotulo, inicio", [
    ({"explicacao": "Curta."}, "phishing", "contrato"),
    (None, "legitimo", "classificação"),
    ({"indicadores": [{"tipo": "ameaca", "trecho": "sua conta será excluída"}]}, "phishing", "trecho fora"),
    ({"indicadores": [{"tipo": "link_suspeito", "trecho": "[LINK]"}]}, "phishing", "trecho só"),
    ({"explicacao": "A classificação informada é phishing. O link não é do banco."}, "phishing", "explicação"),
])
def test_erros(mudanca, rotulo, inicio):
    assert [e for e in erros(mudanca, rotulo) if e.startswith(inicio)]


def test_json_quebrado():
    assert validar("{não é json", EMAIL, "phishing")[0].startswith("contrato")
