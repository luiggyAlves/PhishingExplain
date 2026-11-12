import pytest
from pydantic import ValidationError

from src.schema import Resposta, cortar, formatar_email

OK = {
    "indicadores": [{"tipo": "urgencia", "trecho": "sua conta será bloqueada em 24 horas"}],
    "classificacao": "phishing",
    "explicacao": "O link não é do banco. A ameaça de bloqueio serve para apressar você.",
}


def test_valida():
    Resposta.model_validate(OK)


@pytest.mark.parametrize("mudanca", [
    {"classificacao": "spam"},
    {"classificacao": "legitimo"},
    {"indicadores": []},
    {"indicadores": [{"tipo": "inventado", "trecho": "qualquer coisa"}]},
    {"indicadores": [{"tipo": "urgencia", "trecho": "ok"}]},
    {"explicacao": "Uma frase só, longa o bastante para passar do mínimo."},
])
def test_invalidas(mudanca):
    with pytest.raises(ValidationError):
        Resposta.model_validate(OK | mudanca)


def test_formatar_email():
    assert formatar_email(None, float("nan"), "Olá") == "Olá"
    assert formatar_email("a@b.com", "Oi", "Olá") == "De: a@b.com\nAssunto: Oi\n\nOlá"


def test_cortar():
    texto = "a" * 40 + "\n\n" + "b" * 40 + "\n\n" + "c" * 40
    assert cortar(texto, 1000) == texto
    assert cortar(texto, 90) == "a" * 40 + "\n\n" + "b" * 40
    assert cortar("x" * 100, 50) == "x" * 50
