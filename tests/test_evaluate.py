import json

import pandas as pd

from src.evaluate import classificacao, f1_tipos, ler_previsao

EMAIL = "De: a@b.com\nAssunto: Conta bloqueada\n\nClique aqui hoje."
RESP = {"indicadores": [{"tipo": "ameaca", "trecho": "Conta bloqueada"}, {"tipo": "urgencia", "trecho": "inventado"}],
        "classificacao": "phishing", "explicacao": "A conta não está bloqueada. O banco não pede isso por e-mail."}


def test_ler_previsao_com_cerca_e_think():
    p = ler_previsao({"texto": "<think>x</think>```json\n" + json.dumps(RESP) + "\n```"}, EMAIL)
    assert p["previsto"] == "phishing" and p["json_valido"] and p["ancorados"] == 1 and p["trechos"] == 2


def test_ler_previsao_lixo():
    p = ler_previsao({"texto": "não sei"}, EMAIL)
    assert p["previsto"] is None and not p["json_valido"] and p["trechos"] == 0


def test_classificacao_conta_invalida_como_erro():
    df = pd.DataFrame({"rotulo": ["phishing", "phishing", "legitimo", "legitimo"],
                       "previsto": ["phishing", None, "legitimo", "phishing"]})
    m = classificacao(df)
    assert m["recall"] == 0.5 and m["precisao"] == 0.5 and m["taxa_falso_positivo"] == 0.5


def test_f1_tipos():
    assert f1_tipos({"a", "b"}, {"b", "c"}) == 0.5


def test_comparar(tmp_path, monkeypatch):
    import src.evaluate as ev
    dados = pd.DataFrame({"id": [1, 2, 3, 4], "recorte": "x", "rotulo": ["phishing"] * 4})
    monkeypatch.setattr(ev, "carregar", lambda a: dados.assign(previsto=["phishing"] * 4 if "a" in str(a) else [None] * 4))
    r = ev.comparar("a.jsonl", "b.jsonl")
    assert r["so_a_acerta"] == 4 and r["so_b_acerta"] == 0 and r["p_mcnemar"] == 0.125
