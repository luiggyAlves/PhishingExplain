# Previsões de um modelo servido por API compatível com OpenAI (vLLM na Spark, llama.cpp no PC) para o test.jsonl.
# Sem saída estruturada e com decodificação gulosa, para medir de forma honesta a taxa de JSON válido.
# Uso: python -m src.prever --nome qwen35-4b_zeroshot --modelo Qwen/Qwen3.5-4B --url http://localhost:8000/v1
#      (com adaptador LoRA no vLLM, --modelo é o nome dado em --lora-modules)

import argparse
import json
import time

from openai import OpenAI

from src.config import CFG, DADOS_PROCESSED, SAIDAS
from src.teacher.lote import rodar

PASTA = SAIDAS / "previsoes"


def preditor(cliente: OpenAI, modelo: str, max_tokens: int):
    def prever(e: dict) -> dict:
        inicio = time.perf_counter()
        r = cliente.chat.completions.create(
            model=modelo, messages=e["messages"], max_tokens=max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return {"texto": r.choices[0].message.content or "", "segundos": round(time.perf_counter() - inicio, 2),
                "tokens_entrada": r.usage.prompt_tokens, "tokens_saida": r.usage.completion_tokens,
                "cortada": r.choices[0].finish_reason == "length"}
    return prever


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True, help="nome do arquivo de saída em outputs/previsoes/")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--url", default=CFG.professor_url)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    itens = [{"id": l["id"], "messages": l["messages"]} for l in map(json.loads, (DADOS_PROCESSED / "test.jsonl").open(encoding="utf-8"))]
    PASTA.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    rodar(itens, preditor(OpenAI(base_url=args.url, api_key="sem-chave", timeout=600), args.modelo, args.max_tokens),
          PASTA / f"{args.nome}.jsonl", PASTA / f"{args.nome}_falhas.jsonl")
    print(f"{args.nome}: {time.perf_counter() - inicio:.0f} s no total")
