# Esse é o cliente do professor, por meio do qual todas as outras partes do sistema: anotação dos emails reais, geração de dados sintéticos etc, irão recorrer para se comunicar

import json 
import time 
from dataclasses import dataclass 
from openai import OpenAI
from src.config import CFG 

# Delimitando os parâmetros do modelo professor, para o modo sem raciocínio 

AMOSTRAGEM = {
    "temperature": 0.7, 
    "top_p": 0.8, 
    "presence_penalty": 1.5, 
    "extra_body":{
        "top_k":20,
        "min_p":0.0,
        "chat_template_kwargs": {"enable_thinking": False}
    },
}

cliente = OpenAI(
    base_url = CFG.professor_url,
    api_key="sem-chave",
    timeout=300,
    max_retries=2,
)

@dataclass
class Resposta:
    texto: str
    tokens_entrada: int 
    tokens_saida: int 
    segundos: float 
    cortada: bool 

def perguntar(mensagens: list[dict], max_tokens=1024) -> Resposta:
    """Manda uma conversa pro modelo professor"""
    inicio = time.perf_counter()
    r = cliente.chat.completions.create(
        model = CFG.professor.modelo,
        messages = mensagens, 
        max_tokens = max_tokens,
        **AMOSTRAGEM,
    )
    duracao = time.perf_counter() - inicio
    escolha = r.choices[0]

    return Resposta(
        texto=escolha.message.content or "",
        tokens_entrada=r.usage.prompt_tokens,
        tokens_saida=r.usage.completion_tokens,
        segundos=duracao, 
        cortada=escolha.finish_reason=="length",

    )

if __name__ == "main":
    print(f"Perfil {CFG.nome}: {CFG.professor_modelo} em {CFG.professor_url}\n")

    # Testando com uma pergunta simples em português
    r = perguntar(
        [
            {"role":"system", "content": "Responda em português do Brasil, em uma frase."},
            {"role":"user", "content":"O que é phishing?"},
        ],
        max_tokens=200,
    )
    print("Teste 1:", r.texto)
    print(
        f"{r.tokens_entrada} tokens de entrada, {r.tokens_saida} de saída"
        f"{r.segundos:.1f} s, {r.tokens_saida/r.segundos} tokens/s"
    )
    if "<think>" in r.texto:
        print("O MODO DE RACIOCINIO NÃO FOI DESLIGADO")

    # Teste 2 - Resposta estruturada em JSON

    email = (
        "De: seguranca@bancoexemplo-atualizacao.com\n"
        "Assunto: Sua conta será bloqueada\n\n"
        "Prezado cliente, detectamos um acesso suspeito. Confirme seus dados em"
        "http://bancoexemplo-atualizacao.com/login em até 24 horas"
    )

    r = perguntar(
        [
            {"role":"system",
            "content": (
                "Você analisa e-mails. Responda APENAS com um objeto JSON, "
                "sem nenhum texto antes ou depois, no formato"
                '{"phishing": true ou false, "motivo": "uma frase"}.'
            )},
            {"role": "user", "content": email},
        ],
        max_tokens=300,
    )
    print("\n Teste 2 - texto bruto:")
    print(r.texto)
    if r.cortada:
        print("A resposta foi cortada pelo máximo de tokens")
    try: 
        dados = json.loads(r.texto)
        print(" JSON válido:", dados)
    except json.JSONDecodeError as erro:
        print("JSON inválido", erro)
