# responsável por gerenciar todos os caminhos que vao mudar do PC pra dgx spark, assim a gente não precisa ficar mudando tudo a mão

import os
from dataclasses import dataclass
from pathlib import Path

# Aqui a gente pega, partindo da raiz, o caminho relativo de cada pasta
RAIZ = Path(__file__).resolve().parent.parent  # pega a raiz do projeto
DADOS_RAW = RAIZ / "data" / "raw"
DADOS_INTERIM = RAIZ / "data" / "interim"
DADOS_PROCESSED = RAIZ / "data" / "processed"
MODELOS_LOCAIS = RAIZ / "models"
SAIDAS = RAIZ / "outputs"
RELATORIOS = RAIZ / "reports"

MAX_CHARS_EMAIL = 6000  # corpo enviado a professor, alunos e avaliação (~1.500 tokens); cortado no último parágrafo antes disso

SEMENTE = 42 # permite que tudo que é "aleatório" como split de dados etc, tem que ser igual nos dois lado para permitir reprodutibilidade

@dataclass(frozen=True) # cria uma classe, um tipo de dados sem construtor e o frozen torna constante, n pode mudar nada
class Perfil:
    nome: str 
    professor_url: str 
    professor_modelo: str 
    professor_paralelo: int 
    limite_exemplos: int | None 
    alunos: dict[str, str]

# A gente faz um perfil de teste, bem fraco no PC e roda o real na spark, o do PC serve pra ver se a lógica tá certa e o que efetivamente é executado roda na spark, perfil basicamente vai servir pra na hora do comando a gente escolher o que rodar
PERFIS = {
    "pc": Perfil(
        nome="pc",
        professor_url="http://localhost:8080/v1",
        professor_modelo="professor-dev",
        professor_paralelo=2,
        limite_exemplos=20,
        alunos={
            "qwen35-08b":"Qwen/Qwen3.5-0.8B"
        },
    ),
    "spark":Perfil(
        nome="spark",
        professor_url="http://localhost:8000/v1",
        professor_modelo="Qwen/Qwen3.5-27B",
        professor_paralelo = 32,
        limite_exemplos=None, 
        alunos={
            "gemma4-e2b": "google/gemma-4-E2B-it",
            "qwen35-4b": "Qwen/Qwen3.5-4B",
            "s5-nano": "maritaca-ai/S5-nano",
        },
    ),
}

NOME_PERFIL = os.environ.get("PHISH_PERFIL", "pc")
if NOME_PERFIL not in PERFIS: # evita erro de digitação no comando
    raise ValueError(
        f"PHISH_PERFIL={NOME_PERFIL!r} não existe, use um de : {list(PERFIS)}"
    )
CFG=PERFIS[NOME_PERFIL]

if __name__ == "__main__":
    print(f"Perfil: {CFG.nome}")
    print(f"Raiz do projeto: {RAIZ}")
    print(f"Professor: {CFG.professor_modelo} em {CFG.professor_url}")
    print(f"Requisições em paralelo: {CFG.professor_paralelo}")
    print(f"Limite de exemplos: {CFG.limite_exemplos or 'sem limite'}")
    print("Alunos:")
    for apelido, repo in CFG.alunos.items():
        print(f"  {apelido}: {repo}")
    for pasta in (DADOS_RAW, DADOS_INTERIM, DADOS_PROCESSED, MODELOS_LOCAIS, SAIDAS, RELATORIOS):
        print(f"  {'ok  ' if pasta.exists() else 'FALTA'} {pasta.relative_to(RAIZ)}")