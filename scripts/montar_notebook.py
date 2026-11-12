# Gera notebooks/spark.ipynb (rode da raiz: python scripts/montar_notebook.py). Editar aqui e gerar de novo
# é mais fácil de revisar no git do que editar o JSON do notebook.
import json
from pathlib import Path

CELULAS = []
md = lambda t: CELULAS.append(("markdown", t.strip("\n")))
py = lambda t: CELULAS.append(("code", t.strip("\n")))

md("""é o spark
# PhishExplain na DGX Spark: rodada completa

Este notebook roda **no host da DGX Spark** (por exemplo, no JupyterLab do DGX Dashboard) e faz tudo sozinho:

1. clona o repositório (ou atualiza, se já existir), constrói as imagens Docker e reconstrói os dados (Fase 1);
2. sobe o professor (`Qwen/Qwen3.5-27B`) no vLLM, mede a vazão e estima o tempo total;
3. traduz, gera os sintéticos, filtra e anota (7.1);
4. separa treino e validação, exporta as conversas e checa vazamento (Fase 4);
5. roda o professor, o baseline e os três alunos em zero-shot no teste (7.2);
6. treina os três alunos com LoRA e roda as previsões com os adaptadores (7.3 a 7.5);
7. julga as explicações com um modelo de outra família (8.2);
8. mostra todas as tabelas de resultados (8.1 e 8.3).

**Basta `Run All`.** Precisa só de Docker com GPU (seu usuário no grupo `docker`), internet e cerca de 250 GB livres
(imagens + modelos). Nenhum modelo exige aceite de licença; um token do Hugging Face é opcional (baixa mais rápido).
Cada LLM sobe num container do vLLM e é desligado antes do próximo; cada passo do projeto roda na imagem
`phishexplain-unsloth:0.1`. Tudo é idempotente: se algo cair, rode de novo e o que já foi feito é pulado.
""")

md("## Configuração (o único lugar para editar)")

py("""
from pathlib import Path

REPO_URL = "https://github.com/luiggyAlves/PhishingExplain.git"
REPO = Path.home() / "phishexplain"            # onde o repositório fica na Spark
HF_TOKEN = ""                                  # opcional; vazio = usa ~/.cache/huggingface/token, se existir
VLLM_IMAGEM = "nvcr.io/nvidia/vllm:26.09-py3"  # precisa de vLLM >= 0.17; alternativa: "vllm/vllm-openai:cu130-nightly"
PROFESSOR = "Qwen/Qwen3.5-27B"                 # plano B, se a estimativa passar de uma noite: "Qwen/Qwen3.5-35B-A3B"
ALUNOS = ["gemma4-e2b", "qwen35-4b", "s5-nano"]  # ordem do treino: o menor primeiro, para achar bugs cedo
""")

md("## Preparo: repositório, Docker e imagens")

py("""
import csv, os, platform, shutil, subprocess, sys, time
from IPython.display import Image, Markdown, display

assert platform.machine() == "aarch64", "este notebook é para a DGX Spark"
assert shutil.which("docker"), "Docker não encontrado"
r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
assert r.returncode == 0, "sem acesso ao Docker: rode 'sudo usermod -aG docker $USER' e entre de novo\\n" + r.stderr

if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
else:
    r = subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], capture_output=True, text=True)
    print(r.stdout or r.stderr)

token = Path.home() / ".cache" / "huggingface" / "token"
os.environ["HF_TOKEN"] = HF_TOKEN or (token.read_text().strip() if token.exists() else "")
os.environ["PHISH_PERFIL"] = "spark"
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from src import servidor                     # só biblioteca padrão: roda no host
from src.config import CFG, JUIZ_MODELO      # idem
from src.servidor import executar, modulo, python, subir, parar
servidor.VLLM_IMAGEM = VLLM_IMAGEM

donos = {p.stat().st_uid for p in servidor.HF.glob("hub/*")}
if donos - {os.getuid()}:
    print(f"AVISO: há pastas em {servidor.HF} de outro usuário (root?). Rode: sudo chown -R $USER:$USER {servidor.HF}")
print("Repositório:", REPO, "| commit:", subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
""")

py("""
executar(["./scripts/spark_construir.sh"])   # na primeira vez compila triton e xformers (demora); depois usa o cache
executar(["docker", "pull", VLLM_IMAGEM])
modulo("src.ambiente")                       # 6.2: aarch64, CUDA 13, GB10, cc 12.1, bf16, transformers >= 5.2
""")

md("""
## Fase 1: dados

Os dados não vão para o git (a base PT-BR não tem licença de redistribuição): são baixados das fontes originais e
reconstruídos aqui. O processo é determinístico, e o teste congelado (`splits/teste.csv`, no git) garante os mesmos
399 e-mails de teste; se alguma fonte mudar, `separar_teste` mostra quantos ids continuam na tabela.
""")

py("""
if (REPO / "data" / "interim" / "emails_agrupados.parquet").exists():
    print("Tabela mestre já existe; pulando a Fase 1")
else:
    for passo in ("baixar", "normalizar", "limpar", "agrupar", "separar_teste"):
        modulo(f"src.data.{passo}")
""")

md("## Professor no vLLM e estimativa de tempo")

py("""
url = subir(PROFESSOR, servidor.PROFESSOR, nome=CFG.professor_modelo)  # o nome servido é sempre o do config
modulo("src.teacher.cliente")
""")

md("""
Traduz uma amostra de 64 e-mails (que já conta para a rodada) e estima o tempo de toda a geração do professor.
Se passar de uma noite, troque `PROFESSOR` pelo plano B na configuração e rode de novo a partir da célula do professor.
""")

py('''
inicio = time.time()
modulo("src.teacher.traduzir", PHISH_LIMITE=64)
python("-c", """
import os
from src.config import SINTETICOS_META
from src.teacher import anotar, traduzir
t = len(traduzir.selecionar(False)) + len(traduzir.selecionar(True))
a = len(anotar.selecionar(False)) + t + SINTETICOS_META
h = (t + SINTETICOS_META + a) * float(os.environ["POR_EMAIL"]) / 3600
print(f"{t} traduções, {SINTETICOS_META} sintéticos e ~{a} anotações: ~{h:.1f} h de professor")
print("Mais de uma noite: considere o plano B." if h > 12 else "Cabe numa noite.")
""", POR_EMAIL=(time.time() - inicio) / 64)
''')

md("## 7.1: tradução, sintéticos e anotação")

py("""
modulo("src.teacher.traduzir")              # sementes livres + 1 variante por grupo
modulo("src.teacher.traduzir", "--teste")   # as 299 sementes do recorte traduzido
modulo("src.teacher.gerar")                 # 2.000 sintéticos
modulo("src.teacher.filtrar")               # MinHash contra toda a tabela mestre
modulo("src.teacher.anotar")                # treino: nativos + traduções + sintéticos
modulo("src.teacher.anotar", "--teste")     # referência da avaliação (nunca no treino)
modulo("src.teacher.filtrar")               # de novo, para a revisão trazer as anotações
python("-c", \"\"\"
from src.config import DADOS_INTERIM as D
from src.teacher.lote import ler
for n, r in (("traducoes", "traducoes_reprovadas"), ("sinteticos", "sinteticos_reprovados"),
             ("anotacoes", "anotacoes_reprovadas"), ("anotacoes_teste", "anotacoes_teste_reprovadas")):
    ok, ruim = len(ler(D / f"{n}.jsonl")), len(ler(D / f"{r}.jsonl"))
    print(f"{n:16} {ok:6} aprovados {ruim:5} reprovados  ({ok / max(1, ok + ruim):.0%})")
\"\"\")
""")

md("**Revisão manual (pode ser depois):** `data/interim/revisao_sinteticos.md` traz 50 sintéticos com as anotações.")

md("## Fase 4: splits, exportação, professor no teste e vazamento")

py("""
modulo("src.splits")
modulo("src.exportar")
modulo("src.prever", "--nome", "professor", "--modelo", CFG.professor_modelo, "--url", url)
parar()
modulo("src.vazamento")
with open(REPO / "reports" / "vazamento.csv", encoding="utf-8") as f:
    for linha in list(csv.DictReader(f))[:10]:  # os pares mais parecidos; revise os de similaridade alta
        print(linha["similaridade"], linha["recorte"], linha["id_teste"], "<->", linha["id_treino"])
""")

md("## Baseline e alunos em zero-shot")

py("""
modulo("src.baseline")
for aluno in ALUNOS:
    repo = CFG.alunos[aluno]
    url = subir(repo, servidor.ALUNO)
    modulo("src.prever", "--nome", f"{aluno}_zeroshot", "--modelo", repo, "--url", url)
parar()
""")

md("## Treino LoRA dos três alunos (com o vLLM desligado: os dois disputam a mesma memória)")

py('''
parar()
for aluno in ALUNOS:
    if (REPO / "outputs" / aluno / "adapter_config.json").exists():
        print(aluno, "já treinado")
    else:
        modulo("src.train", "--aluno", aluno)

python("-c", """
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
for aluno in %r:
    h = pd.DataFrame(json.load(open(f"outputs/{aluno}/historico.json")))
    ax = h.dropna(subset=["loss"]).plot(x="step", y="loss", title=aluno, figsize=(6, 3))
    if "eval_loss" in h:
        h.dropna(subset=["eval_loss"]).plot(x="step", y="eval_loss", ax=ax, marker="o")
    plt.tight_layout(); plt.savefig(f"outputs/{aluno}/loss.png"); plt.close()
""" % ALUNOS)
for aluno in ALUNOS:
    display(Image(str(REPO / "outputs" / aluno / "loss.png")))
''')

md("## Previsões com os adaptadores")

py("""
for aluno in ALUNOS:
    repo = CFG.alunos[aluno]
    url = subir(repo, servidor.ALUNO, lora={f"{aluno}-lora": f"outputs/{aluno}"})
    modulo("src.prever", "--nome", f"{aluno}_lora", "--modelo", f"{aluno}-lora", "--url", url)
parar()
""")

md("""
## 8.2: juiz das explicações

Mistral Small 3.2 (família diferente de Qwen e Gemma) dá notas de 1 a 5 sem saber qual modelo escreveu.
Depois, dê suas notas aos 30 casos de `reports/calibracao_juiz.md` para calibrar o juiz.
""")

py("""
previsoes = sorted(f"outputs/previsoes/{p.name}" for p in (REPO / "outputs" / "previsoes").glob("*.jsonl")
                   if not p.stem.endswith("_falhas") and p.stem != "baseline_tfidf")
url = subir(JUIZ_MODELO, servidor.JUIZ)
modulo("src.juiz", "--url", url, *previsoes)
parar()
""")

md("## 8.1 e 8.3: resultados")

py("""
modulo("src.relatorio")
display(Markdown((REPO / "reports" / "resultados.md").read_text(encoding="utf-8")))
""")

md("""
## Depois da rodada

- Commit de `splits/treino.csv`, `splits/validacao.csv` e `reports/` (as tabelas também estão em CSV).
- Revisões manuais: `data/interim/revisao_sinteticos.md`, topo de `reports/vazamento.csv`, `reports/calibracao_juiz.md`.
- Preencha a tabela de resultados e as limitações do `README.md` a partir de `reports/resultados.md`.
""")


def celula(tipo, texto):
    base = {"cell_type": tipo, "metadata": {}, "source": texto.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


if __name__ == "__main__":
    nb = {"cells": [celula(*c) for c in CELULAS], "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    destino = Path(__file__).resolve().parent.parent / "notebooks" / "spark.ipynb"
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{destino}: {len(CELULAS)} células")
