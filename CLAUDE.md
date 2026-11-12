# PhishExplain: plano de execução para o Claude Code

Este arquivo dá todo o contexto necessário para continuar o projeto. Leia-o inteiro antes de mexer em qualquer coisa. O andamento está na seção **9** e o próximo passo na seção **8**.

## 1. O projeto em uma página

Fine-tuning de três LLMs abertos e pequenos para ler um e-mail em português e responder, em JSON, se é phishing ou legítimo, com os trechos do próprio e-mail que sustentam a decisão e uma explicação curta para leigos. O foco é a **explicação**, não só o rótulo.

**Perguntas de pesquisa**

1. O pós-treino brasileiro da Maritaca (S5-nano) melhora a detecção de phishing em PT-BR em relação ao Qwen3.5-4B do qual ele saiu? Quanto se perde com um modelo de cerca de metade do tamanho (Gemma 4 E2B)?
2. O fine-tuning com LoRA melhora cada modelo de forma significativa em relação ao zero-shot?
3. As explicações citam trechos que existem no e-mail e são úteis para um leigo?

**Modelos**

| Papel | Modelo (Hugging Face) | Observações |
| --- | --- | --- |
| Aluno | `maritaca-ai/S5-nano` | Pós-treino do Qwen3.5-4B, Apache 2.0, modos Think/Instruct |
| Aluno | `Qwen/Qwen3.5-4B` | Grupo de controle do S5-nano, Apache 2.0 |
| Aluno | `google/gemma-4-E2B-it` | ≈ 2,3B efetivos |
| Professor | `Qwen/Qwen3.5-27B` | Apache 2.0, roda local na DGX Spark via vLLM |
| Professor de desenvolvimento | `unsloth/Qwen3.5-4B-GGUF` (UD-Q4_K_XL) | Só no PC, só para testar código; **saídas nunca entram no treino** |

Todos os alunos são multimodais, mas usamos só texto. Treino e inferência sempre no **modo sem raciocínio** (`enable_thinking=False` para Qwen3.5 e S5-nano; template `gemma-4` sem thinking para o Gemma).

**Plano detalhado, para leitura humana:** documento "PhishExplain" no claude.ai (https://claude.ai/artifact/E269Zm8dUDP8gVXo9f9crg). Este `CLAUDE.md` contém tudo o que é preciso para executar; o documento é a referência do usuário.

## 2. Como trabalhar neste repositório

O usuário está aprendendo com o projeto. Siga estas regras:

- **Um passo por vez.** Ao terminar um passo, rode os testes e o comando de verificação, mostre o resultado e explique em português, em poucos parágrafos, o que foi feito e por quê. Ao fim de cada **fase**, pare e espere o usuário revisar antes de seguir (salvo quando ele autorizar seguir direto).
- **Código enxuto.** O usuário pediu o código mais conciso possível a partir da Fase 2. Prefira compreensões, funções pequenas e bibliotecas prontas a classes feitas à mão. Sem abstrações que o passo não exige.
- **Nomes e comentários em português**, como no código existente.
- **Scripts rodam como módulos**, da raiz: `python -m src.teacher.prompts`. Todo script executável termina com `if __name__ == "__main__":`.
- **Nada de caminhos, nomes de modelos ou tamanhos escritos à mão.** Use `src/config.py` (`CFG`, `RAIZ`, `DADOS_*`, `SEMENTE`, `MAX_CHARS_EMAIL`) e `src/dados.py`.
- **Scripts idempotentes:** rodar duas vezes dá o mesmo resultado; o que já foi feito é pulado ou retomado.
- **Commit ao fim de cada passo**, com mensagem em português, direto na `main`, **sem linhas de coautoria do Claude**. Nunca commitar `data/`, `models/`, `outputs/` nem `reports/inventario_raw.txt` (o `.gitignore` ancora esses caminhos na raiz). `data/` fica fora por licença: o repositório das bases PT-BR não tem licença e não pode ser redistribuído; quem clona reconstrói tudo com a Fase 1 (o notebook faz isso).
- **Testes com pytest** em `tests/`, rodados com `python -m pytest -q` no ambiente conda `phishexplain`.

## 3. Ambientes

### PC (desenvolvimento)

- Notebook x86_64, Linux, **RTX 4060 de 8 GB**, driver 580 (CUDA até 13.0).
- Ambiente conda `phishexplain`, Python 3.12 (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate phishexplain`). Dependências em `requirements.txt`; as de treino em `requirements-treino.txt`.
- **Professor de desenvolvimento:** llama.cpp `b12207` (binário CUDA 12.8) em `~/apps/llama.cpp/llama-b12207/`, já no PATH. Sobe com `./scripts/professor_dev.sh`: porta 8080, só `127.0.0.1`, alias `professor-dev`, `--ctx-size 16384 --parallel 2` (**8.192 tokens por requisição**), `--jinja`.
- Perfil `pc` (padrão): 20 exemplos por execução (`PHISH_LIMITE=80` muda isso), 2 requisições em paralelo, aluno de teste `Qwen/Qwen3.5-0.8B`.

### DGX Spark (trabalho pesado)

- Chip GB10, **128 GB de memória unificada**, CPU ARM (aarch64), CUDA 13, compute capability 12.1. O gargalo é a largura de banda da memória: geração token a token é lenta, então tudo que gera texto deve rodar em lote.
- Ambiente: container Docker a partir do `Dockerfile_DGX_Spark` da Unsloth, base `nvcr.io/nvidia/pytorch:25.11-py3`. O Dockerfile trava o torch da NGC; **nunca deixe o pip trocar o torch**.
- Perfil `spark`: sem limite de exemplos, 32 requisições em paralelo, professor `Qwen/Qwen3.5-27B` na porta 8000, alunos completos. Ativado com `-e PHISH_PERFIL=spark` no `docker run`.
- Qwen3.5 e S5-nano exigem **transformers v5** (≥ 5.2) e **vLLM ≥ 0.17**. O Qwen3.5-27B denso gera devagar na Spark; plano B: `Qwen/Qwen3.5-35B-A3B` (MoE, 3B ativos, Apache 2.0).

## 4. Estado atual do repositório

```text
phishexplain/
├── CLAUDE.md
├── README.md
├── requirements.txt, requirements-treino.txt
├── docker/                        # Dockerfile.unsloth (cópia do oficial da Unsloth) e Dockerfile do projeto (6.1)
├── notebooks/spark.ipynb          # Fases 7 e 8 inteiras na Spark, até as tabelas finais
├── scripts/
│   ├── professor_dev.sh           # sobe o professor de desenvolvimento (PC)
│   ├── spark_construir.sh         # 6.1: constrói as imagens na Spark
│   └── spark_container.sh         # 6.2: sobe o container com Jupyter e o socket do Docker
├── splits/teste.csv               # TESTE CONGELADO: ids nunca mudam (só a coluna grupo acompanha a 1.3)
├── src/
│   ├── config.py                  # perfis pc/spark, caminhos, modelos, constantes (SEMENTE, MAX_CHARS_EMAIL, ...)
│   ├── schema.py                  # contrato de saída, cortar(), formatar_email()
│   ├── dados.py                   # fonte única: mestre(), teste(), emails_pt(), amostrar(), caminhos das anotações
│   ├── data/                      # Fase 1: baixar, inspecionar, normalizar, limpar, agrupar, separar_teste
│   ├── teacher/
│   │   ├── cliente.py             # perguntar(mensagens, max_tokens, esquema=, **amostragem) -> Retorno
│   │   ├── prompts.py             # prompt do professor (2.2)
│   │   ├── validar.py             # validadores (2.3)
│   │   ├── lote.py                # paralelismo + JSONL incremental + retomada por id
│   │   ├── traduzir.py            # EN -> PT-BR (2.4)
│   │   ├── anotar.py              # anotação com até 3 tentativas (2.4)
│   │   ├── cenarios.py, gerar.py  # sintéticos (3.1, 3.2)
│   │   └── filtrar.py             # MinHash, marcadores e amostra de revisão (3.3)
│   ├── splits.py, exportar.py, vazamento.py, prompts_aluno.py   # Fase 4
│   ├── evaluate.py, baseline.py, train.py, prever.py             # Fase 5
│   ├── ambiente.py, servidor.py                                  # Fase 6 (checagem e vLLM em container irmão)
│   └── juiz.py, relatorio.py                                     # Fase 8
└── tests/                         # test_schema.py, test_validar.py, test_evaluate.py
```

### Dados (em `data/`, fora do git; o repositório no GitHub é público)

- **`emails_agrupados.parquet` é a tabela mestre.** Leia sempre por `src/dados.py`.
- Colunas: `id`, `fonte`, `grupo`, `autoria` (`humano`/`llm`), `idioma`, `traduzido`, `remetente`, `assunto`, `corpo`, `urls`, `anexos`, `rotulo` (`phishing`/`legitimo`/`spam`), `rotulo_original`, `motivacao`, `grupo_original`.
- Ids no formato `fonte:numero`; traduções `<id>@pt`. Grupos do PhishFuzzer: `pf:<Original_ID>`.
- Arquivos gerados pelo professor guardam só ids e texto; grupo, fonte e rótulo vêm da tabela mestre via `dados.emails_pt()`.

| Fonte | Legítimo | Phishing | Spam | Total | Autoria |
| --- | --- | --- | --- | --- | --- |
| `pf_semente` (PhishFuzzer, e-mails reais) | 1.070 | 1.066 | 953 | 3.089 | humano |
| `pf_variante` (PhishFuzzer, reescritas por LLM) | 6.504 | 6.498 | 6.228 | 19.230 | llm |
| `pt_d` (gerado por 4 LLMs) | 2.184 | 2.184 | 0 | 4.368 | llm |
| `pt_b` (provável LLM, assunto vazio) | 250 | 250 | 0 | 500 | llm |
| `pt_opara` (provável tradução ou LLM) | 63 | 63 | 0 | 126 | llm |
| `pt_rnp` (golpes reais brasileiros) | 0 | 89 | 0 | 89 | humano |
| **Total** | **10.071** | **10.150** | **7.181** | **27.402** | |

Histórico (refeito em 2026-11-12): 28.184 e-mails na entrada; 782 removidos (665 idioma fora de pt/en, 10 corpo curto, 15 corpo longo, 16 assunto ou remetente malformado, 76 duplicatas exatas). Quase-duplicatas: 849 pares, 8.288 → 8.055 grupos, nenhum grupo com rótulos misturados, maior grupo `pf:1616` com 252 e-mails.

### O teste congelado (`splits/teste.csv`)

Colunas `id, grupo, fonte, rotulo, recorte`. 399 e-mails, todos de autoria humana:

| Recorte | Phishing | Legítimo | Origem |
| --- | --- | --- | --- |
| `pt_nativo` | 100 | 0 | 89 do RNP + 11 sementes do PhishFuzzer em português |
| `traduzido` | 150 | 149 | sementes em inglês, sorteadas por rótulo; serão traduzidas |

Os ids são fixos. Ao refazer a 1.3, `python -m src.data.separar_teste` só atualiza a coluna `grupo` e tira ids que saíram da tabela mestre (em 2026-11-11 saiu `pf_semente:3209`, corpo longo demais).

**Regra de exclusão, válida para todas as fases:** qualquer e-mail cujo `grupo` aparece em `splits/teste.csv` fica fora do treino, da validação, dos exemplos de prompt e da geração de sintéticos. São 2.119 e-mails reservados, além dos 399 do teste. O professor pode **traduzir** e-mails do teste e anotá-los como referência para a avaliação, mas nunca usá-los como exemplo nem gerar variações deles.

## 5. Decisões já tomadas (não reabrir sem o usuário)

1. **Tarefa binária:** phishing ou legítimo. O spam fica nos dados intermediários e é filtrado na hora dos splits.
2. **zefang-liu e Kaggle Al-Subaiey fora da v1.** Só inglês e português.
3. **Teste:** golpes reais em português + sementes humanas traduzidas, com métricas sempre **separadas por recorte**.
4. **Professor:** Qwen3.5-27B local. **Não use o Claude para gerar dados de treino** (explicações, traduções ou sintéticos): os termos da Anthropic restringem usar as saídas do Claude como alvo de treino. O Claude Code pode escrever código, revisar amostras e servir de juiz na avaliação.
5. **Contrato de saída** (`src/schema.py`): `indicadores` (1 a 6, `tipo` + `trecho` literal de 3 a 300 caracteres) → `classificacao` → `explicacao` (2 a 4 frases, 40 a 800 caracteres). Pelo menos um indicador do lado da classificação.
6. **`formatar_email(remetente, assunto, corpo)`** define o texto que todo modelo vê, já com o corpo cortado em `MAX_CHARS_EMAIL` (6.000 caracteres, no último parágrafo antes do limite). Os trechos são conferidos contra esse texto.
7. **Anonimização:** `[EMAIL]` e `[LINK]`. Marcadores hoje: RNP 48 de 89 golpes; `pt_d` 101 legítimos e 10 phishing; `pf_variante` 39 (27 spam); `pt_b` 5 phishing. **Mitigação na Fase 3:** os sintéticos usam `[EMAIL]` e `[LINK]` na mesma proporção nas duas classes.
8. **Treino:** LoRA em bf16 para os três alunos, mesmos hiperparâmetros.
9. **Variantes em inglês no treino (2026-11-12):** 1 variante por grupo livre (≈ 1.764), além das 1.795 sementes livres.
10. **Mojibake (2026-11-12):** consertado com `ftfy` na limpeza.

## 6. Armadilhas já encontradas

- **Valores ausentes em colunas de texto do pandas chegam como `NaN` (float), não `None`.** Teste `isinstance(v, str)`. Para JSON, use `traduzir.como_itens()` (troca NaN por None).
- **Espaços:** até 2026-11-11 o `limpar.py` apagava todos os espaços, o que quebrava o MinHash por palavras. Ao mexer na limpeza, imprima um e-mail antes de seguir.
- **`.gitignore`:** padrões sem barra inicial (`data/`) também casam com subpastas como `src/data/`; ancore na raiz com `/`.
- **Unicode:** normalize para NFC antes de comparar textos.
- **`if __name__ == "__main__":`** com dois sublinhados de cada lado.
- **Strings adjacentes em Python são coladas sem espaço.**
- **Contexto do professor no PC:** 8.192 tokens por requisição; o corte do `formatar_email` resolve.
- **`Literal[tuple(...)]` no `schema.py`** pode aparecer como erro no editor; funciona em execução.
- **`pkill -f padrão`** casa com o próprio shell que roda o comando; use `pkill -x llama-server`.
- **Modelos Qwen3.5 pensam por padrão:** toda chamada sem saída do aluno manda `chat_template_kwargs={"enable_thinking": False}` (senão os tokens vão para o raciocínio e a resposta volta vazia).
- **`causal_conv1d` ausente** deixa o Qwen3.5 mais lento no treino (cai na implementação de referência); opcional instalar.
- **Na Spark:** containers como root deixam arquivos com dono root no repositório e no cache do Hugging Face; por isso `servidor.py` usa `--user`. Se o cache já tiver pastas de root, `sudo chown -R $USER:$USER ~/.cache/huggingface`.
- **Saída de subprocessos no Jupyter** não aparece sozinha; `servidor.executar` repassa linha a linha.

## 7. Plano

### Fase 2 — Contrato e professor (PC) ✅

- **2.1** Contrato (`src/schema.py`).
- **2.2** Prompt do professor (`src/teacher/prompts.py`) com saída estruturada (`perguntar(..., esquema=Resposta)`).
- **2.3** Validadores (`src/teacher/validar.py`).
- **2.4** Tradução e anotação. No PC: 19 de 20 traduções e 20 de 20 anotações aprovadas (14 na 1ª tentativa).

### Fase 3 — Sintéticos ✅

- **3.1** `cenarios.py`: 14 cenários de golpe, 7 legítimos parecidos com golpe (40% dos legítimos) e 7 legítimos comuns. Eixos sorteados sem olhar o rótulo; `[EMAIL]`/`[LINK]` em ~50% das duas classes.
- **3.2** `gerar.py`: o professor escreve o e-mail com saída estruturada; reprova marcador pedido que faltou e marcador não pedido. Meta `SINTETICOS_META = 2000`, ids `sintetico:<i>`, grupos `sint:<i>`.
- **3.3** `filtrar.py`: MinHash contra toda a tabela mestre (teste incluído) e entre sintéticos; tabela de marcadores; `data/interim/revisao_sinteticos.md` com 50 exemplos para o usuário. Rodar antes e depois da anotação.
- No PC: 23 de 40 aprovados (todas as reprovações por marcador; o 4B põe `[LINK]` em golpes sem pedido).

### Fase 4 — Splits ✅

- **4.1** `splits.py`: 90/10 por grupo com `StratifiedGroupKFold`, estratos origem|fonte|rótulo. Grava `splits/treino.csv` e `splits/validacao.csv` (só commitar os da Spark).
- **4.2** `vazamento.py`: MinHash + `intfloat/multilingual-e5-small` (o `embeddinggemma` exige aceite de licença). Os cossenos do e5 ficam perto de 0,9; limiar de revisão 0,97; `reports/vazamento.csv` ordenado.
- **4.3** `exportar.py` + `prompts_aluno.py`: `data/processed/{train,val,test}.jsonl`. O teste leva a anotação do professor em `referencia`.

### Fase 5 — Avaliador e treino em miniatura ✅

- **5.1** `evaluate.py`: por recorte e total, precisão/recall/F1/taxa de falso positivo com IC por bootstrap, JSON válido, ancoragem, cobertura de tipos, tokens e tempo; `campanhas_rnp()`; `comparar()` (McNemar exato pareado). Previsão sem classificação legível conta como erro.
- **5.2** `baseline.py`: TF-IDF (palavras + caracteres) + regressão logística.
- **5.3** `train.py`: Unsloth `FastModel` em bf16, LoRA r=16/alpha=16/dropout 0 nas camadas de linguagem (visão congelada), lr 2e-4 cosseno, `warmup_steps=0.05` (o transformers 5 tirou o `warmup_ratio`), 2 épocas, lote efetivo 16, `MAX_TOKENS_TREINO = 2048` (conversas maiores ficam de fora e são contadas). Formato prompt/completion do TRL com `completion_only_loss`: o prompt é renderizado pelo template do modelo com `enable_thinking=False`, igual à inferência; isso substitui o `train_on_responses_only` e funciona igual nos três alunos.
- `prever.py`: previsões por API OpenAI (vLLM ou llama.cpp), temperatura 0, sem saída estruturada.
- PC: transformers 5.17, trl 1.15, unsloth 2026.11.4, torch 2.14.1+cu130 (o unsloth baixou o `datasets` para 4.8.5).

### Fase 6 — Migração para a Spark ✅ (código; execução na Spark pelo usuário)

- **6.1** `scripts/spark_construir.sh`: imagem `unsloth-dgx-spark` (Dockerfile oficial) e `phishexplain-unsloth:0.1` (+ projeto e matplotlib, com o torch da NGC travado).
- **6.2** `python -m src.ambiente` (o notebook roda dentro da imagem); `scripts/spark_container.sh` abre um terminal na imagem para depurar.
- **6.3** `git clone`; o notebook reconstrói os dados com a Fase 1 (`baixar`, `normalizar`, `limpar`, `agrupar`, `separar_teste`), que é determinística. O `splits/teste.csv` no git garante o mesmo teste.
- **6.4** `src/servidor.py` roda **no host**, só com a biblioteca padrão: `python()`/`modulo()` executam cada passo na imagem do projeto, e `subir()`/`parar()` controlam o vLLM num container próprio (a imagem da Unsloth não traz vLLM), pela imagem `PHISH_VLLM_IMAGEM` (padrão `nvcr.io/nvidia/vllm:26.09-py3`; a 26.01 traz vLLM 0.13 e não serve). Um modelo por vez; parar antes de treinar. Todos os containers rodam com o usuário do host (`--user`, `USER=phish`, `HOME=.home/`, `HF_HOME=/hf`), então nada fica com dono root.

### Fases 7 e 8 — Rodada completa e resultados (notebook)

`notebooks/spark.ipynb` (gerado por `scripts/montar_notebook.py`; edite o script e gere de novo), pedido do usuário em 2026-11-12. Roda no host da Spark e faz tudo sozinho: clona o repositório, constrói as imagens, sobe e desliga cada LLM; vazão do professor e estimativa de tempo; tradução, sintéticos e anotação; splits, exportação, vazamento; professor, baseline e alunos em zero-shot; treinos (Gemma 4 E2B, Qwen3.5-4B, S5-nano); previsões com LoRA; juiz `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (outra família, cego); `reports/resultados.md` com métricas por recorte, McNemar das perguntas de pesquisa, recall por campanha do RNP, notas do juiz, erros por fonte e marcadores do treino.

### Fase 9 — Opcional

Exportar o melhor modelo para GGUF.

## 8. Próximo passo

**Rodar o notebook na Spark** (`Run All` no host; depende de o repositório estar no GitHub, porque o notebook clona de lá). Depois: revisar `data/interim/revisao_sinteticos.md` (50 sintéticos), `reports/vazamento.csv` (topo) e dar notas aos 30 casos de `reports/calibracao_juiz.md`; commitar `splits/treino.csv`, `splits/validacao.csv` e `reports/`; preencher resultados e limitações do `README.md`.

Pontos a confirmar na primeira execução na Spark: a tag da imagem do vLLM (precisa de vLLM >= 0.17 com Qwen3.5); se o vLLM aceita os adaptadores LoRA dos alunos multimodais com `--language-model-only`; o template do Gemma 4 no `train.prompt_completion` (se não renderizar como prefixo, cai no `eos_token`).

## 9. Andamento

- [x] Fase 0: ambiente, config e professor de desenvolvimento
- [x] Fase 1: dados (refeita a 1.3 em 2026-11-11/12: espaços e mojibake)
- [x] Fase 2: contrato, prompt, validadores, tradução e anotação
- [x] Fase 3: sintéticos (código; testado no PC)
- [x] Fase 4: splits, exportação, vazamento (código; testado no PC)
- [x] Fase 5: avaliador, baseline, treino em miniatura (Qwen3.5-0.8B no PC)
- [x] Fase 6: Dockerfile, scripts, checagem de ambiente, vLLM em container (não executado: precisa da Spark)
- [x] Notebook da Spark (Fases 7 e 8) escrito
- [ ] Rodada completa na Spark (usuário)
- [ ] Fase 9: demo (opcional)

Atualize esta lista ao concluir cada passo.
