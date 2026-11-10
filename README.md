# PhishExplain: finetunning de LLMs abertos para detecção explicável de phishing em português brasileiro

**Uma comparação entre o Maritaca S5-nano, Qwen3.5-4B e Gemma 4 E2B, ajustado com LoRA para classificar e-mails em português e explicar a decisão.**

## A dor que buscamos resolver:

Phishing é um golpe de engenharia social: o atacante se passa por uma instituição ou pessoa de confiança para levar a vítima a entregar dados (senhas, cartões, códigos), fazer um pagamento ou abrir um link ou anexo malicioso. No Brasil, os golpes têm formato próprio: Pix de devolução, boleto falso, taxa dos Correios, pendência na Receita Federal, multa de pedágio digital.

Dois problemas motivam este trabalho:

1. **Pouco material em português.** A maior parte das bases públicas e dos estudos de detecção de phishing está em inglês. Para o português brasileiro, há poucas bases públicas e poucas avaliações com LLMs abertos.
2. **Classificar não basta.** Um detector que só responde "phishing" ou "legítimo" não ensina nada a quem recebe o e-mail. Uma explicação que aponta os trechos suspeitos ajuda a pessoa a decidir e a reconhecer o próximo golpe.

A Maritaca AI lançou em novembro de 2026 o S5-nano, um modelo de 4B parâmetros pós-treinado sobre o Qwen3.5-4B, com ganhos em benchmarks brasileiros segundo o próprio model card. Como os dois têm a mesma arquitetura e o mesmo ponto de partida, compará-los isola o efeito do treino em português numa tarefa de segurança.

## Perguntas de pesquisa:

1. O Maritaca S5-nano detecta phishing em PT-BR melhor que o Qwen3.5-4B do qual ele saiu? E quanto se perde com um modelo de cerca de metade do tamanho, o Gemma 4 E2B?
2. O finetunning com LoRA melhora de forma significativa os modelos em relação à sua versão original (zero-shot)?
3. As explicações citam trechos que de fato existem no e-mail?

## Tarefa:

Cada modelo recebe um e-mail e responde em JSON, listando as evidências antes de decidir:

```json
{
  "indicadores": [
    {"tipo": "urgencia", "trecho": "sua conta será bloqueada em 24 horas"},
    {"tipo": "link_suspeito", "trecho": "http://bancoexemplo-atualizacao.com/login"}
  ],
  "classificacao": "phishing",
  "explicacao": "O e-mail usa o nome do banco, mas o link leva a um domínio que não é dele. A ameaça de bloqueio serve para apressar você. Não clique; acesse o app do banco diretamente."
}
```

Todo `trecho` precisa ser uma cópia literal do e-mail, o que permite medir automaticamente se a explicação está ancorada no texto. O contrato completo está em [`src/schema.py`](src/schema.py).

## Modelos

| Papel | Modelo | Tamanho |
| --- | --- | --- |
| Aluno | [maritaca-ai/S5-nano](https://huggingface.co/maritaca-ai/S5-nano) | ≈ 4,2B |
| Aluno | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | ≈ 4,2B |
| Aluno | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | ≈ 2,3B efetivos |
| Professor | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B |

## Metodologia

### 1. Bases de dados

| Fonte | E-mails | Idioma | Autoria | Uso |
| --- | --- | --- | --- | --- |
| [PhishFuzzer](https://github.com/DataPhish/PhishFuzzer), sementes | 3.300 | 97% inglês | Humanos | Treino (traduzidas) e teste |
| PhishFuzzer, variantes | 19.800 | 97% inglês | LLM | Treino |
| [stylometric-phishing-detection-ptbr](https://github.com/jeanderson-mdrs/stylometric-phishing-detection-ptbr), `dataset_fraudes_rnp` | 90 | PT-BR | Humanos (golpes reais) | Teste |
| Mesmo repositório, `dataset_d`, `dataset_b` e `dataset_opara` | 4.994 | Português | LLM ou tradução | Treino |

Bases avaliadas e deixadas de fora: [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) (e-mails antigos, mistura spam com phishing) e o [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) de Al-Subaiey et al. (repete as fontes das sementes do PhishFuzzer).

### 2. Normalização e tratamento

Todas as fontes foram convertidas para um esquema único e depois limpas:

- **Texto:** normalização Unicode (NFC), remoção de caracteres invisíveis e padronização de espaços e quebras de linha.
- **Filtros:** só português e inglês; corpo entre 30 e 20.000 caracteres; descarte de registros com assunto ou remetente malformado.
- **Anonimização:** os marcadores da base do RNP (`cliente@email.com`, `[link removed]`) foram padronizados como `[EMAIL]` e `[LINK]`, para não virarem um atalho que identifica a classe.
- **Duplicatas exatas:** removidas, mantendo a cópia escrita por humanos.
- **Quase-duplicatas:** detectadas com MinHash e LSH (similaridade de Jaccard ≥ 0,8 sobre trechos de três palavras) e reunidas no mesmo grupo, para nunca ficarem divididas entre treino e teste.

| Etapa | E-mails |
| --- | --- |
| Entrada | 28.184 |
| Removidos (idioma, tamanho, malformados, duplicatas) | 780 |
| Após o tratamento | 27.404 |

O spam (cerca de um terço do PhishFuzzer) fica nos dados intermediários, mas fora desta versão do experimento: a tarefa é binária, phishing ou legítimo.

### 3. Divisão treino-teste

O teste foi separado e congelado em [`splits/teste.csv`](splits/teste.csv) antes de qualquer contato do professor com os e-mails. Ele tem 400 e-mails escritos por humanos, em dois recortes:

| Recorte | Phishing | Legítimo | Mede |
| --- | --- | --- | --- |
| Português nativo (RNP e sementes em português) | 100 | 0 | Recall em golpes reais |
| Traduzido (sementes em inglês, sorteadas por grupo) | 150 | 150 | Precisão e falso positivo |

Os 1.866 e-mails que pertencem aos mesmos grupos do teste (variantes das sementes) ficam fora do treino. O restante é dividido em treino e validação por grupo.

### 4. Geração das explicações (destilação)

Nenhuma base traz explicações. O professor (Qwen3.5-27B, Apache 2.0, rodando localmente) recebe cada e-mail **com o rótulo correto** e escreve a explicação no formato do contrato; ele nunca decide o rótulo. Também traduz os e-mails em inglês e gera e-mails sintéticos do contexto brasileiro, incluindo legítimos parecidos com golpe. Toda resposta passa por filtros automáticos: JSON válido, classificação igual ao rótulo e trechos encontrados literalmente no e-mail.

### 5. Treinamento

Os três alunos passam pelo mesmo fine-tuning supervisionado com LoRA (Unsloth e TRL), com os mesmos hiperparâmetros, no modo sem raciocínio.

### 6. Avaliação

Cada modelo é avaliado antes (zero-shot) e depois do fine-tuning, sempre no mesmo teste, com as métricas separadas por recorte:

- **Classificação:** precisão, recall, F1 e taxa de falso positivo, com intervalos de confiança por bootstrap.
- **Formato:** proporção de respostas em JSON válido.
- **Ancoragem:** proporção de trechos encontrados literalmente no e-mail.
- **Qualidade da explicação:** nota de um juiz LLM, calibrada com revisão humana.

Baselines: TF-IDF com regressão logística, os três modelos em zero-shot e o próprio professor.

## Resultados

*Em andamento.* A tabela abaixo será preenchida ao fim do experimento.

| Modelo | Versão | F1 phishing | Falso positivo | JSON válido | Ancoragem |
| --- | --- | --- | --- | --- | --- |
| S5-nano | zero-shot / LoRA | | | | |
| Qwen3.5-4B | zero-shot / LoRA | | | | |
| Gemma 4 E2B | zero-shot / LoRA | | | | |

## Limitações

- **Teste parcialmente traduzido.** Não há e-mails legítimos reais em português nas bases públicas usadas. A precisão e o falso positivo são medidos em e-mails traduzidos, que podem soar diferentes de e-mails escritos em português.
- **Golpes reais em pequeno número.** O recorte nativo tem 100 golpes, e uma mesma campanha (pedágio digital) aparece várias vezes.
- **Um único professor.** As explicações herdam os acertos e os vícios do Qwen3.5-27B.
- **Dados sintéticos.** Boa parte do treino foi escrita por LLMs, o que pode levar os alunos a aprender o estilo dos geradores em vez dos sinais de golpe.

## Como reproduzir

O desenvolvimento acontece num PC com GPU pequena; o trabalho pesado roda numa NVIDIA DGX Spark. O perfil é escolhido pela variável `PHISH_PERFIL` (`pc` ou `spark`).

```bash
conda create -n phishexplain python=3.12 -y
conda activate phishexplain
pip install -r requirements.txt

# Dados (Fase 1)
python -m src.data.baixar         # baixa as bases para data/raw/
python -m src.data.inspecionar    # inventário das bases brutas
python -m src.data.normalizar     # esquema único
python -m src.data.limpar         # limpeza e duplicatas exatas
python -m src.data.agrupar        # grupos de quase-duplicatas
python -m src.data.separar_teste  # congela o teste (só na primeira vez)

# Testes
python -m pytest -q
```

## Estrutura

```text
phishexplain/
├── data/          # bases brutas e intermediárias (fora do git)
├── models/        # modelos GGUF locais (fora do git)
├── outputs/       # adaptadores e previsões (fora do git)
├── reports/       # tabelas e análises
├── scripts/       # utilitários (ex.: subir o professor de desenvolvimento)
├── splits/        # ids do teste congelado
├── src/
│   ├── config.py  # perfis pc e spark, caminhos, modelos, semente
│   ├── schema.py  # contrato de saída
│   ├── data/      # download, normalização, limpeza, grupos e teste
│   └── teacher/   # cliente e prompts do professor
└── tests/
```
