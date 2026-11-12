# Aqui baixamos os dados brutos que vamos usar. É uma etapa muito importante,
# porque a qualidade dos dados determina a qualidade do fine-tuning.
# Os arquivos são salvos em data/raw/ sem nenhuma alteração: os do GitHub ficam
# no formato original (JSON e CSV) e o dataset do Hugging Face é salvo em Parquet.
# A limpeza acontece depois, no passo 1.3.
# Arquivos que já existem são pulados, então rodar de novo é seguro.

import urllib.request
from pathlib import Path

from datasets import load_dataset

from src.config import DADOS_RAW

GITHUB_RAW = "https://raw.githubusercontent.com"

# Arquivos baixados direto por URL: caminho em data/raw -> endereço
ARQUIVOS = {
    "phishfuzzer/original_seed.json": (
        f"{GITHUB_RAW}/DataPhish/PhishFuzzer/main/PhishFuzzer_emails_original_seed_v1.json"
    ),
    "phishfuzzer/entity_rephrased.json": (
        f"{GITHUB_RAW}/DataPhish/PhishFuzzer/main/PhishFuzzer_emails_entity_rephrased_v1.json"
    ),
}

# E-mails reais em português do repositório stylometric-phishing-detection-ptbr.
# O repositório não tem licença: usamos só localmente, sem redistribuir.
for nome in ["dataset_b", "dataset_d", "dataset_fraudes_rnp", "dataset_opara"]:
    ARQUIVOS[f"ptbr_estilometria/{nome}.csv"] = (
        f"{GITHUB_RAW}/jeanderson-mdrs/stylometric-phishing-detection-ptbr/main/{nome}.csv"
    )

# Datasets do Hugging Face: caminho em data/raw -> nome do repositório
DATASETS_HF = {
    "zefang/train.parquet": "zefang-liu/phishing-email-dataset",
}


# Baixa uma URL para um arquivo temporário e só depois o renomeia para o nome
# certo. Assim, um download interrompido nunca fica com cara de arquivo completo.
def baixar_arquivo(destino: Path, url: str) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    urllib.request.urlretrieve(url, temporario)
    temporario.rename(destino)


# Baixa um dataset do Hugging Face e salva em Parquet
def baixar_hf(destino: Path, repo: str) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo, split="train")
    ds.to_parquet(str(destino))


def main() -> None:
    tarefas = [(caminho, url, baixar_arquivo) for caminho, url in ARQUIVOS.items()]
    tarefas += [(caminho, repo, baixar_hf) for caminho, repo in DATASETS_HF.items()]

    for caminho, origem, funcao in tarefas:
        destino = DADOS_RAW / caminho
        if destino.exists():
            print(f"já existe  {caminho}")
            continue

        print(f"baixando   {caminho} ...", flush=True)
        try:
            funcao(destino, origem)
        except Exception as erro:
            print(f"  FALHOU: {erro}")
            continue

        tamanho_mb = destino.stat().st_size / 1e6
        print(f"  ok ({tamanho_mb:.1f} MB)")


if __name__ == "__main__":
    main()