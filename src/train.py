# 5.3: treino LoRA de um aluno com Unsloth + TRL. Mesmos hiperparâmetros para os três alunos.
# Cada conversa vira prompt (system + user, renderizado pelo template do próprio modelo com enable_thinking=False,
# exatamente como na inferência) + completion (o JSON do professor e o fim de turno). A loss é só na completion.
# Gemma 4: loss inicial de 13 a 15 é normal.
# Uso: python -m src.train --aluno qwen35-4b            (PC: --aluno qwen35-08b --max-exemplos 50)

import argparse
import json

from src.config import CFG, DADOS_PROCESSED, MAX_TOKENS_TREINO, SAIDAS, SEMENTE

LORA = {"r": 16, "lora_alpha": 16, "lora_dropout": 0, "bias": "none"}
LOTE_EFETIVO = 16


def prompt_completion(tk, msgs: list[dict]) -> dict:
    prompt = tk.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tudo = tk.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    completion = tudo[len(prompt):] if tudo.startswith(prompt) else msgs[-1]["content"] + tk.eos_token
    return {"prompt": prompt.removeprefix(tk.bos_token or ""), "completion": completion}  # o tokenizador põe o BOS


def carregar(tk, nome: str, limite: int | None):
    from datasets import Dataset

    linhas = [json.loads(l) for l in (DADOS_PROCESSED / f"{nome}.jsonl").open(encoding="utf-8")][:limite]
    pares = [prompt_completion(tk, l["messages"]) for l in linhas]
    cabem = [p for p in pares if len(tk(p["prompt"] + p["completion"]).input_ids) <= MAX_TOKENS_TREINO]
    print(f"{nome}: {len(cabem)} conversas ({len(pares) - len(cabem)} acima de {MAX_TOKENS_TREINO} tokens ficaram de fora)")
    return Dataset.from_list(cabem)


def amostra(modelo, tk, msgs: list[dict]) -> str:
    """Gera a resposta para uma conversa, para conferir que o adaptador produz um JSON plausível."""
    entrada = tk.apply_chat_template(msgs[:-1], add_generation_prompt=True, enable_thinking=False,
                                     return_tensors="pt", return_dict=True).to(modelo.device)
    saida = modelo.generate(**entrada, max_new_tokens=600, do_sample=False)
    return tk.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aluno", required=True, choices=list(CFG.alunos))
    ap.add_argument("--max-exemplos", type=int, help="só para testes rápidos no PC")
    ap.add_argument("--eval-passos", type=int, default=100)
    args = ap.parse_args()

    from unsloth import FastModel  # a Unsloth precisa ser importada antes de transformers e trl
    from trl import SFTConfig, SFTTrainer

    saida = SAIDAS / args.aluno
    modelo, processador = FastModel.from_pretrained(CFG.alunos[args.aluno], max_seq_length=MAX_TOKENS_TREINO,
                                                    load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    tk = getattr(processador, "tokenizer", processador)  # modelos multimodais devolvem um processador; usamos só o texto
    modelo = FastModel.get_peft_model(modelo, finetune_vision_layers=False, finetune_language_layers=True,
                                      finetune_attention_modules=True, finetune_mlp_modules=True, random_state=SEMENTE, **LORA)
    limite_val = args.max_exemplos and max(10, args.max_exemplos // 5)
    treino, val = carregar(tk, "train", args.max_exemplos), carregar(tk, "val", limite_val)

    lote = CFG.lote_por_passo
    trainer = SFTTrainer(model=modelo, processing_class=tk, train_dataset=treino, eval_dataset=val, args=SFTConfig(
        output_dir=str(saida / "checkpoints"), seed=SEMENTE, report_to="none",
        per_device_train_batch_size=lote, per_device_eval_batch_size=lote, gradient_accumulation_steps=LOTE_EFETIVO // lote,
        num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05,
        max_length=MAX_TOKENS_TREINO, completion_only_loss=True, bf16=True, optim="adamw_8bit",
        eval_strategy="steps", eval_steps=args.eval_passos, logging_steps=1, save_strategy="no",
    ))
    trainer.train()
    print("loss de validação final:", trainer.evaluate()["eval_loss"])

    modelo.save_pretrained(saida)  # só o adaptador LoRA
    tk.save_pretrained(saida)
    (saida / "historico.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    FastModel.for_inference(modelo)
    exemplo = json.loads((DADOS_PROCESSED / "val.jsonl").open(encoding="utf-8").readline())
    print("\nAmostra do adaptador para", exemplo["id"], "(rótulo:", exemplo["rotulo"] + ")\n" + amostra(modelo, tk, exemplo["messages"]))


if __name__ == "__main__":
    main()
