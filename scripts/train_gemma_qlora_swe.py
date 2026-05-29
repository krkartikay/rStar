from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path) as reader:
        for line in reader:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--train_file", default="data/swe_sft/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/gemma4-e2b-swe-qlora")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA with bitsandbytes requires CUDA. This machine has no CUDA device.")

    try:
        import bitsandbytes  # noqa: F401
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "QLoRA training requires bitsandbytes, peft, transformers, and trl in the CUDA training env."
        ) from exc

    train_dataset = load_dataset(args.train_file)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=training_args,
        packing=False,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved QLoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
