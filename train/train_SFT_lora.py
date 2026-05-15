# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.distributed.tensor
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer

from train_SFT import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    make_supervised_data_module,
    seed_torch,
    smart_tokenizer_and_embedding_resize,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    attn_impl: Optional[str] = field(default="eager")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Format SFT sources with tokenizer.apply_chat_template."},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    add_rstar_special_tokens: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    overwrite_output_dir: bool = field(default=True)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args, remaining_args = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )
    data_args.data_length = int(remaining_args[1])
    print(training_args.run_name)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    training_args.seed = seed

    torch_dtype = torch.float16 if training_args.fp16 else torch.float32
    if training_args.bf16:
        torch_dtype = torch.bfloat16

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation=model_args.attn_impl,
        torch_dtype=torch_dtype,
        use_cache=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left" if data_args.use_chat_template else ("left" if "mistral" in model_args.model_name_or_path.lower() else "right"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path.lower() and "llama-3" not in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    if lora_args.add_rstar_special_tokens:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<code>",
                    "<end_of_step>",
                    "<end_of_code>",
                    "<output>",
                    "<end_of_output>",
                    "<answer>",
                    "<end_of_answer>",
                    "<|user|>",
                    "<|assistant|>",
                    "<refine>",
                    "<end_of_refine>",
                    "\n<|assistant|>",
                    "<error_info>",
                    "<end_of_error_info>",
                    "<BACK>",
                ]
            },
            replace_additional_special_tokens=False,
        )
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=[m.strip() for m in lora_args.lora_target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
