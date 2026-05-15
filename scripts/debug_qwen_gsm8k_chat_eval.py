#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].replace(",", "").strip()

    markers = [
        "####",
        "final answer is",
        "answer is",
        "therefore",
        "so,",
    ]
    lowered = text.lower()
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx != -1:
            nums = NUMBER_RE.findall(text[idx:])
            if nums:
                return nums[-1].replace(",", "").strip()

    nums = NUMBER_RE.findall(text)
    return nums[-1].replace(",", "").strip() if nums else ""


def normalize_number(text: str) -> str:
    text = str(text).strip().replace(",", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def answer_equiv(gold: str, pred: str) -> bool:
    gold_norm = normalize_number(gold)
    pred_norm = normalize_number(pred)
    if gold_norm == pred_norm:
        return True
    try:
        return abs(float(gold_norm) - float(pred_norm)) < 1e-9
    except ValueError:
        return False


def load_examples(path: Path, limit: int | None) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if "example" in data:
        data = data["example"]
    return data[:limit] if limit else data


def make_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following grade school math problem. "
                "Give the final answer after ####.\n\n"
                f"{question}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--data", default="eval_data/GSM8K_test.json")
    parser.add_argument("--out", default="outputs/debug_qwen_instruct_chat/gsm8k.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("chat_template:", bool(tokenizer.chat_template))
    print("sample_chat_prompt:")
    print(make_prompt(tokenizer, "Janet has 16 eggs and uses 7. How many are left?")[:500])

    examples = load_examples(Path(args.data), args.limit or None)
    prompts = [make_prompt(tokenizer, item["question"]) for item in examples]

    if args.backend == "vllm":
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            stop=["<|im_end|>"],
            skip_special_tokens=True,
            seed=42,
        )
        texts = [output.outputs[0].text for output in llm.generate(prompts, params)]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        texts = []
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="generate"):
            batch_prompts = prompts[start:start + args.batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_model_len - args.max_tokens,
            ).to(model.device)
            input_len = inputs.input_ids.shape[1]
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts.extend(
                tokenizer.batch_decode(
                    generated[:, input_len:],
                    skip_special_tokens=True,
                )
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with out_path.open("w") as f:
        for item, text in tqdm(list(zip(examples, texts)), desc="score"):
            pred = extract_answer(text)
            ok = answer_equiv(str(item["answer"]), pred)
            correct += int(ok)
            f.write(json.dumps({
                "index": item.get("index"),
                "question": item["question"],
                "answer": item["answer"],
                "prediction": pred,
                "correct": ok,
                "text": text,
            }, ensure_ascii=False) + "\n")

    total = len(examples)
    print(f"score: {correct} / {total} = {correct / total:.6f} = {100 * correct / total:.2f}%")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
