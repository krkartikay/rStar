from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rstar_deepthink.agents.utils import swe_step_result_unwrap
from rstar_deepthink.coding import WorkspaceManager, coding_task_from_row, run_bash


SYSTEM_PROMPT = (
    "You are inside a checked-out software repository at the task's base commit. "
    "Respond with exactly one action at a time using <bash>COMMAND<end_of_bash> or "
    "<finish>summary<end_of_finish>."
)


def load_tasks(path: str, limit: int) -> list:
    rows = []
    with open(path) as reader:
        for line in reader:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--tasks", default="/tmp/rstar_trainable_tasks.jsonl")
    parser.add_argument("--output", default="outputs/gemma4-e2b-swe-eval.jsonl")
    parser.add_argument("--workspace_root", default="/tmp/rstar_swe_model_eval_workspaces")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manager = WorkspaceManager(args.workspace_root, keep=True)

    with output_path.open("w") as writer:
        for row in load_tasks(args.tasks, args.limit):
            task = coding_task_from_row(row)
            workspace = manager.create_root(task)
            trace = ""
            terminal = None
            for _step in range(args.max_steps):
                prompt = f"{SYSTEM_PROMPT}\n\nTask:\n{task.problem_statement}\n\nTrace so far:\n{trace}"
                text = generate(model, tokenizer, prompt, args.max_new_tokens)
                step_text, parsed = swe_step_result_unwrap(text)
                if parsed is None:
                    terminal = {"status": "parse_error", "text": text}
                    break
                trace += step_text
                if parsed["action"] == "bash":
                    obs = run_bash(parsed["action_input"], workspace, timeout_seconds=30, output_max_chars=12000)
                    trace += f"\n<output>{obs['output']}<end_of_output>"
                elif parsed["final_answer"]:
                    test = run_bash(task.test_command, workspace, timeout_seconds=30, output_max_chars=12000)
                    terminal = {
                        "status": "finished",
                        "final_answer": parsed["final_answer"],
                        "test_exit_code": test["exit_code"],
                        "test_output": test["output"],
                        "candidate_patch": WorkspaceManager.git_diff(workspace),
                    }
                    break
            if terminal is None:
                terminal = {"status": "max_steps", "candidate_patch": WorkspaceManager.git_diff(workspace)}
            writer.write(json.dumps({"instance_id": task.instance_id, "trace": trace, "terminal": terminal}) + "\n")
            shutil.rmtree(Path(args.workspace_root) / WorkspaceManager.safe_name(task.instance_id), ignore_errors=True)
    print(f"Wrote eval results to {output_path}")


if __name__ == "__main__":
    main()
