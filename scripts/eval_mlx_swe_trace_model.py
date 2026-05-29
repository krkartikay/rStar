from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from mlx_lm import generate, load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rstar_deepthink.agents.utils import swe_obs_wrap, swe_step_result_unwrap
from rstar_deepthink.coding import WorkspaceManager, coding_task_from_row, run_bash


SYSTEM_PROMPT = (
    "You are inside a checked-out software repository at the task's base commit. "
    "Respond with exactly one action using <bash>COMMAND<end_of_bash> or "
    "<finish>summary<end_of_finish>."
)


def load_rows(path: str, limit: int) -> list[dict]:
    rows = []
    with open(path) as reader:
        for line in reader:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def first_action(text: str) -> str:
    starts = [(text.find("<bash>"), "bash"), (text.find("<finish>"), "finish")]
    starts = [(idx, name) for idx, name in starts if idx != -1]
    if not starts:
        return text
    start, name = min(starts)
    if name == "bash":
        match = re.search(r"<bash>.*?(?:<end_of_bash>|</end_of_bash>|</bash>)", text[start:], re.S)
    else:
        match = re.search(r"<finish>.*?(?:<end_of_finish>|</end_of_finish>|</finish>)", text[start:], re.S)
    if match:
        return match.group(0)
    return text[start:]


def prompt_for(tokenizer, task, trace: str) -> str:
    user_prompt = f"Task:\n{task.problem_statement}\n\nTrace so far:\n{trace}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="outputs/mlx-qwen05-swe-lora-100")
    parser.add_argument("--tasks", default="/tmp/rstar_trainable_retry_tasks.jsonl")
    parser.add_argument("--output", default="outputs/mlx-qwen05-swe-lora-100-eval.jsonl")
    parser.add_argument("--workspace-root", default="/tmp/rstar_mlx_eval_workspaces")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    manager = WorkspaceManager(args.workspace_root, keep=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as writer:
        for row in load_rows(args.tasks, args.limit):
            task = coding_task_from_row(row)
            workspace = manager.create_root(task)
            trace = ""
            terminal = {"status": "max_steps", "candidate_patch": ""}
            steps = []
            for step_idx in range(args.max_steps):
                raw = generate(
                    model,
                    tokenizer,
                    prompt_for(tokenizer, task, trace),
                    verbose=False,
                    max_tokens=args.max_tokens,
                )
                step_text = first_action(raw)
                rendered_step, parsed = swe_step_result_unwrap(step_text)
                steps.append({"raw": raw, "step_text": step_text, "parsed": parsed})
                if parsed is None:
                    terminal = {"status": "parse_error", "raw": raw, "candidate_patch": WorkspaceManager.git_diff(workspace)}
                    break
                trace += rendered_step
                if parsed["action"] == "bash":
                    obs = run_bash(parsed["action_input"], workspace, timeout_seconds=30, output_max_chars=12000)
                    trace += "\n" + swe_obs_wrap(obs["output"])
                    continue
                if parsed["final_answer"]:
                    test = run_bash(task.test_command, workspace, timeout_seconds=30, output_max_chars=12000)
                    terminal = {
                        "status": "finished",
                        "final_answer": parsed["final_answer"],
                        "test_exit_code": test["exit_code"],
                        "test_output": test["output"],
                        "candidate_patch": WorkspaceManager.git_diff(workspace),
                    }
                    break
            if terminal["status"] == "max_steps":
                terminal["candidate_patch"] = WorkspaceManager.git_diff(workspace)
            writer.write(
                json.dumps(
                    {
                        "instance_id": task.instance_id,
                        "trace": trace,
                        "steps": steps,
                        "terminal": terminal,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            shutil.rmtree(Path(args.workspace_root) / WorkspaceManager.safe_name(task.instance_id), ignore_errors=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
