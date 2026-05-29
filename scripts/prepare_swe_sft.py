from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are inside a checked-out software repository at the task's base commit. "
    "Respond with one action at a time using <bash>COMMAND<end_of_bash> or "
    "<finish>summary<end_of_finish>. Inspect files, edit source code, run focused "
    "tests, and finish only after a real patch is present."
)


def clean_finish_text(text: str) -> str:
    text = text or ""
    text = re.split(
        r"</?end_of_finish>|</?finish>|</final>|assistant to=python code>",
        text,
        maxsplit=1,
    )[0]
    return re.sub(r"\s+", " ", text).strip()


def clean_response(response: str, final_answer: str) -> str:
    response = response or ""
    finish_start = response.find("<finish>")
    if finish_start == -1:
        return response.strip() + f"\n<finish>{final_answer}<end_of_finish>"

    prefix = response[:finish_start]
    finish_body = clean_finish_text(response[finish_start + len("<finish>") :])
    if not finish_body:
        finish_body = final_answer
    return prefix.rstrip() + f"\n<finish>{finish_body}<end_of_finish>"


def convert_row(row: dict) -> dict | None:
    if int(row.get("test_exit_code", -1)) != 0:
        return None
    patch = row.get("candidate_patch") or ""
    response = row.get("response") or ""
    if "diff --git" not in patch or "<bash>" not in response:
        return None

    final_answer = clean_finish_text(row.get("final_answer") or "")
    cleaned_response = clean_response(response, final_answer)
    user_prompt = (
        f"Task:\n{row['prompt']}\n\n"
        "Generate a repository-editing trace that fixes the task. Use the required tags."
    )
    return {
        "id": row["instance_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": cleaned_response},
        ],
        "text": f"{SYSTEM_PROMPT}\n\n<|user|>\n{user_prompt}\n<|assistant|>\n{cleaned_response}",
        "candidate_patch": patch,
        "test_command": row.get("test_command", ""),
        "num_steps": row.get("num_steps", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated_traces/swe_fixture_sft_traces.jsonl")
    parser.add_argument("--output", default="data/swe_sft/train.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    with input_path.open() as reader:
        for line in reader:
            if not line.strip():
                continue
            row = convert_row(json.loads(line))
            if row is not None:
                converted.append(row)

    with output_path.open("w") as writer:
        for row in converted:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"input": str(input_path), "output": str(output_path), "rows": len(converted)}, indent=2))


if __name__ == "__main__":
    main()
