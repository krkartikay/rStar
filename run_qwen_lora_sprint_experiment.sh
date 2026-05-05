#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

NAME="${NAME:?set NAME}"
DATA_PATH="${DATA_PATH:?set DATA_PATH}"
DATA_LENGTH="${DATA_LENGTH:-}"
BASE_MODEL="${BASE_MODEL:-models/qwen2.5-0.5b-instruct}"
RUN_ROOT="${RUN_ROOT:-outputs/qwen05_lora_3h_sprint}"
TASK="${TASK:-gsm8k}"
DEVICE="${DEVICE:-0}"
HF_HOME="${HF_HOME:-$ROOT_DIR/hf_cache}"
NOTEBOOK="${NOTEBOOK:-docs/qwen_lora_3h_sprint_lab_notebook.md}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
LR="${LR:-5e-5}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.3}"
MASTER_PORT="${MASTER_PORT:-1970}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MASTER_ADDR=localhost
export MASTER_PORT

mkdir -p "$RUN_ROOT"
EXP_DIR="$RUN_ROOT/$NAME"
ADAPTER_DIR="$EXP_DIR/adapter"
RESULTS_JSONL="$RUN_ROOT/results.jsonl"
RESULTS_MD="$RUN_ROOT/results.md"
mkdir -p "$EXP_DIR"

if [[ -z "$DATA_LENGTH" ]]; then
  DATA_LENGTH="$(python - "$DATA_PATH" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
fi

if [[ ! -f "$RESULTS_MD" ]]; then
  {
    echo "# Qwen LoRA 3h Sprint Results"
    echo
    echo "| Name | Data | Samples | Epochs | LR | LoRA r | Dropout | Correct | Total | Pass@1 | Adapter |"
    echo "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
  } > "$RESULTS_MD"
fi
if [[ ! -f "$RESULTS_JSONL" ]]; then
  printf '' > "$RESULTS_JSONL"
fi

echo "=== $NAME: train $DATA_LENGTH samples, epochs=$TRAIN_EPOCHS lr=$LR r=$LORA_R dropout=$LORA_DROPOUT ==="
python -m torch.distributed.launch \
  --master_addr localhost \
  --master_port "$MASTER_PORT" \
  --nproc_per_node=1 \
  --use_env train/train_SFT_lora.py \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --data_length "$DATA_LENGTH" \
  --output_dir "$ADAPTER_DIR" \
  --num_train_epochs "$TRAIN_EPOCHS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate "$LR" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --fp16 True \
  --gradient_checkpointing True \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --attn_impl eager \
  --report_to none \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  2>&1 | tee "$EXP_DIR/train.log"

echo "=== $NAME: eval $TASK ==="
python eval.py \
  --model "$BASE_MODEL" \
  --policy_lora_dir "$ADAPTER_DIR" \
  --device "$DEVICE" \
  --task "$TASK" \
  2>&1 | tee "$EXP_DIR/eval_generation.log"

EVAL_JSONL="$ADAPTER_DIR/$TASK.jsonl"
python eval_output.py --file_path "$EVAL_JSONL" 2>&1 | tee "$EXP_DIR/eval_score.log"

python - "$NAME" "$DATA_PATH" "$DATA_LENGTH" "$TRAIN_EPOCHS" "$LR" "$LORA_R" "$LORA_DROPOUT" "$ADAPTER_DIR" "$EVAL_JSONL" "$EXP_DIR/eval_score.log" "$RESULTS_JSONL" "$RESULTS_MD" "$NOTEBOOK" <<'PY'
import json
import re
import sys
from datetime import datetime

(
    name, data_path, data_length, epochs, lr, lora_r, dropout,
    adapter_dir, eval_jsonl, score_log, results_jsonl, results_md, notebook
) = sys.argv[1:]
text = open(score_log, encoding="utf-8", errors="replace").read()
matches = re.findall(r"(?m)^(\d+)\s+(\d+)\s+Pass 1 :\s+([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit(f"could not find Pass 1 in {score_log}")
correct, total, pass1 = matches[-1]
record = {
    "name": name,
    "time": datetime.now().isoformat(timespec="seconds"),
    "data_path": data_path,
    "samples": int(data_length),
    "epochs": float(epochs),
    "lr": lr,
    "lora_r": int(lora_r),
    "dropout": float(dropout),
    "adapter": adapter_dir,
    "eval_output": eval_jsonl,
    "correct": int(correct),
    "total": int(total),
    "pass1": float(pass1),
}
with open(results_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
with open(results_md, "a", encoding="utf-8") as f:
    f.write(
        f"| {name} | `{data_path}` | {record['samples']} | {record['epochs']:g} | "
        f"{lr} | {record['lora_r']} | {record['dropout']:g} | {record['correct']} | "
        f"{record['total']} | {record['pass1']:.6f} | `{adapter_dir}` |\n"
    )
with open(notebook, "a", encoding="utf-8") as f:
    f.write(
        "\n## Experiment: " + name + "\n\n"
        "```text\n"
        f"time: {record['time']}\n"
        f"data: {data_path}\n"
        f"samples: {record['samples']}\n"
        f"epochs: {record['epochs']:g}\n"
        f"lr: {lr}\n"
        f"lora_r: {record['lora_r']}\n"
        f"lora_dropout: {record['dropout']:g}\n"
        f"adapter: {adapter_dir}\n"
        f"GSM8K output: {eval_jsonl}\n"
        f"GSM8K score: {record['correct']} / {record['total']} = {record['pass1']} = {record['pass1'] * 100:.2f}%\n"
        "```\n"
    )
PY

echo "=== $NAME complete ==="
