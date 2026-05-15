#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/.pyenv/versions/rstar/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/qwen2.5-0.5b-instruct}"
START_ADAPTER="${START_ADAPTER:-}"
BOOTSTRAP_QAF="${BOOTSTRAP_QAF:-eval_data/gsm8k_20_bootstrap.json}"
BOOTSTRAP_CFG="${BOOTSTRAP_CFG:-config/qwen_chat_mcts_bootstrap.yaml}"
EVAL_CFG="${EVAL_CFG:-config/sft_eval_greedy_qwen_chat.yaml}"
TASK="${TASK:-gsm8k}"
DEVICE="${DEVICE:-0}"
EVAL_EACH_ITER="${EVAL_EACH_ITER:-1}"
CUMULATIVE_SFT="${CUMULATIVE_SFT:-0}"
START_ITER="${START_ITER:-1}"
MAX_ITERS="${MAX_ITERS:-2}"
RUN_ROOT="${RUN_ROOT:-outputs/qwen_chat_mcts_lora_iterations}"
NOTEBOOK="${NOTEBOOK:-docs/qwen_chat_template_rstar_lab_notebook.md}"
RESULTS_JSONL="${RESULTS_JSONL:-$RUN_ROOT/results.jsonl}"
RESULTS_MD="${RESULTS_MD:-$RUN_ROOT/results.md}"

DATA_N="${DATA_N:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LR="${LR:-5e-5}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-2060}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

mkdir -p "$RUN_ROOT"

if [[ ! -f "$RESULTS_JSONL" ]]; then
  : > "$RESULTS_JSONL"
fi

if [[ ! -f "$RESULTS_MD" ]]; then
  {
    echo "# Qwen Chat MCTS LoRA Iteration Results"
    echo
    echo "| Iteration | Rollout policy | Trained adapter | SFT samples | Correct | Total | Pass@1 |"
    echo "| --- | --- | --- | ---: | ---: | ---: | ---: |"
  } > "$RESULTS_MD"
fi

count_json_items() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json
import sys
try:
    print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
except FileNotFoundError:
    print(0)
PY
}

extract_score_json() {
  local score_log="$1"
  local score_json="$2"
  "$PYTHON" - "$score_log" "$score_json" <<'PY'
import json
import re
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
matches = re.findall(r"(?m)^(\d+)\s+(\d+)\s+Pass 1 :\s+([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit(f"could not find Pass 1 line in {sys.argv[1]}")
correct, total, pass1 = matches[-1]
json.dump({"correct": int(correct), "total": int(total), "pass1": float(pass1)}, open(sys.argv[2], "w"))
PY
}

prev_adapter="$START_ADAPTER"

for ((iter = START_ITER; iter <= MAX_ITERS; iter++)); do
  iter_tag="$(printf '%02d' "$iter")"
  iter_dir="$RUN_ROOT/iter$iter_tag"
  rollout_dir="$iter_dir/rollout"
  train_dir="$iter_dir/train"
  adapter_dir="$iter_dir/adapter"
  mkdir -p "$rollout_dir" "$train_dir"

  rollout_policy="${prev_adapter:-$BASE_MODEL}"
  echo "=== Iteration $iter: MCTS rollout with $rollout_policy ==="
  rollout_cmd=(
    "$PYTHON" main.py
    --qaf "$BOOTSTRAP_QAF"
    --custom_cfg "$BOOTSTRAP_CFG"
    --model_dir "$BASE_MODEL"
    --save_in_model "$rollout_dir/run"
  )
  if [[ -n "$prev_adapter" ]]; then
    rollout_cmd+=(--policy_lora_dir "$prev_adapter")
  fi
  "${rollout_cmd[@]}" 2>&1 | tee "$iter_dir/rollout.log"

  echo "=== Iteration $iter: extract positive SFT traces ==="
  "$PYTHON" extra_sft_file.py \
    --data_dir "$rollout_dir" \
    --output_file "$train_dir/sft_extra_result.jsonl" \
    2>&1 | tee "$iter_dir/extra_sft.log"

  "$PYTHON" train/sample_sft_data.py \
    --data_file "$train_dir/sft_extra_result.jsonl" \
    --output_file "$train_dir/sft_current.json" \
    --n "$DATA_N" \
    2>&1 | tee "$iter_dir/sample_sft.log"

  if [[ "$CUMULATIVE_SFT" == "1" && "$iter" -gt 1 ]]; then
    prev_tag="$(printf '%02d' "$((iter - 1))")"
    prev_sft="$RUN_ROOT/iter$prev_tag/train/sft.json"
    "$PYTHON" - "$prev_sft" "$train_dir/sft_current.json" "$train_dir/sft.json" <<'PY'
import json
import sys

items = []
seen = set()
for path in sys.argv[1:3]:
    try:
        data = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        data = []
    for item in data:
        key = (item.get("query", ""), item.get("response", ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

json.dump(items, open(sys.argv[3], "w", encoding="utf-8"), indent=4)
print(len(items))
PY
  else
    cp "$train_dir/sft_current.json" "$train_dir/sft.json"
  fi

  sft_samples="$(count_json_items "$train_dir/sft.json")"
  if [[ "$sft_samples" -eq 0 ]]; then
    echo "Iteration $iter produced 0 SFT samples; stopping."
    exit 1
  fi

  echo "=== Iteration $iter: train Qwen-chat LoRA adapter on $sft_samples samples ==="
  master_port="$((MASTER_PORT_BASE + iter))"
  export MASTER_ADDR=localhost
  export MASTER_PORT="$master_port"

  "$PYTHON" -m torch.distributed.launch \
    --master_addr localhost \
    --master_port "$master_port" \
    --nproc_per_node=1 \
    --use_env train/train_SFT_lora.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$train_dir/sft.json" \
    --data_length "$sft_samples" \
    --use_chat_template True \
    --output_dir "$adapter_dir" \
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
    --max_grad_norm 0.3 \
    --attn_impl eager \
    --report_to none \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    2>&1 | tee "$iter_dir/train_lora.log"

  eval_jsonl=""
  if [[ "$EVAL_EACH_ITER" == "1" || "$iter" -eq "$MAX_ITERS" ]]; then
    echo "=== Iteration $iter: greedy $TASK eval with Qwen-chat rStar ==="
    "$PYTHON" eval.py \
      --model "$BASE_MODEL" \
      --policy_lora_dir "$adapter_dir" \
      --device "$DEVICE" \
      --task "$TASK" \
      --custom_cfg "$EVAL_CFG" \
      2>&1 | tee "$iter_dir/eval_generation.log"

    eval_jsonl="$adapter_dir/$TASK.jsonl"
    "$PYTHON" eval_output.py --file_path "$eval_jsonl" 2>&1 | tee "$iter_dir/eval_score.log"
    extract_score_json "$iter_dir/eval_score.log" "$iter_dir/eval_score.json"
  else
    echo '{"correct": null, "total": null, "pass1": null}' > "$iter_dir/eval_score.json"
  fi

  "$PYTHON" - "$iter" "$rollout_policy" "$adapter_dir" "$sft_samples" "$eval_jsonl" "$iter_dir/eval_score.json" "$RESULTS_JSONL" "$RESULTS_MD" <<'PY'
import json
import sys
iteration, rollout_policy, adapter, sft_samples, eval_jsonl, score_json, results_jsonl, results_md = sys.argv[1:]
score = json.load(open(score_json))
record = {
    "iteration": int(iteration),
    "rollout_policy": rollout_policy,
    "trained_adapter": adapter,
    "sft_samples": int(sft_samples),
    "eval_output": eval_jsonl,
    **score,
}
with open(results_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
with open(results_md, "a", encoding="utf-8") as f:
    pass1 = "" if record["pass1"] is None else f"{record['pass1']:.6f}"
    correct = "" if record["correct"] is None else record["correct"]
    total = "" if record["total"] is None else record["total"]
    f.write(
        f"| {record['iteration']} | `{rollout_policy}` | `{adapter}` | "
        f"{record['sft_samples']} | {correct} | {total} | {pass1} |\n"
    )
PY

  {
    echo
    echo "### Qwen-chat MCTS LoRA iteration $iter"
    echo
    echo '```text'
    echo "rollout policy: $rollout_policy"
    echo "trained adapter: $adapter_dir"
    echo "SFT samples: $sft_samples"
    cat "$iter_dir/eval_score.json"
    echo
    echo '```'
  } >> "$NOTEBOOK"

  prev_adapter="$adapter_dir"
done
