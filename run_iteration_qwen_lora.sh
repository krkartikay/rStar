#!/usr/bin/env bash
set -euo pipefail

# Iterative local rStar loop for Qwen2.5-0.5B-Instruct + LoRA.
# Default seed adapter is the first clean LoRA round already recorded in the lab notebook.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-models/qwen2.5-0.5b-instruct}"
START_ADAPTER="${START_ADAPTER:-outputs/qwen05_lora_sft_gsm8k20_tree_clean}"
BOOTSTRAP_QAF="${BOOTSTRAP_QAF:-eval_data/gsm8k_20_bootstrap.json}"
BOOTSTRAP_CFG="${BOOTSTRAP_CFG:-config/qwen05_tree_mcts_bootstrap.yaml}"
TASK="${TASK:-gsm8k}"
DEVICE="${DEVICE:-0}"
MAX_ITERS="${MAX_ITERS:-3}"
START_ITER="${START_ITER:-2}"
RUN_ROOT="${RUN_ROOT:-outputs/qwen05_lora_iterations}"
HF_HOME="${HF_HOME:-$ROOT_DIR/hf_cache}"
NOTEBOOK="${NOTEBOOK:-docs/rstar_math_lab_notebook.md}"
RESULTS_JSONL="${RESULTS_JSONL:-$RUN_ROOT/results.jsonl}"
RESULTS_MD="${RESULTS_MD:-$RUN_ROOT/results.md}"

DATA_N="${DATA_N:-2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
LR="${LR:-1e-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-1960}"

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

mkdir -p "$RUN_ROOT"

if [[ ! -f "$RESULTS_JSONL" ]]; then
  printf '' > "$RESULTS_JSONL"
fi

if [[ ! -f "$RESULTS_MD" ]]; then
  {
    echo "# Qwen LoRA Iteration Results"
    echo
    echo "| Iteration | Rollout adapter | Trained adapter | SFT samples | GSM8K correct | GSM8K total | Pass@1 |"
    echo "| --- | --- | --- | ---: | ---: | ---: | ---: |"
  } > "$RESULTS_MD"
fi

if ! rg -q "Qwen LoRA Iteration Runner" "$NOTEBOOK"; then
  {
    echo
    echo "## Qwen LoRA Iteration Runner"
    echo
    echo "Date: $(date +%F)"
    echo
    echo "Added \`run_iteration_qwen_lora.sh\` to automate the local loop:"
    echo
    echo "1. Generate MCTS rollouts with the current adapter."
    echo "2. Extract and sample positive SFT traces."
    echo "3. Train a fresh LoRA adapter on those traces."
    echo "4. Run greedy GSM8K evaluation for the new adapter."
    echo "5. Append machine-readable results to \`$RESULTS_JSONL\` and table rows to \`$RESULTS_MD\`."
    echo
    echo "Default command:"
    echo
    echo "\`\`\`bash"
    echo "bash run_iteration_qwen_lora.sh"
    echo "\`\`\`"
    echo
    echo "Useful overrides:"
    echo
    echo "\`\`\`bash"
    echo "MAX_ITERS=5 BOOTSTRAP_QAF=eval_data/gsm8k_100_bootstrap.json bash run_iteration_qwen_lora.sh"
    echo "\`\`\`"
    echo
    echo "The script starts at iteration \`$START_ITER\` by default because the notebook already records base, instruct, and the first clean LoRA round."
  } >> "$NOTEBOOK"
fi

score_file() {
  local file_path="$1"
  local score_log="$2"
  python eval_output.py --file_path "$file_path" 2>&1 | tee "$score_log"
}

extract_score_json() {
  local score_log="$1"
  local score_json="$2"
  python - "$score_log" "$score_json" <<'PY'
import json
import re
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
text = open(log_path, encoding="utf-8", errors="replace").read()
matches = re.findall(r"(?m)^(\d+)\s+(\d+)\s+Pass 1 :\s+([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit(f"could not find Pass 1 line in {log_path}")
correct, total, pass1 = matches[-1]
record = {
    "correct": int(correct),
    "total": int(total),
    "pass1": float(pass1),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(record, f)
PY
}

count_json_items() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
    print(len(data))
except FileNotFoundError:
    print(0)
PY
}

prev_adapter="$START_ADAPTER"

for ((iter = START_ITER; iter <= MAX_ITERS; iter++)); do
  iter_dir="$RUN_ROOT/iter$(printf '%02d' "$iter")"
  rollout_dir="$iter_dir/rollout"
  train_dir="$iter_dir/train"
  eval_dir="$iter_dir/eval"
  adapter_dir="$iter_dir/adapter"
  mkdir -p "$rollout_dir" "$train_dir" "$eval_dir"

  echo "=== Iteration $iter: rollout with $prev_adapter ==="
  python main.py \
    --qaf "$BOOTSTRAP_QAF" \
    --custom_cfg "$BOOTSTRAP_CFG" \
    --model_dir "$BASE_MODEL" \
    --policy_lora_dir "$prev_adapter" \
    --save_in_model "$rollout_dir/run" \
    2>&1 | tee "$iter_dir/rollout.log"

  echo "=== Iteration $iter: extract SFT/RM data ==="
  python extra_sft_file.py \
    --data_dir "$rollout_dir" \
    --output_file "$train_dir/sft_extra_result.jsonl" \
    2>&1 | tee "$iter_dir/extra_sft.log"

  python extra_rm_file.py \
    --data_dir "$rollout_dir" \
    --output_file "$train_dir/rm_extra_result.jsonl" \
    --mode all \
    2>&1 | tee "$iter_dir/extra_rm.log"

  python train/sample_sft_data.py \
    --data_file "$train_dir/sft_extra_result.jsonl" \
    --output_file "$train_dir/sft.json" \
    --n "$DATA_N" \
    2>&1 | tee "$iter_dir/sample_sft.log"

  sft_samples="$(count_json_items "$train_dir/sft.json")"
  if [[ "$sft_samples" -eq 0 ]]; then
    echo "Iteration $iter produced 0 SFT samples; stopping before training."
    exit 1
  fi

  echo "=== Iteration $iter: train adapter on $sft_samples samples ==="
  master_port="$((MASTER_PORT_BASE + iter))"
  export MASTER_ADDR=localhost
  export MASTER_PORT="$master_port"

  python -m torch.distributed.launch \
    --master_addr localhost \
    --master_port "$master_port" \
    --nproc_per_node=1 \
    --use_env train/train_SFT_lora.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$train_dir/sft.json" \
    --data_length "$sft_samples" \
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

  echo "=== Iteration $iter: greedy $TASK eval ==="
  python eval.py \
    --model "$BASE_MODEL" \
    --policy_lora_dir "$adapter_dir" \
    --device "$DEVICE" \
    --task "$TASK" \
    2>&1 | tee "$iter_dir/eval_generation.log"

  eval_jsonl="$adapter_dir/$TASK.jsonl"
  if [[ ! -f "$eval_jsonl" ]]; then
    echo "Expected eval output not found: $eval_jsonl"
    exit 1
  fi

  score_file "$eval_jsonl" "$iter_dir/eval_score.log"
  extract_score_json "$iter_dir/eval_score.log" "$iter_dir/eval_score.json"

  python - "$iter" "$prev_adapter" "$adapter_dir" "$sft_samples" "$eval_jsonl" "$iter_dir/eval_score.json" "$RESULTS_JSONL" "$RESULTS_MD" <<'PY'
import json
import sys

iteration, rollout_adapter, trained_adapter, sft_samples, eval_jsonl, score_json, results_jsonl, results_md = sys.argv[1:]
score = json.load(open(score_json, encoding="utf-8"))
record = {
    "iteration": int(iteration),
    "rollout_adapter": rollout_adapter,
    "trained_adapter": trained_adapter,
    "sft_samples": int(sft_samples),
    "eval_output": eval_jsonl,
    **score,
}
with open(results_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
with open(results_md, "a", encoding="utf-8") as f:
    f.write(
        f"| {record['iteration']} | `{rollout_adapter}` | `{trained_adapter}` | "
        f"{record['sft_samples']} | {record['correct']} | {record['total']} | "
        f"{record['pass1']:.6f} |\n"
    )
PY

  echo "=== Iteration $iter complete: adapter $adapter_dir ==="
  prev_adapter="$adapter_dir"
done

echo "Done. Results:"
echo "  $RESULTS_JSONL"
echo "  $RESULTS_MD"
