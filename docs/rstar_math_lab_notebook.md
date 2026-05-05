# rStar-Math Local Experiment Lab Notebook

Date: 2026-05-05  
Workspace: `/home/krkartikay/lossfunk/auto_replicate`  
Repository branch: `rStar-math`  
Repository commit: `b3c1846185f4af877e1e441a67c4c5fcfed9d442`

## Goal

Set up the Microsoft rStar-Math repository without conda, then attempt a scaled-down bootstrapping and SFT training workflow on a single consumer GPU with 8 GB VRAM. The original README targets A100 80 GB and large Qwen math models; this notebook records the adaptations needed for a small local run.

## Environment Setup

The repository was already on `rStar-math`, tracking `origin/rStar-math`.

Created and selected a pyenv environment:

```bash
pyenv virtualenv 3.11.13 rstar
pyenv local rstar
```

This wrote `.python-version` with:

```text
rstar
```

Installed README requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installed MARIO_EVAL as requested by the README:

```bash
git clone https://github.com/MARIO-Math-Reasoning/MARIO_EVAL.git
cd MARIO_EVAL/latex2sympy
python -m pip install .
cd ..
python -m pip install -e .
```

Key package versions:

```text
Python 3.11.13
torch 2.5.1
vllm 0.6.6.post1
transformers 4.45.2
trl 0.9.6
```

CUDA was checked in the host environment:

```text
torch: 2.5.1+cu124
CUDA runtime: 12.4
cuda available: True
device count: 1
device 0: NVIDIA GeForce RTX 3070
```

## Important Repo Constraints Found

`main.py` accepts `--model_dir`, but the tree/agent validator requires `config.model_dir` to be an existing local filesystem path. A Hugging Face model ID such as `gpt2` works for vLLM, but fails the agent validator.

Workaround: materialize Hugging Face models locally with `save_pretrained`.

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id='Qwen/Qwen2.5-0.5B-Instruct'; out='models/qwen2.5-0.5b-instruct'; m=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True); t=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True); m.save_pretrained(out); t.save_pretrained(out); print('saved', out)"
```

The vLLM engine in `rstar_deepthink/llms/llm_engine.py` hard-codes `dtype="bfloat16"`. This did not block the RTX 3070 runs, but it is worth remembering when debugging GPU compatibility.

## GPT-2 Smoke Attempts

Local model directory:

```text
models/gpt2
```

Configs added:

```text
config/gpt2_tiny_mcts.yaml
config/gpt2_tiny_mcts_fewshot.yaml
```

Tiny QAF added:

```text
eval_data/gpt2_bootstrap_tiny.json
```

Result: GPT-2 loaded and ran through `main.py`, but did not produce valid rStar final answers. Even with few-shot examples, it mostly copied demonstration text or produced malformed code/answers.

Extraction produced empty datasets:

```text
outputs/gpt2_bootstrap_tiny/sft_extra_result.jsonl: 0 lines
outputs/gpt2_bootstrap_tiny/rm_extra_result.jsonl: 0 lines
outputs/gpt2_bootstrap_tiny_fewshot/sft_extra_result.jsonl: 0 lines
outputs/gpt2_bootstrap_tiny_fewshot/rm_extra_result.jsonl: 0 lines
```

Conclusion: the pipeline worked, but GPT-2 is not instruction-following enough for rStar bootstrapping.

## Qwen 0.5B Smoke Attempts

Materialized model:

```text
models/qwen2.5-0.5b-instruct
```

Configs added:

```text
config/qwen05_tiny_mcts_fewshot.yaml
config/qwen05_tiny_mcts_singlepass.yaml
```

Initial Qwen few-shot runs with the README-style stop list:

```yaml
stop: ["<end_of_step>", "<end_of_code>", "<end_of_output>", "<end_of_answer>"]
```

Qwen generated correct code and `<output>4<end_of_output>`, but often stopped before writing the final `<answer>...\boxed{}...</answer>` block. SFT extraction stayed empty.

Working adaptation:

```yaml
stop: ["<end_of_answer>"]
max_depth: 1
iterations: 1
max_tokens: 256
```

This lets Qwen emit the full code/output/answer trace in one completion. With this single-pass config, the tiny `2 + 2` run generated:

```text
<code>
# Step 1: Calculate the sum of 2 and 2
result = 2 + 2
<end_of_step>

# Now print the final answer
print(result)
<end_of_code>
<output>4
<end_of_output><answer>The sum of 2 and 2 is \boxed{4}
<end_of_answer>
```

Tiny extraction result:

```text
outputs/qwen05_bootstrap_singlepass/sft_extra_result.jsonl: 1 line
outputs/qwen05_bootstrap_singlepass/rm_extra_result.jsonl: 0 lines
```

RM extraction stayed empty because the single-pass setup generates only one branch per question, not positive/negative branch pairs.

## 100-Question GSM8K Bootstrapping Run

Created a 100-example subset:

```bash
python -c "import json; d=json.load(open('eval_data/GSM8K_test.json'))[:100]; json.dump(d, open('eval_data/gsm8k_100_bootstrap.json','w'), indent=2); print(len(d))"
```

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python main.py \
  --qaf eval_data/gsm8k_100_bootstrap.json \
  --custom_cfg config/qwen05_tiny_mcts_singlepass.yaml \
  --model_dir models/qwen2.5-0.5b-instruct \
  --save_in_model outputs/qwen05_gsm8k100_singlepass/run
```

Generation result:

```text
outputs/qwen05_gsm8k100_singlepass/run.jsonl: 100 lines
runtime: about 3m39s
```

Observed vLLM memory profile:

```text
total GPU memory: 7.62 GiB
gpu_memory_utilization: 0.50
vLLM budget: 3.81 GiB
model weights: 0.93 GiB
activation peak during profile: 1.39 GiB
KV cache reserved: 1.39 GiB
```

Live `nvidia-smi` sample during generation:

```text
2941 MiB / 8192 MiB
```

Some postprocess warnings appeared:

```text
can only concatenate str (not "int") to str
```

These did not stop the run. They appear when processing some generated traces.

## Data Extraction From 100-Question Run

Commands:

```bash
python extra_sft_file.py \
  --data_dir outputs/qwen05_gsm8k100_singlepass \
  --output_file outputs/qwen05_gsm8k100_singlepass/sft_extra_result.jsonl

python extra_rm_file.py \
  --data_dir outputs/qwen05_gsm8k100_singlepass \
  --output_file outputs/qwen05_gsm8k100_singlepass/rm_extra_result.jsonl

python train/sample_sft_data.py \
  --data_file outputs/qwen05_gsm8k100_singlepass/sft_extra_result.jsonl \
  --output_file outputs/qwen05_gsm8k100_singlepass/sft.json \
  --n 2
```

Counts:

```text
run.jsonl: 100 traces
terminal answers: 56
exact final-answer matches: 4
sft_extra_result.jsonl: 56 traces
rm_extra_result.jsonl: 0 traces
sft.json: 4 positive training samples
```

Interpretation:

Only 4 of the 100 generated traces were correct by exact final-answer match and survived SFT positive filtering. RM data remains empty because the single-pass generation does not create paired positive/negative sibling branches.

## SFT Training Runs

### Tiny 1-Sample Smoke

Output:

```text
outputs/qwen05_sft_smoke
```

This verified that the training code and save/load path worked.

### 4-Sample GSM8K100 SFT Run

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
WANDB_DISABLED=true \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
NCCL_BLOCKING_WAIT=0 \
FLASH_ATTENTION_DETERMINISTIC=1 \
MASTER_ADDR=localhost \
MASTER_PORT=1943 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_IFNAME=lo \
python -m torch.distributed.launch \
  --master_addr localhost \
  --master_port 1943 \
  --nproc_per_node=1 \
  --use_env train/train_SFT.py \
  --model_name_or_path models/qwen2.5-0.5b-instruct \
  --data_path outputs/qwen05_gsm8k100_singlepass/sft.json \
  --data_length 4 \
  --output_dir outputs/qwen05_sft_gsm8k100 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 7e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --model_max_length 768 \
  --fp16 True \
  --gradient_checkpointing True \
  --attn_impl eager \
  --report_to none
```

Output:

```text
outputs/qwen05_sft_gsm8k100
```

Training metrics:

```text
train_runtime: 2.1684 seconds
train_samples_per_second: 5.534
train_steps_per_second: 1.384
train_loss: 2.2959569295247397
epoch: 3.0
```

The model reloads successfully:

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('outputs/qwen05_sft_gsm8k100'); AutoModelForCausalLM.from_pretrained('outputs/qwen05_sft_gsm8k100'); print('load ok')"
```

Observed issue:

```text
grad_norm: nan / inf
```

This suggests the fp16 full fine-tune setup is numerically fragile, especially with only 4 examples. The run completed, but this should not be treated as a quality training result.

## VRAM Measurements

Generation, Qwen 0.5B, single-pass, 1536 max model length:

```text
Peak observed by nvidia-smi: about 2941 MiB
vLLM planned budget: 3.81 GiB
```

SFT training, Qwen 0.5B full fine-tune:

```text
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
model_max_length: 768
fp16: True
gradient_checkpointing: True
peak observed by nvidia-smi: about 6333 MiB / 8192 MiB
idle after training: 233 MiB / 8192 MiB
```

Answer to "How much VRAM does training take?" for this exact run:

```text
Approximately 6.3 GB peak on the RTX 3070.
```

## Current Artifacts

Key added files:

```text
.python-version
config/gpt2_tiny_mcts.yaml
config/gpt2_tiny_mcts_fewshot.yaml
config/qwen05_tiny_mcts_fewshot.yaml
config/qwen05_tiny_mcts_singlepass.yaml
eval_data/gpt2_bootstrap_tiny.json
eval_data/gsm8k_100_bootstrap.json
models/gpt2/
models/qwen2.5-0.5b-instruct/
outputs/qwen05_gsm8k100_singlepass/
outputs/qwen05_sft_gsm8k100/
```

## Conclusions

The repository can run end-to-end on an 8 GB RTX 3070 if the model and search settings are scaled down aggressively.

GPT-2 is too weak for rStar bootstrapping. Qwen2.5-0.5B-Instruct can produce valid traces, but with the small single-pass setup only 4 out of 100 GSM8K generations were correct enough for positive SFT.

The single-pass stop condition is currently the most reliable small-model adaptation:

```yaml
stop: ["<end_of_answer>"]
```

The README-style stepwise stop list is closer to the intended algorithm, but on Qwen 0.5B it often stops after `<output>` and fails to reach `<answer>`, which prevents SFT extraction.

RM training was not reached because RM extraction needs paired positive and negative branches. The current single-pass setup does not create those pairs.

## Next Steps

1. Improve bootstrapping quality before scaling training. The limiting factor is not VRAM; it is low correctness in generated traces.
2. Try a slightly stronger model that still fits 8 GB, or use quantization/LoRA if moving above 0.5B.
3. Consider fixing the postprocess warning:

```text
can only concatenate str (not "int") to str
```

4. For real RM data, use multiple sampled branches per question (`n_generate_sample > 1`) and restore a tree-like setup once final-answer generation is reliable.
5. For SFT stability, consider fp32, lower learning rate, LoRA, or a larger positive sample set. The observed `grad_norm` nan/inf makes the current full fp16 run a smoke test, not a trustworthy training run.

## Tree-Style Bootstrapping Pilot

Created a scaled tree MCTS config:

```text
config/qwen05_tree_mcts_bootstrap.yaml
```

This keeps the sample MCTS shape but lowers constants for the 8 GB GPU:

```yaml
n_generate_sample: 4
best_of: 4
iterations: 12
max_depth: 4
max_tokens: 512
batch_size: 8
max_model_len: 2048
llm_gpu_memory_utilization: 0.5
tp: 1
stop: ["<end_of_code>", "<end_of_answer>"]
```

The stop tokens intentionally differ from the README config. The earlier Qwen 0.5B runs with `<end_of_step>` in the stop list often stopped before emitting `<answer>`. This pilot branches over complete code attempts and final answers instead.

Created a 20-example pilot subset:

```text
eval_data/gsm8k_20_bootstrap.json
```

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python main.py \
  --qaf eval_data/gsm8k_20_bootstrap.json \
  --custom_cfg config/qwen05_tree_mcts_bootstrap.yaml \
  --model_dir models/qwen2.5-0.5b-instruct \
  --save_in_model outputs/qwen05_gsm8k20_tree/run
```

Result:

```text
outputs/qwen05_gsm8k20_tree/run.jsonl: 20 rows
runtime: about 9 minutes
tree nodes total: 207
nodes per question: min 2, max 21
terminal answers total: 32
correct terminal leaves total: 7
questions with at least one correct leaf: 5 / 20
questions with at least two terminal answers: 10 / 20
root children average: 3.55
```

Intermediate rollout files were saved:

```text
outputs/qwen05_gsm8k20_tree/rollout/rollout00run.jsonl
...
outputs/qwen05_gsm8k20_tree/rollout/rollout11run.jsonl
```

The old generation warning still appears:

```text
can only concatenate str (not "int") to str
```

This config improved data structure but did not fix that warning.

Extraction commands:

```bash
python extra_sft_file.py \
  --data_dir outputs/qwen05_gsm8k20_tree \
  --output_file outputs/qwen05_gsm8k20_tree/sft_extra_result.jsonl

python extra_rm_file.py \
  --data_dir outputs/qwen05_gsm8k20_tree \
  --output_file outputs/qwen05_gsm8k20_tree/rm_extra_result.jsonl \
  --mode all

python train/sample_sft_data.py \
  --data_file outputs/qwen05_gsm8k20_tree/sft_extra_result.jsonl \
  --output_file outputs/qwen05_gsm8k20_tree/sft.json \
  --n 2
```

Extraction result:

```text
sft_extra_result.jsonl: 57 traces
positive SFT leaves: 7
negative SFT leaves: 50
rm_extra_result.jsonl: 2 preference pairs
sft.json: 7 positive training samples
```

This confirms the tree setup can produce both SFT positives and RM preference pairs, though the sample is still tiny.

## SFT Smoke on Tree-Pilot Data

First attempted fp32 SFT on the 7 positive tree samples. It failed with CUDA OOM. At the time, a stale host Python GPU process from the previous vLLM run was holding about 2912 MiB VRAM. After terminating that process, idle GPU memory returned to about 233 MiB.

Successful fp16 smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCH_NCCL_BLOCKING_WAIT=0 \
FLASH_ATTENTION_DETERMINISTIC=1 \
MASTER_ADDR=localhost \
MASTER_PORT=1948 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_IFNAME=lo \
python -m torch.distributed.launch \
  --master_addr localhost \
  --master_port 1948 \
  --nproc_per_node=1 \
  --use_env train/train_SFT.py \
  --model_name_or_path models/qwen2.5-0.5b-instruct \
  --data_path outputs/qwen05_gsm8k20_tree/sft.json \
  --data_length 7 \
  --output_dir outputs/qwen05_sft_gsm8k20_tree_smoke \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 3e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --model_max_length 768 \
  --fp16 True \
  --gradient_checkpointing True \
  --attn_impl eager \
  --report_to none
```

Result:

```text
output_dir: outputs/qwen05_sft_gsm8k20_tree_smoke
train_runtime: 0.9669 seconds
train_samples_per_second: 7.24
train_steps_per_second: 1.034
train_loss: 1.623293161392212
epoch: 0.57
checkpoint size: 2.4G
saved model reload: ok
```

Observed issue:

```text
grad_norm: inf
```

The SFT script works on the tree-bootstrap SFT JSON and saves a reloadable checkpoint. Numeric stability is still unresolved for fp16 full fine-tuning.

## LoRA SFT Stability Test

Installed PEFT into the `rstar` pyenv environment and added it to `requirements.txt`:

```text
peft==0.19.1
```

Added a LoRA SFT entrypoint:

```text
train/train_SFT_lora.py
```

The script reuses the repository's existing SFT dataset and collator, wraps the base model with PEFT LoRA, trains only adapter weights, and saves adapter-only output.

One compatibility patch was needed for the installed Torch/PEFT combination:

```python
import torch.distributed.tensor
```

Without that import, PEFT raised:

```text
AttributeError: module 'torch.distributed' has no attribute 'tensor'
```

Qwen2.5-0.5B LoRA target modules:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

First LoRA run added the rStar markers as tokenizer special tokens, matching the original full SFT script. It trained successfully, but PEFT saved resized embeddings with the adapter:

```text
output_dir: outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep
trainable params: 4,399,104
trainable percent: 0.8830%
runtime: 39.0935 seconds
train_loss: 0.4750397420355252
adapter output size: 551M
actual adapter_model.safetensors: 561,261,480 bytes
grad logs: 140
finite grad logs: 137
nonfinite grad logs: 3
finite grad_norm min/max: 0.030059609562158585 / 6.896251201629639
```

The first three nonfinite grad logs occurred during zero-learning-rate warmup. After warmup, gradients were finite.

To keep LoRA lightweight, the script was changed to make the rStar special-token resize optional and default it off:

```python
add_rstar_special_tokens: bool = False
```

This still trains on the same text, but the marker strings are tokenized as normal subwords instead of adding new embedding rows.

Successful no-embedding LoRA run:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
WANDB_DISABLED=true \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCH_NCCL_BLOCKING_WAIT=0 \
FLASH_ATTENTION_DETERMINISTIC=1 \
MASTER_ADDR=localhost \
MASTER_PORT=1953 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_IFNAME=lo \
python -m torch.distributed.launch \
  --master_addr localhost \
  --master_port 1953 \
  --nproc_per_node=1 \
  --use_env train/train_SFT_lora.py \
  --model_name_or_path models/qwen2.5-0.5b-instruct \
  --data_path outputs/qwen05_gsm8k20_tree/sft.json \
  --data_length 7 \
  --output_dir outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed \
  --num_train_epochs 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 512 \
  --fp16 True \
  --gradient_checkpointing True \
  --max_grad_norm 0.3 \
  --attn_impl eager \
  --report_to none \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
```

Result:

```text
output_dir: outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed
trainable params: 4,399,104
trainable percent: 0.8826%
runtime: 39.14 seconds
train_samples_per_second: 3.577
train_steps_per_second: 3.577
train_loss: 0.10542248098188013
epoch: 20.0
output size: 33M
adapter_model.safetensors: 17,640,136 bytes
adapter reload against base model: ok
```

Gradient summary:

```text
grad logs: 140
finite grad logs: 138
nonfinite grad logs: 2
finite grad_norm min/max: 0.007082108408212662 / 6.826104640960693
```

Interpretation:

LoRA fixes the memory issue and is much faster for this local setup. The first two `nan` grad logs occur at zero LR during warmup, then gradients become finite. Unlike full fp16 fine-tuning, the LoRA run completes a longer 20-epoch test without `inf` gradients or OOM.

The tiny 7-example dataset is heavily overfit by 20 epochs, so this is a stability test rather than a useful model-quality result.

## LoRA Inference Loop Support

Goal:

Patch the rest of the local rStar loop so a LoRA policy adapter can be used without merging it into a full checkpoint. This keeps the base model path stable and passes the adapter path only where needed.

Files patched:

```text
rstar_deepthink/config.py
rstar_deepthink/llms/llm_engine.py
rstar_deepthink/llms/llms.py
rstar_deepthink/solver.py
main.py
eval.py
train/train_RM.py
train/save_rm.py
```

Interface added:

```text
main.py --policy_lora_dir <adapter_dir>
main.py --reward_model_lora_dir <adapter_dir>
eval.py --policy_lora_dir <adapter_dir>
train/train_RM.py --model_lora_path <adapter_dir>
train/save_rm.py --sft_lora_path <adapter_dir>
```

Generation now initializes vLLM with LoRA enabled when `policy_lora_dir` is set and passes a `LoRARequest` into `LLM.generate`. Reward-model scoring has the matching hook for `LLM.encode` when `reward_model_lora_dir` is set.

The eval launcher now saves adapter eval outputs under the adapter directory when `--policy_lora_dir` is provided.

RM training can now start from a base model plus LoRA SFT adapter:

```text
AutoModelForCausalLM.from_pretrained(base_model)
PeftModel.from_pretrained(base_model, adapter, is_trainable=True)
RewardModelWithValueHead(...)
```

`train/save_rm.py` supports `--sft_lora_path` and merges that adapter into the base before writing the reward-model export. This is intended for vLLM compatibility because the exported reward model directory needs full base weights plus `value_head.bin`.

Verification:

```bash
python -m py_compile main.py eval.py rstar_deepthink/config.py rstar_deepthink/solver.py rstar_deepthink/llms/llms.py rstar_deepthink/llms/llm_engine.py train/train_RM.py train/save_rm.py train/train_SFT_lora.py
```

Passed.

Host CUDA/vLLM smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -c "from vllm import LLM, SamplingParams; from vllm.lora.request import LoRARequest; llm=LLM(model='models/qwen2.5-0.5b-instruct', trust_remote_code=True, enable_lora=True, max_lora_rank=64, max_model_len=512, gpu_memory_utilization=0.5, enforce_eager=True); out=llm.generate(['Solve 1+1.'], SamplingParams(max_tokens=8, temperature=0), lora_request=LoRARequest('test', 1, 'outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed')); print(out[0].outputs[0].text[:80])"
```

Result:

```text
vLLM loaded models/qwen2.5-0.5b-instruct with LoRA enabled.
Adapter loaded from outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed.
Generation completed.
```

Next practical command for adapter-backed policy generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python main.py \
  --qaf eval_data/gsm8k_20_bootstrap.json \
  --custom_cfg config/qwen05_tree_mcts_bootstrap.yaml \
  --model_dir models/qwen2.5-0.5b-instruct \
  --policy_lora_dir outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed \
  --save_in_model outputs/qwen05_gsm8k20_tree_lora_policy/run
```

Full `main.py` smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/krkartikay/lossfunk/auto_replicate/hf_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python main.py \
  --qaf eval_data/gpt2_bootstrap_tiny.json \
  --custom_cfg config/qwen05_tiny_mcts_singlepass.yaml \
  --model_dir models/qwen2.5-0.5b-instruct \
  --policy_lora_dir outputs/qwen05_lora_sft_gsm8k20_tree_r8_20ep_noembed \
  --save_in_model outputs/qwen05_lora_main_smoke/run
```

Result:

```text
Completed successfully.
Output: outputs/qwen05_lora_main_smoke/run.jsonl
vLLM policy model loaded with policy_lora_dir set.
```

Note:

The smoke test emitted the pre-existing tokenizer warning:

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (739 > 512).
```

The run still completed. This appears tied to tokenizer metadata, not the adapter patch itself, because vLLM was configured with `max_model_len: 1536` for the run.
