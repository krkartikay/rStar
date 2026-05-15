# Qwen Chat Template rStar Lab Notebook

Date: 2026-05-15

## Goal

Make the local rStar inference path use Qwen's tokenizer chat template for Qwen instruct models, then rerun GSM8K evaluation to check whether the very low `Qwen2.5-0.5B-Instruct` score was caused by prompt formatting rather than model quality.

## Starting Point

Earlier local notes recorded these greedy GSM8K scores:

```text
Qwen2.5-0.5B base: 8 / 1319 = 0.61%
Qwen2.5-0.5B-Instruct: 17 / 1319 = 1.29%
Initial clean LoRA: 89 / 1319 = 6.75%
Best LoRA sprint adapter: 417 / 1319 = 31.61%
```

The `Qwen2.5-0.5B-Instruct` score was suspiciously low relative to published Qwen-family GSM8K benchmarks.

## Diagnosis

The original rStar greedy eval path used `config/sft_eval_greedy.yaml`, which renders prompts through `rstar_prompt_wrap` and `rstar_deepthink/few_shots/sft_prompt.json`.

That prompt was not Qwen chat format:

```text
<|user|>:
{input}
<|assistant|>: Let's think step by step and solve the problem with code.
```

This is incompatible with Qwen's actual chat template, which renders messages with `<|im_start|>...<|im_end|>` blocks.

Separately, a direct Transformers sanity check with Qwen's chat template showed that decoder-only batched generation must use left padding:

```text
20-example slice, right padding: 1 / 20 = 5%
20-example slice, left padding: 6 / 20 = 30%
100-example slice, left padding: 37 / 100 = 37%
```

## Code Changes

Added optional chat-template rendering to `rstar_prompt_wrap`:

```text
rstar_deepthink/config.py
rstar_deepthink/agents/utils.py
```

New config keys:

```yaml
use_chat_template: true
chat_system_prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
```

Added Qwen-specific prompt/config files:

```text
rstar_deepthink/few_shots/sft_prompt_qwen_chat.json
rstar_deepthink/few_shots/mcts_prompt_qwen_chat.json
config/sft_eval_greedy_qwen_chat.yaml
config/qwen_chat_mcts_bootstrap.yaml
```

Also added `--custom_cfg` to `eval.py` so full task evaluation can select the Qwen-chat config.

## Environment

Use the repo's original environment for rStar runs:

```bash
~/.pyenv/versions/rstar/bin/python
```

This has:

```text
torch 2.5.1+cu124
vllm 0.6.6.post1
transformers 4.45.2
```

GPU is visible when commands are run with escalated access:

```text
NVIDIA GeForce RTX 3070
```

## GSM8K rStar Qwen-Chat Smoke Run

Command:

```bash
~/.pyenv/versions/rstar/bin/python main.py \
  --custom_cfg config/sft_eval_greedy_qwen_chat.yaml \
  --qaf eval_data/gsm8k_20_bootstrap.json \
  --model_dir models/qwen2.5-0.5b-instruct \
  --save_in_model outputs/qwen_chat_template_rstar/gsm8k20_greedy_fixed/run
```

Result:

```text
output: outputs/qwen_chat_template_rstar/gsm8k20_greedy_fixed/run.jsonl
score: 8 / 20 = 40.00%
```

Important parser fix:

The first Qwen-chat smoke run produced boxed answers, but rStar scored `0 / 20` because `extract_math_answer` truncated the response at the first `.\n` before extracting the boxed answer. The extractor now prefers the last `\boxed{...}` answer whenever `boxed` appears.

## Full GSM8K rStar Qwen-Chat Greedy Run

Command:

```bash
~/.pyenv/versions/rstar/bin/python main.py \
  --custom_cfg config/sft_eval_greedy_qwen_chat.yaml \
  --qaf eval_data/GSM8K_test.json \
  --model_dir models/qwen2.5-0.5b-instruct \
  --save_in_model outputs/qwen_chat_template_rstar/gsm8k_full_greedy/run
```

Generation completed successfully:

```text
output: outputs/qwen_chat_template_rstar/gsm8k_full_greedy/run.jsonl
rows: 1319
generation time: about 3.6 minutes
```

Scoring:

```bash
~/.pyenv/versions/rstar/bin/python eval_output.py \
  --file_path outputs/qwen_chat_template_rstar/gsm8k_full_greedy/run.jsonl
```

Result:

```text
641 / 1319 = 0.48597422289613346 = 48.60%
```

This replaces the earlier suspicious `17 / 1319 = 1.29%` result. The low result was caused by using a non-Qwen prompt format plus brittle answer extraction, not by the instruct model being intrinsically that weak.

## MATH / math500

Started a `math500` greedy generation with the same Qwen-chat rStar config, but the user asked to skip MATH/math500 follow-up. The generation file exists and has 500 rows:

```text
outputs/qwen_chat_template_rstar/math500_greedy/run.jsonl
```

It was intentionally not scored or used in the conclusions here.

### Qwen-chat MCTS LoRA iteration 1

```text
rollout policy: models/qwen2.5-0.5b-instruct
trained adapter: outputs/qwen_chat_mcts_lora_iterations_train100_64traj/iter01/adapter
SFT samples: 10
{"correct": null, "total": null, "pass1": null}

```
