# Qwen LoRA 3h Sprint Lab Notebook

Date: 2026-05-06
Window: run continuously until 07:00 IST unless interrupted.

Starting point:

```text
Committed baseline code checkpoint: 30168fc Add Qwen LoRA iteration runner

Known GSM8K greedy scores:
Qwen2.5-0.5B base: 8 / 1319 = 0.61%
Qwen2.5-0.5B-Instruct: 17 / 1319 = 1.29%
Initial clean LoRA: 89 / 1319 = 6.75%
Iteration 2 LoRA: 365 / 1319 = 27.67%
Iteration 3 LoRA: 256 / 1319 = 19.41%
```

Plan:

1. Treat iteration 2 as the current best checkpoint.
2. Try short, low-learning-rate LoRA retrains on pooled successful traces to reduce overfit/drift.
3. If those do not beat iteration 2, generate a larger bootstrap rollout from iteration 2 and train from that data.
4. Record every completed full GSM8K result here and in machine-readable sprint result files.

Result files:

```text
outputs/qwen05_lora_3h_sprint/results.jsonl
outputs/qwen05_lora_3h_sprint/results.md
```


## Experiment: pool25_lr5e5_ep10

```text
time: 2026-05-06T02:32:36
data: outputs/qwen05_lora_3h_sprint/data/sft_pool_25.json
samples: 25
epochs: 10
lr: 5e-5
lora_r: 8
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/pool25_lr5e5_ep10/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/pool25_lr5e5_ep10/adapter/gsm8k.jsonl
GSM8K score: 256 / 1319 = 0.19408642911296436 = 19.41%
```

## Experiment: iter02_5_lr5e5_ep10

```text
time: 2026-05-06T02:56:28
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 10
lr: 5e-5
lora_r: 8
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_lr5e5_ep10/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_lr5e5_ep10/adapter/gsm8k.jsonl
GSM8K score: 166 / 1319 = 0.12585291887793784 = 12.59%
```

## Experiment: iter100_from_iter2_loop

```text
time: 2026-05-06T03:43:24
rollout adapter: outputs/qwen05_lora_iterations/iter02/adapter
bootstrap set: eval_data/gsm8k_100_bootstrap.json
runner: run_iteration_qwen_lora.sh
generated SFT samples used by training: 63
epochs: 20
lr: 1e-4
adapter: outputs/qwen05_lora_3h_sprint/iter100_from_iter2_loop/iter01/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter100_from_iter2_loop/iter01/adapter/gsm8k.jsonl
GSM8K score: 206 / 1319 = 0.1561789234268385 = 15.62%
```

Notes:

- This run generated substantially more SFT traces than the earlier 5-sample iteration-2 data, but it regressed below both iteration 2 and iteration 3.
- The full eval produced a second generation step for 244 unfinished prompts and warnings for prompts that hit the 4096-token limit. The score is still usable as a full-script comparison, but the long completions and prompt-limit warnings suggest this adapter is drifting toward verbose/unfinished reasoning.
- Current best remains `outputs/qwen05_lora_iterations/iter02/adapter` at 365 / 1319 = 27.67%.

## Experiment: iter100_sft63_lr5e5_ep5

```text
time: 2026-05-06T03:56:20
data: outputs/qwen05_lora_3h_sprint/iter100_from_iter2_loop/iter01/train/sft.json
samples: 63
epochs: 5
lr: 5e-5
lora_r: 8
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/iter100_sft63_lr5e5_ep5/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter100_sft63_lr5e5_ep5/adapter/gsm8k.jsonl
GSM8K score: 192 / 1319 = 0.14556482183472327 = 14.56%
```

Notes:

- Reducing training aggressiveness on the generated 63-sample set did not recover performance; it was slightly worse than the 20-epoch, 1e-4 run.
- This points more strongly toward generated-trace quality or answer-style drift as the limiting factor for this batch, not just over-training.

## Experiment: iter02_5_r16_lr1e4_ep20

```text
time: 2026-05-06T04:34:36
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 20
lr: 1e-4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_lr1e4_ep20/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_lr1e4_ep20/adapter/gsm8k.jsonl
GSM8K score: 400 / 1319 = 0.3032600454890068 = 30.33%
```

Notes:

- New best sprint result and new best overall measured LoRA so far.
- Increasing LoRA rank from 8 to 16 on the high-quality iteration-2 traces improved GSM8K from 365 / 1319 = 27.67% to 400 / 1319 = 30.33%.
- This run was much slower than the r8 runs and produced many KV-cache preemption warnings, but its final score improved. Continue with nearby high-quality-trace variants instead of generated-data variants.

## Experiment: iter100_sft63_lr5e5_ep5

```text
time: 2026-05-06T03:56:27
data: outputs/qwen05_lora_3h_sprint/iter100_from_iter2_loop/iter01/train/sft.json
samples: 63
epochs: 5
lr: 5e-5
lora_r: 8
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/iter100_sft63_lr5e5_ep5/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter100_sft63_lr5e5_ep5/adapter/gsm8k.jsonl
GSM8K score: 192 / 1319 = 0.14556482183472327 = 14.56%
```

## Experiment: iter02_5_r16_lr1e4_ep20

```text
time: 2026-05-06T04:34:42
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 20
lr: 1e-4
lora_r: 16
lora_dropout: 0.05
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_lr1e4_ep20/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_lr1e4_ep20/adapter/gsm8k.jsonl
GSM8K score: 400 / 1319 = 0.3032600454890068 = 30.33%
```

## Experiment: iter02_5_r16_d0_lr1e4_ep20

```text
time: 2026-05-06T05:08:18
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 20
lr: 1e-4
lora_r: 16
lora_dropout: 0
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr1e4_ep20/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr1e4_ep20/adapter/gsm8k.jsonl
GSM8K score: 417 / 1319 = 0.3161485974222896 = 31.61%
```

## Experiment: iter02_5_r16_d0_lr7e5_ep20

```text
time: 2026-05-06T05:42:51
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 20
lr: 7e-5
lora_r: 16
lora_dropout: 0
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr7e5_ep20/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr7e5_ep20/adapter/gsm8k.jsonl
GSM8K score: 385 / 1319 = 0.2918877937831691 = 29.19%
```

## Experiment: iter02_5_r16_d0_lr1e4_ep30

```text
time: 2026-05-06T06:13:37
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 30
lr: 1e-4
lora_r: 16
lora_dropout: 0
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr1e4_ep30/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr1e4_ep30/adapter/gsm8k.jsonl
GSM8K score: 414 / 1319 = 0.31387414708112205 = 31.39%
```

## Experiment: iter02_5_r16_d0_lr12e5_ep20

```text
time: 2026-05-06T06:53:03
data: outputs/qwen05_lora_iterations/iter02/train/sft.json
samples: 5
epochs: 20
lr: 1.2e-4
lora_r: 16
lora_dropout: 0
adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr12e5_ep20/adapter
GSM8K output: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr12e5_ep20/adapter/gsm8k.jsonl
GSM8K score: 388 / 1319 = 0.2941622441243366 = 29.42%
```

## Sprint closeout

```text
time: 2026-05-06T06:53:11 IST
best adapter: outputs/qwen05_lora_3h_sprint/iter02_5_r16_d0_lr1e4_ep20/adapter
best GSM8K score: 417 / 1319 = 0.3161485974222896 = 31.61%
```

Summary:

- Best result came from the original high-quality iteration-2 SFT data with only 5 samples, LoRA rank 16, alpha 32, dropout 0, 20 epochs, LR 1e-4.
- Increasing rank from r8 to r16 helped substantially on the 5-sample data: previous iteration-2 adapter was 365 / 1319 = 27.67%, while r16/dropout-0 reached 417 / 1319 = 31.61%.
- Nearby LR/epoch variants did not beat the best: LR 7e-5 scored 29.19%, LR 1.2e-4 scored 29.42%, and 30 epochs at LR 1e-4 scored 31.39%.
- Generated-data variants regressed: the 63-sample bootstrap-derived set scored 15.62% with the iteration runner and 14.56% with a gentler sprint SFT run.
- The next useful direction is likely better filtering/selection of generated traces, not simply more generated traces or more aggressive SFT.
