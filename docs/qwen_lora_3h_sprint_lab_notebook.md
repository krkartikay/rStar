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

