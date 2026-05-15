# Convert rStar-Math to SWE-bench-Style Coding Search

## Summary
- Add a coding-problem mode that keeps the existing MCTS/beam-search loop but replaces math prompts, Python execution, boxed-answer evaluation, and Q/A schemas with SWE-bench-style repository tasks.
- Target SWE-bench JSONL first, normalized into an internal `CodingTask`.
- Use git-based workspace snapshots so each MCTS node owns an isolated repo checkout.
- Add a development LLM backend that generates MCTS traces through the OpenAI Responses API instead of VLLM, with `gpt-5.4-nano` as a configurable placeholder model name.

## Key Changes
- Add task/config surface:
  - `task_type: "math" | "coding"`
  - `llm_backend: "vllm" | "openai_api"`
  - `api_model: "gpt-5.4-nano"`
  - `workspace_root`, `keep_workspaces`, `bash_timeout_seconds`, `bash_output_max_chars`
  - SWE-bench loader maps `instance_id`, `repo`, `base_commit`, `problem_statement`, and test metadata into `CodingTask`.
- Add OpenAI dev backend:
  - Refactor `llm_engine`/`llm_generate` behind a provider interface returning VLLM-shaped output objects.
  - For `openai_api`, call the Responses API with the configured model and prompt.
  - Emulate `n_generate_sample` with multiple API calls per prompt.
  - Preserve downstream fields: `.outputs`, `.text`, `.stop_reason`, and `.value_estimate`.
  - API dev mode supports policy generation only; it does not require policy logits.
  - Disable local PRM/value-head scoring by default in API dev mode.
- Add bash execution and node workspaces:
  - Replace `python_interpreter` with `bash` in coding mode.
  - Execute `/bin/bash -lc <command>` inside the node workspace.
  - Capture stdout, stderr, exit code, timeout, and truncated output.
  - Root node starts from a clean checkout pinned to `base_commit`; child nodes get isolated snapshots from their parent.
  - Terminal coding nodes submit `git diff` from their workspace as the candidate patch.
- Add coding prompts/parsing:
  - Add `prompt_wrap: "swe"`.
  - Use `<bash>...<end_of_bash>` for actions and `<finish>...<end_of_finish>` for submission.
  - Keep math mode unchanged.
- Add coding evaluation:
  - On terminal nodes, run the task test command in that node workspace.
  - Passing tests get `positive_reward`; failing tests get `negative_reward`.
  - Output JSONL includes `instance_id`, tree state, candidate patch, test exit code, and test output summary.

## Test Plan
- Unit tests:
  - SWE-bench row normalizes into `CodingTask`.
  - OpenAI API backend converts API responses into VLLM-compatible request/completion objects.
  - Bash parser extracts `<bash>` and `<finish>`.
  - Child workspace edits do not affect parent or sibling workspaces.
- Integration tests:
  - Use a tiny local fixture repo with one failing test and one known fix.
  - Run with `llm_backend: "openai_api"` and small settings: `iterations <= 2`, `n_generate_sample <= 2`, `max_depth <= 4`.
  - Verify MCTS trace generation writes a result JSONL and terminal patches are replayable.
  - Verify passing terminal nodes receive positive reward and failing nodes receive negative reward.
- Development constraints:
  - Use the `rstar` pyenv environment.
  - Do not run full SWE-bench evals during development; run 1-20 problems only.
  - Use non-sandbox mode for GPU-backed VLLM runs.

## Assumptions
- V1 targets SWE-bench JSONL, with a normalized internal task object for later adapters.
- V1 uses git worktree/copy snapshots, not Docker or overlayfs.
- API dev mode is for trace-generation and eval smoke tests, not final benchmark numbers.
- The policy generation path does not need local logits/logprobs.
- Local model internals are only needed for trained reward-model/value-head scoring and fine-tuning.
- Test-based reward is the source of truth for coding correctness in v1.
