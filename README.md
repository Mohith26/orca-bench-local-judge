# Can a small local model stand in for ORCA-bench's judge?

[ORCA-bench](https://github.com/ORCA-bench/ORCA-bench) scores an agent's root-cause report with an LLM judge (GPT-5.4 by default). I wanted to know whether the judge can run on a free local model, and what goes wrong if you try. This repo has the script, the raw results, and what I found. The code changes that came out of it are in [ORCA-bench PR: run the judge on any OpenAI-compatible endpoint](https://github.com/ORCA-bench/ORCA-bench/pulls?q=is%3Apr+author%3AMohith26).

## Setup

- Judge: `qwen2.5:7b` (Q4_K_M) served by Ollama 0.40 on an Apple Silicon laptop, through its OpenAI-compatible `/v1` API. No paid API calls.
- Two server configs: Ollama's defaults (`ollama_default`), and the same weights with `num_ctx 32768` (`ctx32k`).
- 8 public ORCA-bench tasks (4 with one plausible root cause, 4 with two or three), picked deterministically from the public set on the Harbor Hub (`orca-bench/orca-bench@latest`).
- For each task, three reports whose correct verdict is known without a judge:
  - `oracle`: the rubric passthrough report from `solve.py` (names every root cause and its mechanism). Correct: `rca_accuracy = 1`.
  - `symptom_only`: restates the user-facing issue and says no root cause was found. Correct: `rca_accuracy = 0`, no flag matched.
  - `wrong_cause`: the oracle report of a different task whose root cause is not in this task's answer key. Correct: `rca_accuracy = 0`, no flag matched.
- One judge call per report (no retry). "Contract violation" means the verdict did not have exactly one entry per rubric with the rubric's flag and the same number of metric/log/trace entries, which the judge prompt asks for but the scorer did not check.

## Results (48 judge calls, about 20 minutes total)

| config | condition | n | contract violations | median input tokens | `rca_accuracy` wrong (scored as-is) | verdicts crediting a root cause |
|---|---|---|---|---|---|---|
| ollama_default | oracle | 8 | 6 | 2050 | 1 | 8 |
| ollama_default | symptom_only | 8 | 5 | 2050 | 6 | 7 |
| ollama_default | wrong_cause | 8 | 4 | 2050 | 7 | 7 |
| ctx32k | oracle | 8 | 1 | 14161 | 0 | 8 |
| ctx32k | symptom_only | 8 | 4 | 8257 | 5 | 8 |
| ctx32k | wrong_cause | 8 | 3 | 8697 | 5 | 6 |

What I take from it:

1. **Ollama's default context silently truncates the judge prompt.** 22 of 24 calls reported exactly 2050 input tokens whatever the prompt length (prompts here were 5K to 94K characters; the other two were short enough to fit). The judge never sees most of the report or the rubrics, yet still returns well-formed JSON, and the old scorer turned it into a number.
2. **Malformed verdicts were being scored.** 15 of 24 default-context verdicts and 8 of 24 full-context verdicts violated the contract (missing rubric entries, verdicts for flags that are not in the rubric such as `shoppingCartFailure`, wrong cluster counts). Before the PR these were aggregated anyway; for example one verdict for a four-rubric task makes `rca_accuracy` an `all()` over one rubric. The PR rejects them (one retry, then a verifier error).
3. **Even with the full context, a 7B judge cannot be trusted on the core question.** It credited a root cause in 8 of 8 reports that name none, and in 6 of 8 reports that name the wrong one. The contract check removes some of these but not all: among well-formed verdicts, `rca_accuracy` was right on 7/7 oracle reports but only 1/4 symptom-only and 1/5 wrong-cause reports. So the useful result is negative: making the judge runnable locally is good for development and for testing the harness, but leaderboard numbers need a strong judge, and anyone swapping the judge should run checks like these first.

## What this does not show

- Nothing here measures GPT-5.4. The paper reports human agreement for the real judge (weighted kappa 0.91); I did not re-run it.
- 8 tasks and one 7B model is a small sample. The direction of finding 3 was consistent across tasks, but the exact rates will move with more tasks or a stronger local model.
- The three report types are synthetic edge cases, not real agent reports.

## Reproduce

```bash
# ORCA-bench checkout (with the PR applied) and the public tasks
uv run harbor download orca-bench/orca-bench@latest -o data
printf 'FROM qwen2.5:7b\nPARAMETER num_ctx 32768\n' > Modelfile && ollama create qwen2.5-7b-ctx32k -f Modelfile
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=unused
python local_judge_study.py --orca ../ORCA-bench --tasks data/orca-bench --model qwen2.5-7b-ctx32k --config ctx32k --out results.jsonl
python local_judge_study.py --orca ../ORCA-bench --tasks data/orca-bench --model qwen2.5:7b --config ollama_default --out results.jsonl
python summarize.py
```

`results.jsonl` has one line per judge call, including the verdict's flag names, per-rubric flag matches, token counts and the contract error text.
