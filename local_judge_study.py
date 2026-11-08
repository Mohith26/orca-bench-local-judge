"""Run ORCA-bench's LLM judge on a local open-weight model and check its verdicts.

For each sampled public task, three reports with a known correct verdict are judged:
  oracle        - the rubric passthrough report from solve.py (names every root cause)
  symptom_only  - restates the user-facing symptom, names no root cause
  wrong_cause   - the oracle report of a different task whose flags don't overlap
Each judge call is made once (no retry). We record whether the verdict lines up with
the rubrics (validate_judge_verdicts) and what the unpatched scorer would have reported.

Usage: python local_judge_study.py --orca ../ORCA-bench --tasks ../data/orca-bench \
           --model qwen2.5:7b --config default --out results.jsonl [--n-single 4 --n-multi 4]
Env: OPENAI_BASE_URL (e.g. http://localhost:11434/v1), OPENAI_API_KEY (any string for Ollama)
"""
import argparse, asyncio, glob, json, os, sys, time, tomllib


def load_task(d):
    meta = tomllib.load(open(os.path.join(d, "task.toml"), "rb"))["metadata"]
    rubrics = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "tests/rubrics/*.json")))]
    return meta, rubrics


def sample_tasks(root, n_single, n_multi):
    """Deterministic sample: sorted task ids, distinct authored flags, incident tasks only."""
    single, multi, seen = [], [], set()
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        meta, rubrics = load_task(d)
        if not rubrics:
            continue
        flags = tuple(r["feature_flag"] for r in rubrics)
        key = (len(rubrics) == 1, flags[0])
        if key in seen:
            continue
        if len(rubrics) == 1 and len(single) < n_single:
            single.append(d); seen.add(key)
        elif 2 <= len(rubrics) <= 3 and len(multi) < n_multi:
            multi.append(d); seen.add(key)
    return single + multi


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orca", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-single", type=int, default=4)
    ap.add_argument("--n-multi", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.orca))
    import check_prediction as cp
    from solve import generate_report
    from openai import AsyncOpenAI

    client = AsyncOpenAI(max_retries=0, timeout=900)
    tasks = sample_tasks(a.tasks, a.n_single, a.n_multi)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["config"], r["task"], r["condition"]))
    all_meta = {d: load_task(d) for d in tasks}
    calls = 0
    for d in tasks:
        meta, rubrics = all_meta[d]
        flags = {r["feature_flag"] for r in rubrics}
        oracle, _ = await generate_report(None, rubrics, model=None, effort=None)
        # wrong cause: first other sampled-or-not task whose flags are disjoint
        wrong = None
        for d2 in sorted(glob.glob(os.path.join(a.tasks, "*"))):
            m2, r2 = load_task(d2)
            if len(r2) == 1 and r2[0]["feature_flag"] not in flags:
                wrong, _ = await generate_report(None, r2, model=None, effort=None)
                wrong_flag = r2[0]["feature_flag"]
                break
        symptom = (
            "# Incident report\n\n"
            f"Users reported that {meta['user_facing_issue']} ({meta['reported_styled']}).\n\n"
            "I confirmed elevated frontend error rates and slower responses around the reported "
            "time. I was not able to determine which component or change caused it.\n"
        )
        for cond, report in (("oracle", oracle), ("symptom_only", symptom), ("wrong_cause", wrong)):
            tid = os.path.basename(d)
            if (a.config, tid, cond) in done or calls >= a.max_calls:
                continue
            calls += 1
            prompt = cp.build_judge_prompt(rubrics, report)
            rec = {"config": a.config, "model": a.model, "task": tid, "qid": meta["qid"],
                   "condition": cond, "n_rubrics": len(rubrics), "flags": sorted(flags),
                   "prompt_chars": len(prompt)}
            if cond == "wrong_cause":
                rec["wrong_flag"] = wrong_flag
            t0 = time.time()
            try:
                raw, _, usage = await cp.async_call_llm_judge(
                    client, prompt, model=a.model, reasoning_effort=None,
                    output_schema=cp.build_judge_output_schema(rubrics))
                rec["seconds"] = round(time.time() - t0, 1)
                rec["input_tokens"] = (usage or {}).get("input_tokens") or (usage or {}).get("prompt_tokens")
                parsed = cp.parse_judge_response(raw)
                agg = cp.aggregate_judge_response(parsed)
                rec["unpatched_rca_accuracy"] = agg["rca_accuracy"]
                rec["unpatched_hallucinate_any"] = agg["hallucinate_any"]
                rec["unpatched_reward"] = (agg["rca_depth"] or 0) / 3
                rec["verdict_flags"] = [v.get("feature_flag") for v in parsed.get("rubrics", [])]
                rec["flag_matches"] = [bool(v.get("feature_flag_match")) for v in parsed.get("rubrics", [])]
                try:
                    cp.validate_judge_verdicts(parsed, rubrics)
                    rec["contract_ok"] = True
                except cp.JudgeContractError as e:
                    rec["contract_ok"] = False
                    rec["contract_error"] = str(e)
            except Exception as e:  # parse errors and API errors are data too
                rec["seconds"] = round(time.time() - t0, 1)
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            truth_rca = cond == "oracle"
            rec["truth_rca_accuracy"] = truth_rca
            if "unpatched_rca_accuracy" in rec:
                rec["unpatched_rca_correct"] = bool(rec["unpatched_rca_accuracy"]) == truth_rca
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(a.config, tid, cond, rec.get("seconds"), rec.get("contract_ok"), rec.get("unpatched_rca_correct"), rec.get("error", ""), flush=True)


asyncio.run(main())
