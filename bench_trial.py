#!/usr/bin/env python3
"""
bench_trial.py -- one benchmark round under a declared prompt strategy.

    python3 ~/bin/bench_trial.py --strategy invariant-first-v1 --backend ollama \
        --model qwen2.5-coder:7b --tasks backlog,claims,gaps [--k 1] [--events]

The same round osiris_bench.py runs -- same tasks, hidden tests, candidate
loop, gates, sandbox, write-ahead ledger and artifacts -- with one declared
difference: each task prompt starts with the strategy's fixed preamble
(prompt_strategies.py). osiris_bench.py itself is not modified, so
runner_sha256 is unchanged and the round stays comparable with a baseline
round of the same model. The start entry records strategy and
strategy_sha256; protege.py keeps strategy rounds out of the mentor
scorecard, and trials.py compares them with their control. Exit codes as
osiris_bench.py (0 ran, 1 recording failed, 2 bad args, 3 no sandbox,
130 interrupted).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import osiris_bench as bench  # noqa: E402
import prompt_strategies  # noqa: E402


class TrialRunLog(bench.RunLog):
    """RunLog whose start entry also declares the strategy (and optional verifier mode)."""

    def __init__(self, out_dir, tasks, k, backend, model, strategy, verifier="baseline"):
        self.out_dir, self.ledger = out_dir, bench._load_ledger(out_dir)
        self.run_id = "bench-" + bench.time.strftime("%Y%m%dT%H%M%SZ", bench.time.gmtime()) + "-" + os.urandom(3).hex()
        self.planned = [t["id"] for t in tasks]
        body = {
            "run_id": self.run_id,
            "commit": bench._git("rev-parse", "HEAD") or None,
            "dirty": bool(bench._git("status", "--porcelain", "--untracked-files=no")),
            "suite_sha256": bench.suite_sha256(), "runner_sha256": bench.runner_sha256(),
            "backend": backend, "model": model, "k": k, "planned": self.planned,
            "strategy": strategy, "strategy_sha256": prompt_strategies.sha256(strategy),
        }
        if verifier != "baseline":
            body["verifier"] = verifier
        self.start = bench._append_durably(self.ledger, "bench_run_start", body)


def _run_l5_suite(tasks, k, backend, model, strategy, on_task=None):
    """Closed-loop L5 trial mode: runs `production_synth.run_production_synthesis` (delimited/JSON
    transport + AST contract gate + isolated L5 Critic) and scores every attempt against the
    untouched hidden `acceptance.py` suite."""
    from osiris_cli import production_synth

    osiris_mod = bench._load_osiris()
    only = None if backend == "auto" else backend
    results = {}
    interrupted = None
    for idx, task in enumerate(tasks, start=1):
        print(f"[bench-l5] task {idx}/{len(tasks)}: {task['id']}")
        bench.emit("task_start", task=task["id"], index=idx, total=len(tasks))
        prompts_used = []
        backends_used = []

        def propose_raw(full_prompt: str) -> str:
            prompt_with_strat = prompt_strategies.apply(strategy, full_prompt)
            raw = osiris_mod._synthesize(prompt_with_strat, structured=False, only=only, ollama_model=model)
            prompts_used.append(prompt_with_strat)
            backends_used.append(osiris_mod._last_backend.get("name") or backend)
            return raw

        def on_attempt(att):
            print(f"[bench-l5] {task['id']} candidate {att['n']}/{k}: L5 check "
                  f"{'pass' if att['ok'] else 'fail -- ' + bench._tail(att['reason'])}")
            bench.emit("attempt", task=task["id"], n=att["n"], k=k, own_ok=bool(att["ok"]))

        try:
            outcome = production_synth.run_production_synthesis(
                root=os.getcwd(),
                target_rel_path=f"{task['module']}.py",
                story_text=task["spec"],
                propose_raw_fn=propose_raw,
                sandbox_test_fn=osiris_mod._run_sandboxed_test,
                existing_source=task.get("existing"),
                k=k,
                use_delimited_prompt=True,
                enforce_target_path=True,
                run_repo_pytest=False,
                on_attempt=on_attempt,
            )
        except KeyboardInterrupt:
            interrupted = task["id"]
            break

        rows, texts = [], {}
        for att, bname, ptext in zip(outcome.attempts, backends_used, prompts_used):
            row = {"n": att["n"], "backend": bname, "own_ok": bool(att["ok"])}
            row["prompt_sha256"] = bench._keep(texts, ptext)
            cand = att.get("candidate")
            if cand is None:
                hidden_ok, hidden_reason = False, att["reason"]
            else:
                hidden_ok, hidden_reason = bench.score(task, cand["module"], osiris_mod._run_sandboxed_test)
                row["module_sha256"] = bench._keep(texts, cand["module"])
                row["test_sha256"] = bench._keep(texts, cand["test"])
            row.update(hidden_ok=hidden_ok, hidden_reason=hidden_reason)
            rows.append(row)

        delivered = bool(outcome.winner) and bool(rows) and rows[-1]["hidden_ok"]
        res = {
            "pass_at_1": rows[0]["hidden_ok"] if rows else False,
            "delivered": delivered,
            "oracle_at_k": any(r["hidden_ok"] for r in rows),
            "false_confidence": sum(1 for r in rows if r["own_ok"] and not r["hidden_ok"]),
            "attempts": rows,
            "_texts": texts,
        }
        status = "PASS" if delivered else ("NOTHING PROPOSED" if not outcome.winner else "FAIL (L5 passed)")
        print(f"[bench-l5] {task['id']}: delivered {status}"
              + ("" if delivered else f" -- hidden: {rows[-1]['hidden_reason'][:150]}"))
        results[task["id"]] = res
        if on_task:
            on_task(task["id"], res)
        bench.emit("task_done", task=task["id"], delivered=delivered,
                   false_confidence=res["false_confidence"])
    return results, interrupted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One benchmark round under a declared prompt strategy.")
    ap.add_argument("--strategy", required=True, help=f"one of: {', '.join(sorted(prompt_strategies.STRATEGIES))}")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--backend", default="ollama", choices=bench.BACKENDS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--verifier", default="baseline", choices=("baseline", "l5"),
                    help="baseline (frozen JSON-schema runner) or l5 (delimited + AST contract + L5 Critic)")
    ap.add_argument("--out", default=bench.DEFAULT_OUT)
    ap.add_argument("--events", action="store_true")
    args = ap.parse_args(argv)
    bench._events["on"] = args.events
    try:
        prompt_strategies.get(args.strategy)
        tasks = bench.load_tasks([t for t in args.tasks.split(",") if t] or None)
    except (KeyError, ValueError) as e:
        print(f"[!] {e}", file=sys.stderr)
        return 2
    if args.k < 1 or (args.model and args.backend != "ollama"):
        print("[!] --k must be >= 1; --model only applies to --backend ollama", file=sys.stderr)
        return 2
    unavailable = bench._load_osiris()._sandbox_probe()
    if unavailable:
        print(f"[!] Sandbox unavailable in this environment: {unavailable}")
        return 3
    model = args.model or (bench._load_osiris().CODE_MODEL if args.backend == "ollama" else None)
    baseline_prompt = bench.prompt_for
    print(f"Trial: strategy {args.strategy} ({prompt_strategies.sha256(args.strategy)[:12]}) · "
          f"verifier={args.verifier} · {args.backend}{f' ({model})' if model else ''} · k={args.k}")
    try:
        log = TrialRunLog(args.out, tasks, args.k, args.backend, model, args.strategy, verifier=args.verifier)
        print(f"Run {log.run_id}: each task is recorded as soon as it finishes")
        bench.emit("run_start", run_id=log.run_id, backend=args.backend, model=model, k=args.k,
                   tasks=[t["id"] for t in tasks], strategy=args.strategy)
        if args.verifier == "l5":
            results, cut = _run_l5_suite(tasks, args.k, args.backend, model, args.strategy, on_task=log.task)
        else:
            bench.prompt_for = lambda task: prompt_strategies.apply(args.strategy, baseline_prompt(task))
            try:
                results, cut = bench.run_suite(tasks, args.k, args.backend, model, on_task=log.task)
            finally:
                bench.prompt_for = baseline_prompt
        log.end(results, cut)
        bench.emit("run_end", status="partial" if cut else "complete", completed=len(results),
                   planned=len(tasks), interrupted_during=cut)
    except OSError as e:
        print(f"[!] Recording failed: {e} -- run left without an end entry", file=sys.stderr)
        return 1
    s = bench.summarize(results)
    print(f"\ndelivered {s['delivered']}/{s['tasks']}   false confidence {s['false_confidence']}   "
          f"({args.strategy}) -- {'PARTIAL' if cut else 'complete'}: {log.run_id}")
    return 130 if cut else 0


if __name__ == "__main__":
    sys.exit(main())
