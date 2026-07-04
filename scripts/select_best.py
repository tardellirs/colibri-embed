#!/usr/bin/env python3
"""
Select the best training checkpoint by FaqBacenRetrieval (proxy for mean_22).

FaqBacenRetrieval has Spearman 0.867 with MTEB(por) mean_22 — a far better
selector than Assin2STS (0.706). It is test-only, so evaluating exp1/exp2
(which do NOT train on it) measures honest retrieval generalization.

Usage:
  python select_best.py --ckpt-dir /root/saved_ckpts [--run-mteb]

Steps:
  1. Score every checkpoint in --ckpt-dir on FaqBacenRetrieval (fast).
  2. Print the ranked table.
  3. Optionally run the full MTEB(por) 22-task suite on the winner.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path

import mteb
import mteb_pt.register  # noqa: F401 — registers custom PT tasks
from sentence_transformers import SentenceTransformer


def score_task(model_path: str, task_name: str, hf_token: str | None, batch_size: int) -> float:
    model = SentenceTransformer(model_path, token=hf_token)
    task = mteb.get_task(task_name)
    results = mteb.evaluate(
        model, tasks=[task],
        overwrite_strategy="always",
        encode_kwargs={"batch_size": batch_size},
        prediction_folder=f"/tmp/{task_name}_{Path(model_path).name}",
        raise_error=False,
    )
    # MTEB 2.16+: evaluate() returns ModelResult with .task_results[i].get_score()
    r = results[0] if isinstance(results, list) else results
    try:
        return r.task_results[0].get_score()
    except Exception as e:
        print(f"    (score extraction failed: {e})")
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/root/saved_ckpts")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Extra checkpoint/model paths to also score (e.g. best_model dir)")
    ap.add_argument("--task", default="FaqBacenRetrieval",
                    help="MTEB task to use as selector (default FaqBacenRetrieval)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="If set, only score checkpoints whose basename contains one of these substrings")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--run-mteb", action="store_true",
                    help="Run full MTEB(por) on the winning checkpoint")
    args = ap.parse_args()

    hf_token = os.getenv("HF_TOKEN")

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*")))
    ckpts = [c for c in ckpts if os.path.isdir(c)] + list(args.extra)
    if args.only:
        ckpts = [c for c in ckpts if any(o in os.path.basename(c) for o in args.only)]
    if not ckpts:
        print(f"No checkpoints found in {args.ckpt_dir}")
        sys.exit(1)

    print(f"Scoring {len(ckpts)} checkpoints on {args.task} …\n")
    import time
    scores = {}
    for c in ckpts:
        try:
            t0 = time.time()
            s = score_task(c, args.task, hf_token, args.batch_size)
            dt = time.time() - t0
        except Exception as e:
            print(f"  {os.path.basename(c)}: FAILED ({e})")
            continue
        scores[c] = s
        print(f"  {os.path.basename(c)}: {args.task} = {s:.4f}  ({dt:.0f}s)")

    if not scores:
        print("No checkpoint scored successfully.")
        sys.exit(1)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n=== Ranked by {args.task} ===")
    for c, s in ranked:
        print(f"  {s:.4f}  {os.path.basename(c)}")

    winner, win_score = ranked[0]
    print(f"\nWINNER: {os.path.basename(winner)}  ({args.task} {win_score:.4f})")

    with open(os.path.join(args.ckpt_dir, f"_{args.task}_selection.json"), "w") as f:
        json.dump({"task": args.task, "winner": winner, "score": win_score,
                   "all": {os.path.basename(c): s for c, s in scores.items()}}, f, indent=2)

    if args.run_mteb:
        print(f"\nRunning full MTEB(por) on winner …")
        os.system(f"python3 run_mtebpt.py --model {winner} "
                  f"--output ./mteb_winner --batch-size {args.batch_size}")


if __name__ == "__main__":
    main()
