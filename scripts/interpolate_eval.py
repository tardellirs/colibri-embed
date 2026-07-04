#!/usr/bin/env python3
"""
A1 — interpolate a fine-tuned checkpoint with the base and eval each alpha on MTEB(por).

theta = (1-alpha)*base + alpha*ft   for alpha in the sweep.
Weight interpolation is CPU/seconds; the cost is one MTEB eval per alpha.

Usage:
  HF_TOKEN=... python interpolate_eval.py --ckpt ./ckpt_full/final_model \\
      --alphas 0.3 0.5 0.7 --out ./a1_results
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import torch
from sentence_transformers import SentenceTransformer

BASE = os.getenv("COLIBRI_BASE", "tardellirs/embeddinggemma-pt-br")  # merge target (retrim-stack for v2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def eval_mteb(model, batch_size=256, task_filter=None, exclude=None):
    import mteb, mteb_pt.register  # noqa
    _EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
    tasks = [cls() for cls in mteb_pt.register._TASKS_TO_REGISTER
             if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    if task_filter:
        tasks = [t for t in tasks if t.metadata.name in set(task_filter)]
    if exclude:
        tasks = [t for t in tasks if t.metadata.name not in set(exclude)]
    res = mteb.evaluate(model, tasks=tasks, overwrite_strategy="always",
                        encode_kwargs={"batch_size": batch_size},
                        prediction_folder="/tmp/a1preds", raise_error=False)
    mr = res[0] if isinstance(res, list) else res
    scores = {}
    for tr in getattr(mr, "task_results", []) or []:
        try:
            scores[tr.task_name] = tr.get_score()
        except Exception:
            pass
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return mean, scores


def interpolate(base_model, ft_sd, alpha):
    """Return a fresh SentenceTransformer with theta=(1-a)*base + a*ft."""
    m = SentenceTransformer(BASE, token=os.getenv("HF_TOKEN"))
    sd = m.state_dict()
    base_sd = base_model.state_dict()
    n = 0
    for k in sd:
        if k in ft_sd and k in base_sd and sd[k].shape == ft_sd[k].shape and sd[k].dtype.is_floating_point:
            b = base_sd[k].float(); f = ft_sd[k].float().to(b.device)  # device-safe
            sd[k] = ((1 - alpha) * b + alpha * f).to(base_sd[k].dtype)
            n += 1
    m.load_state_dict(sd)
    return m, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint path")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--out", default="./a1_results")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="proxy task subset (fast); default = all 22")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="drop these tasks (e.g. Quati -> mean_21 sweep)")
    ap.add_argument("--eval-endpoints", action="store_true",
                    help="also eval base (alpha=0) and ft (alpha=1)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    token = os.getenv("HF_TOKEN")

    print("Loading base + ft state dicts...")
    base_model = SentenceTransformer(BASE, token=token)
    ft_model = SentenceTransformer(args.ckpt, token=token)
    ft_sd = ft_model.state_dict()
    del ft_model

    alphas = list(args.alphas)
    if args.eval_endpoints:
        alphas = [0.0] + alphas + [1.0]

    results = {}
    for a in alphas:
        print(f"\n=== alpha={a} ===")
        if a == 0.0:
            m = SentenceTransformer(BASE, token=token)
        else:
            m, n = interpolate(base_model, ft_sd, a)
            print(f"  interpolated {n} float tensors")
        mean, scores = eval_mteb(m, args.batch_size, task_filter=args.tasks, exclude=args.exclude)
        results[str(a)] = {"mean": mean, "scores": scores}
        print(f"  alpha={a}: mean_{len(scores)} = {mean:.4f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items())))
        with open(os.path.join(args.out, "a1_sweep.json"), "w") as f:
            json.dump(results, f, indent=2)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n=== A1 SWEEP SUMMARY ===")
    for a in sorted(results, key=float):
        print(f"  alpha={a}: {results[a]['mean']:.4f}")
    best = max(results, key=lambda k: results[k]["mean"])
    print(f"\nBEST alpha={best}: {results[best]['mean']:.4f}  (base=0.6598)")


if __name__ == "__main__":
    main()
