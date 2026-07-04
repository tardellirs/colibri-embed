#!/usr/bin/env python3
"""Run the official MTEB(por, v2) 22-task suite on a local SentenceTransformer model."""
import os, sys, time
import mteb
import mteb_pt.register  # registers custom tasks into the mteb registry
from sentence_transformers import SentenceTransformer

_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
_PRIORITY = {"Retrieval": 0, "Reranking": 1, "Clustering": 2}


def v2_tasks():
    tasks = [cls() for cls in mteb_pt.register._TASKS_TO_REGISTER
             if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    tasks.sort(key=lambda t: _PRIORITY.get(t.metadata.type, 9))
    return tasks


def main(model_path: str, output: str, batch_size: int) -> None:
    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading: {model_path}")
    model = SentenceTransformer(model_path, token=hf_token)

    tasks = v2_tasks()
    print(f"Tasks: {len(tasks)}")

    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks=tasks,
        overwrite_strategy="always",  # local models share no_model_name cache → force fresh
        encode_kwargs={"batch_size": batch_size},
        prediction_folder=output,
        raise_error=False,
    )
    elapsed = (time.time() - t0) / 60

    # Collect scores. MTEB 2.16+: evaluate() → ModelResult with .task_results
    scores = {}
    model_results = results if isinstance(results, list) else [results]
    for mr in model_results:
        task_results = getattr(mr, "task_results", None) or [mr]
        for tr in task_results:
            try:
                name = tr.task_name
                scores[name] = tr.get_score()
            except Exception:
                pass

    print("\n--- Per-task scores ---")
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v:.4f}")
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\nmean_{len(scores)} = {mean:.4f}  ({elapsed:.1f} min)")

    import json, pathlib
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output, "_summary.json"), "w") as f:
        json.dump({"model": model_path, "mean": mean, "n_tasks": len(scores),
                   "scores": scores}, f, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="./mtebpt_results")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    main(args.model, args.output, args.batch_size)
