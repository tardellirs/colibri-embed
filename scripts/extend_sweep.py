#!/usr/bin/env python3
"""
Extend the alpha sweep for the top5 soup beyond the 0.1-0.3 cap.
Reuses the already-evaluated soup ingredients (no re-eval of the 10).

  1. rebuild top5 soup = uniform avg of its 5 member checkpoints
  2. sweep alpha on mean_21 (CHEAP: excludes the slow Quati retrieval)
  3. full mean_22 (with Quati) ONLY on the best alpha
  4. push the true winner to HF

Usage:
  COLIBRI_BASE=tardellirs/embeddinggemma-pt-br-64k-retrim-stack-test HF_TOKEN=... \
    python extend_sweep.py --ckpts-dir /root/ckpt_v2/ckpts \
      --members ckpt_v2_step_528 ckpt_v2_step_176 ckpt_v2_epoch_1 ckpt_v2_step_440 ckpt_v2_step_88 \
      --alphas 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
      --hub-repo tardellirs/colibri-v2-distill --out /root/v2_extend_results
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interpolate_eval import eval_mteb, interpolate, BASE  # noqa

TOKEN = os.getenv("HF_TOKEN")
TARGET = ["BRTaxQAR", "JurisTCUClusteringP2P", "SciELOClusteringP2P",
          "StackoverflowPtClustering", "WikipediaPTCategoriesClusteringP2P"]


def load_sd(path):
    m = SentenceTransformer(path, token=TOKEN)
    sd = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
    del m
    return sd


def avg_sds(sds):
    out, ref = {}, sds[0]
    for k in ref:
        out[k] = sum(sd[k] for sd in sds) / len(sds) if ref[k].dtype.is_floating_point else ref[k]
    return out


def m21(model, bs):
    return eval_mteb(model, bs, exclude=["Quati"])  # excludes slow Quati retrieval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts-dir", required=True)
    ap.add_argument("--members", nargs="+", required=True,
                    help="soup member names = <parent-of-ckpts>_<basename> (e.g. ckpt_v2_step_528)")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--out", default="./v2_extend_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # map member display-name -> its checkpoint dir
    def nm(d):
        return f"{os.path.basename(os.path.dirname(os.path.dirname(d)))}_{os.path.basename(d)}"
    dirs = {nm(d): d for d in sorted(glob.glob(os.path.join(args.ckpts_dir, "*")))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "modules.json"))}
    missing = [m for m in args.members if m not in dirs]
    if missing:
        print(f"ERROR: members not found on disk: {missing}\n  available: {list(dirs)}")
        sys.exit(1)

    print(f"rebuilding top5 soup from: {args.members}", flush=True)
    soup_sd = avg_sds([load_sd(dirs[m]) for m in args.members])
    base_model = SentenceTransformer(BASE, token=TOKEN)

    # cheap sweep on mean_21
    sweep = {}
    for a in args.alphas:
        m, s = m21(interpolate(base_model, soup_sd, a)[0], args.batch_size)
        sweep[str(a)] = {"mean21": m, "scores": s}
        tgt = " ".join(f"{t}={s.get(t, float('nan')):.3f}" for t in TARGET)
        print(f"  alpha={a}: mean_21={m:.4f}  {tgt}", flush=True)
        json.dump(sweep, open(os.path.join(args.out, "extend_m21.json"), "w"), indent=2)

    best_a = max(sweep, key=lambda k: sweep[k]["mean21"])
    print(f"\nBEST alpha={best_a} on mean_21={sweep[best_a]['mean21']:.4f}", flush=True)

    # full mean_22 (with Quati) only on best alpha
    wm, _ = interpolate(base_model, soup_sd, float(best_a))
    mfull, sfull = eval_mteb(wm, args.batch_size)  # no exclude -> full 22
    print(f"\n=== EXTEND RESULT ===\n  top5 @ alpha={best_a}: mean_22={mfull:.4f}", flush=True)
    for t in sorted(sfull):
        print(f"    {t}={sfull[t]:.3f}")
    json.dump({"best_alpha": best_a, "mean22": mfull, "scores": sfull, "m21_sweep": sweep},
              open(os.path.join(args.out, "extend_full22.json"), "w"), indent=2)

    if args.hub_repo:
        try:
            wm.push_to_hub(args.hub_repo, token=TOKEN, private=True, exist_ok=True,
                           commit_message=f"top5 soup @ alpha={best_a} mean_22={mfull:.4f} (extended sweep)")
            print(f"  [hub] pushed extended winner -> {args.hub_repo}", flush=True)
        except Exception as e:
            print(f"  [hub] push failed: {e}", flush=True)
    print("=== EXTEND-SWEEP DONE ===", flush=True)


if __name__ == "__main__":
    main()
