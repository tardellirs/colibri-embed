#!/usr/bin/env python3
"""
Checkpoint SOUP + eval. Averages distillation checkpoints (SWA/greedy-soup style),
then interpolates the best soup with the base. Composes the two averaging axes:
  soup (avg of trajectory checkpoints)  x  alpha-merge (soup <-> base).

Model soups usually beat any single checkpoint (papers: "model soups = free +0.8").

Steps:
  1. eval each ingredient checkpoint on mean_21 (reports BRTaxQAR/JurisCl/SciELO etc.)
  2. build soups: uniform-all, top-K by mean_21, and (optional) greedy soup
  3. pick the best soup, interpolate with base over alphas, eval FULL-22 on base+best
  4. push the winning model to HF

Usage:
  HF_TOKEN=... python soup_eval.py --ckpts-dir ./ckpt_soup/ckpts \\
     --alphas 0.3 0.5 0.7 --hub-repo tardellirs/embeddinggemma-pt-br-distill \\
     --greedy
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interpolate_eval import eval_mteb, interpolate, BASE  # reuse harness + merge

TARGET = ["BRTaxQAR", "JurisTCUClusteringP2P", "SciELOClusteringP2P"]  # user focus
TOKEN = os.getenv("HF_TOKEN")


def load_sd(path):
    m = SentenceTransformer(path, token=TOKEN)
    sd = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
    del m
    return sd


def avg_sds(sds):
    """uniform average of float tensors (keep non-float from the first)."""
    out = {}
    ref = sds[0]
    for k in ref:
        if ref[k].dtype.is_floating_point:
            out[k] = sum(sd[k] for sd in sds) / len(sds)
        else:
            out[k] = ref[k]
    return out


def model_from_sd(sd):
    m = SentenceTransformer(BASE, token=TOKEN)
    base = m.state_dict()
    for k in base:
        if k in sd and base[k].shape == sd[k].shape:
            base[k] = sd[k].to(base[k].dtype)
    m.load_state_dict(base)
    return m


def m21(model, bs):
    mean, scores = eval_mteb(model, bs, exclude=["Quati"])
    return mean, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts-dir", nargs="+", required=True,
                    help="one or more dirs each holding step_*/epoch_* checkpoint subdirs")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="./soup_results")
    ap.add_argument("--hub-repo", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dirs = []
    for cd in args.ckpts_dir:
        for d in sorted(glob.glob(os.path.join(cd, "*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "modules.json")):
                dirs.append(d)
    # unique display name = <parent-of-ckpts>_<basename>, e.g. ckpt_soupA_step_88
    def nm(d):
        return f"{os.path.basename(os.path.dirname(os.path.dirname(d)))}_{os.path.basename(d)}"
    print(f"found {len(dirs)} ingredient checkpoints:", [nm(d) for d in dirs])

    # 1. eval each ingredient on mean_21
    per_ckpt = {}
    sds = {}
    for d in dirs:
        name = nm(d)
        sds[name] = load_sd(d)
        mean, scores = m21(model_from_sd(sds[name]), args.batch_size)
        per_ckpt[name] = {"mean": mean, "scores": scores}
        tgt = " ".join(f"{t}={scores.get(t, float('nan')):.3f}" for t in TARGET)
        print(f"  [{name}] mean_21={mean:.4f}  {tgt}", flush=True)
        json.dump(per_ckpt, open(os.path.join(args.out, "per_ckpt.json"), "w"), indent=2)

    ranked = sorted(per_ckpt, key=lambda k: per_ckpt[k]["mean"], reverse=True)
    print("\nbest single per target task:")
    for t in TARGET:
        bk = max(per_ckpt, key=lambda k: per_ckpt[k]["scores"].get(t, -1))
        print(f"  {t}: {bk} ({per_ckpt[bk]['scores'].get(t):.3f})")

    # 2. candidate soups
    soups = {}
    soups["uniform_all"] = list(per_ckpt.keys())
    soups["top3"] = ranked[:3]
    soups["top5"] = ranked[:5]
    if args.greedy:
        keep = [ranked[0]]
        best = per_ckpt[ranked[0]]["mean"]
        for c in ranked[1:]:
            cand = avg_sds([sds[n] for n in keep + [c]])
            m, _ = m21(model_from_sd(cand), args.batch_size)
            print(f"  [greedy] try +{c}: soup mean_21={m:.4f} (best {best:.4f})", flush=True)
            if m >= best:
                keep.append(c); best = m
        soups["greedy"] = keep

    soup_res = {}
    for name, members in soups.items():
        if not members:
            continue
        mean, scores = m21(model_from_sd(avg_sds([sds[n] for n in members])), args.batch_size)
        soup_res[name] = {"mean": mean, "members": members, "scores": scores}
        print(f"  [soup:{name}] mean_21={mean:.4f} ({len(members)} ckpts) "
              + " ".join(f"{t}={scores.get(t, float('nan')):.3f}" for t in TARGET), flush=True)
        json.dump(soup_res, open(os.path.join(args.out, "soups.json"), "w"), indent=2)

    # 3. best soup -> alpha sweep on mean_21 (cheap) -> full-22 only on base + best alpha
    best_soup = max(soup_res, key=lambda k: soup_res[k]["mean"])
    print(f"\nBEST soup = {best_soup} (mean_21={soup_res[best_soup]['mean']:.4f})", flush=True)
    soup_sd = avg_sds([sds[n] for n in soup_res[best_soup]["members"]])
    base_model = SentenceTransformer(BASE, token=TOKEN)

    sweep = {"0.0": m21(SentenceTransformer(BASE, token=TOKEN), args.batch_size)[0]}
    print(f"  base mean_21={sweep['0.0']:.4f}", flush=True)
    for a in args.alphas:
        mm, _ = m21(interpolate(base_model, soup_sd, a)[0], args.batch_size)
        sweep[str(a)] = mm
        print(f"  soup interp alpha={a}: mean_21={mm:.4f} ({mm-sweep['0.0']:+.4f})", flush=True)
        json.dump(sweep, open(os.path.join(args.out, "soup_m21_sweep.json"), "w"), indent=2)
    best_a = max(sweep, key=lambda k: sweep[k])

    # full-22 (with Quati) headline: base + winning soup@best_a
    m0, s0 = eval_mteb(SentenceTransformer(BASE, token=TOKEN), args.batch_size)
    full = {"base_mean22": m0, "base_scores": s0, "best_soup": best_soup, "best_alpha": best_a,
            "m21_sweep": sweep}
    if best_a != "0.0":
        wm, _ = interpolate(base_model, soup_sd, float(best_a))
        mfull, sfull = eval_mteb(wm, args.batch_size)
        full["soup_mean22"], full["soup_scores"] = mfull, sfull
        print(f"\n=== SOUP RESULT ===\n  base mean_22={m0:.4f}\n  {best_soup}@a={best_a}: "
              f"mean_22={mfull:.4f} ({mfull-m0:+.4f})", flush=True)
        if args.hub_repo:
            try:
                wm.push_to_hub(args.hub_repo, token=TOKEN, private=True, exist_ok=True,
                               commit_message=f"soup {best_soup} a={best_a} mean_22={mfull:.4f}")
                print(f"  [hub] pushed winning soup -> {args.hub_repo}", flush=True)
            except Exception as e:
                print(f"  [hub] push failed: {e}", flush=True)
    else:
        print(f"\n=== SOUP RESULT: no alpha beat base on mean_21 (base={m0:.4f}) ===", flush=True)
    json.dump(full, open(os.path.join(args.out, "soup_full22.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
