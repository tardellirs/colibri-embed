#!/usr/bin/env python3
"""
Distillation step 2: relational (similarity-preserving) KD into the student.

For each batch of texts, the student's pairwise cosine-similarity matrix is
trained to match the (precomputed, frozen) teacher's similarity matrix:
    L = MSE( sim_student , sim_teacher )
Dimension-agnostic (student 768 vs teacher 2560). Full fine-tune, low LR.

Spot-safe: pushes checkpoints to the HF Hub at each checkpoint point (history
retained → resume after preemption via --init-from <hub revision>). Optional
mid-training eval of the MERGED model (alpha*student + (1-alpha)*base) on a
fast no-Quati MTEB(por) subset — the merged model is the real deliverable, so
its trajectory is the honest "is it improving?" signal.

Usage:
  HF_TOKEN=... python distill_train.py --corpus data/distill_corpus.parquet \\
    --teacher-emb data/distill_teacher_emb.npy --output ./ckpt_distill \\
    --epochs 3 --batch-size 256 --lr 5e-6 \\
    --hub-repo tardellirs/embeddinggemma-pt-br-distill \\
    --eval-every-epoch --merge-alpha 0.3
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

BASE = "tardellirs/embeddinggemma-pt-br"          # original base (merge target)
D_PROMPT = "title: none | text: "                  # passage prompt
_EXCLUDED = {"OffComBR", "CSTNewsClustering", "BBCNewsPTClustering", "TweetSentBR"}
# external reference: google/embeddinggemma-300m (full 262k) per-task, from
# docs/SCORE_MATRIX_FINAL_22.md — for context only; our base is the trimmed 64k.
GEMMA300M_MEAN21 = 0.6510


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def eval_mteb(model, batch_size, exclude=("Quati",), include=None):
    """mean_21 (all 22 MTEB(por) tasks minus `exclude`, default drops slow Quati).
    If `include` is given, keep ONLY those tasks (for a fast smoke eval).
    Same harness/prompts as the score matrix -> mteb injects the gemma task prompts.
    mteb 2.16: evaluate() -> ModelResult; get_score() per task; always overwrite."""
    import mteb, mteb_pt.register  # noqa
    tasks = [cls() for cls in mteb_pt.register._TASKS_TO_REGISTER
             if cls.metadata.name not in _EXCLUDED]
    tasks.append(mteb.get_task("Assin2STS"))
    if include:
        keep = set(include)
        tasks = [t for t in tasks if t.metadata.name in keep]
    else:
        ex = set(exclude)
        tasks = [t for t in tasks if t.metadata.name not in ex]
    res = mteb.evaluate(model, tasks=tasks, overwrite_strategy="always",
                        encode_kwargs={"batch_size": batch_size},
                        prediction_folder="/tmp/mid_preds", raise_error=False)
    mr = res[0] if isinstance(res, list) else res
    scores = {}
    for tr in getattr(mr, "task_results", []) or []:
        try:
            scores[tr.task_name] = tr.get_score()
        except Exception:
            pass
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return mean, scores


def merged_model(base_sd, cur_sd, alpha, token):
    """theta = (1-alpha)*base + alpha*current  -> fresh eval model."""
    m = SentenceTransformer(BASE, token=token)
    sd = m.state_dict()
    n = 0
    for k in sd:
        if k in cur_sd and k in base_sd and sd[k].shape == cur_sd[k].shape and sd[k].dtype.is_floating_point:
            sd[k] = ((1 - alpha) * base_sd[k].float() + alpha * cur_sd[k].float().cpu()).to(base_sd[k].dtype)
            n += 1
    m.load_state_dict(sd)
    return m


def push_ckpt(model, hub_repo, token, tag, msg):
    if not hub_repo:
        return
    try:
        model.push_to_hub(hub_repo, token=token, private=True, exist_ok=True, commit_message=msg)
        print(f"  [hub] pushed '{tag}' -> {hub_repo}: {msg}", flush=True)
    except Exception as e:
        print(f"  [hub] push failed ({tag}): {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--teacher-emb", required=True)
    ap.add_argument("--teacher-emb-2", default=None,
                    help="second teacher's embeddings (multi-teacher: KD target = avg of both sim matrices)")
    ap.add_argument("--base", default=BASE, help="merge-target base model (default: %(default)s)")
    ap.add_argument("--output", default="./ckpt_distill")
    ap.add_argument("--init-from", default=BASE, help="training init (base, or a hub ckpt to resume)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hub-repo", default=None, help="HF repo id for checkpoints")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="also save+push a checkpoint every N steps (soup ingredients)")
    ap.add_argument("--eval-every-epoch", action="store_true")
    ap.add_argument("--eval-exclude", nargs="+", default=["Quati"],
                    help="tasks to skip in mid-eval (default: slow Quati -> mean_21)")
    ap.add_argument("--eval-include", nargs="+", default=None,
                    help="keep ONLY these tasks in mid-eval (fast smoke); overrides exclude")
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--merge-alpha", type=float, default=0.3, help="alpha for mid-eval merge")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    globals()["BASE"] = args.base  # merge-target base (retrim-stack for v2)
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_norm(p):
        e = np.load(p).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    texts = pd.read_parquet(args.corpus)["text"].astype(str).tolist()
    t_emb = _load_norm(args.teacher_emb)
    assert len(texts) == len(t_emb), f"corpus {len(texts)} != teacher_emb {len(t_emb)}"
    t_emb2 = None
    if args.teacher_emb_2:
        t_emb2 = _load_norm(args.teacher_emb_2)
        assert len(t_emb2) == len(t_emb), "teacher embs misaligned"
        print(f"MULTI-TEACHER: averaging sim matrices of {t_emb.shape[1]}d + {t_emb2.shape[1]}d teachers", flush=True)
    if args.smoke_test:
        texts, t_emb = texts[:512], t_emb[:512]
        if t_emb2 is not None:
            t_emb2 = t_emb2[:512]
        args.epochs = 1
        print("=== SMOKE TEST (512 rows, 1 epoch) ===", flush=True)

    print(f"Loading student from {args.init_from}", flush=True)
    model = SentenceTransformer(args.init_from, token=token).to(dev)
    model.max_seq_length = args.max_seq_length
    # snapshot ORIGINAL base weights (CPU) for the merge — even when resuming
    base_sd = {k: v.detach().clone().cpu() for k, v in
               SentenceTransformer(BASE, token=token).state_dict().items()}
    if torch.cuda.is_available():
        try:
            model[0].auto_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model[0].auto_model.config.use_cache = False
            print("gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"grad ckpt not enabled: {e}", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"full FT: {n_train:,} trainable | corpus={len(texts)} | bs={args.batch_size} | dev={dev}", flush=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    idx = np.arange(len(texts))
    steps_per_epoch = max(len(texts) // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    t_emb_t = torch.tensor(t_emb, device=dev)
    t_emb2_t = torch.tensor(t_emb2, device=dev) if t_emb2 is not None else None
    Path(args.output).mkdir(parents=True, exist_ok=True)
    eval_log = []
    baseline = {"mean": None, "scores": {}}

    def _save_log():
        with open(os.path.join(args.output, "mid_eval.json"), "w") as f:
            json.dump({"baseline": baseline, "gemma300m_mean21": GEMMA300M_MEAN21,
                       "trajectory": eval_log}, f, indent=2)

    def run_eval(m, tag, ep, alpha):
        """eval model `m` on mean_21 and log with delta-vs-baseline."""
        with torch.no_grad():
            mean, scores = eval_mteb(m, args.eval_batch_size, args.eval_exclude, args.eval_include)
        d = f"{mean - baseline['mean']:+.4f}" if baseline["mean"] is not None else "  base"
        row = {"tag": tag, "epoch": ep, "merge_alpha": alpha, "mean": mean,
               "delta_vs_base": (mean - baseline["mean"]) if baseline["mean"] is not None else 0.0,
               "scores": scores}
        eval_log.append(row); _save_log()
        print(f"  [eval] {tag}: mean_{len(scores)}={mean:.4f} (Δbase {d}; gemma300m {GEMMA300M_MEAN21}) " +
              " ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items())), flush=True)
        return mean, scores

    def checkpoint_and_eval(tag, ep):
        model.save(os.path.join(args.output, "final_model"))
        push_ckpt(model, args.hub_repo, token, tag, f"{tag} (epoch {ep})")
        if args.eval_every_epoch:
            print(f"  [mid-eval] merged alpha={args.merge_alpha}, mean_21 (no Quati) ...", flush=True)
            model.eval()
            try:
                em = merged_model(base_sd, model.state_dict(), args.merge_alpha, token).to(dev)
                run_eval(em, tag, ep, args.merge_alpha)
                del em
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [mid-eval] FAILED: {e}", flush=True)
            model.train()

    # step-0 baseline: OUR trimmed base on mean_21 (same harness -> also verifies
    # the gemma task prompts are applied; expect Assin2STS~0.79, MedPTr~0.77).
    if args.eval_every_epoch:
        print("=== [baseline] eval our base (alpha=0) on mean_21 ===", flush=True)
        model.eval()
        try:
            bmean, bscores = eval_mteb(model, args.eval_batch_size, args.eval_exclude, args.eval_include)
            baseline["mean"], baseline["scores"] = bmean, bscores
            eval_log.append({"tag": "base", "epoch": 0, "merge_alpha": 0.0,
                             "mean": bmean, "delta_vs_base": 0.0, "scores": bscores})
            _save_log()
            print(f"  [baseline] our base mean_{len(bscores)}={bmean:.4f}  (gemma300m {GEMMA300M_MEAN21}) " +
                  " ".join(f"{k}={v:.3f}" for k, v in sorted(bscores.items())), flush=True)
        except Exception as e:
            print(f"  [baseline] FAILED: {e}", flush=True)
        model.train()

    model.train()
    gstep = 0
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        for b in range(steps_per_epoch):
            bidx = idx[b * args.batch_size:(b + 1) * args.batch_size]
            batch_texts = [D_PROMPT + texts[i] for i in bidx]
            feats = model.tokenize(batch_texts)
            feats = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feats.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                s_emb = F.normalize(model(feats)["sentence_embedding"], dim=1)
                sim_s = s_emb @ s_emb.T
                sim_t = t_emb_t[bidx] @ t_emb_t[bidx].T
                if t_emb2_t is not None:  # multi-teacher: average the similarity targets
                    sim_t = 0.5 * (sim_t + t_emb2_t[bidx] @ t_emb2_t[bidx].T)
                loss = F.mse_loss(sim_s, sim_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); gstep += 1
            if gstep % 25 == 0 or gstep == 1:
                print(f"  ep{ep} step {gstep}/{total_steps} loss={loss.item():.5f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.save_steps and gstep % args.save_steps == 0 and gstep < total_steps:
                model.save(os.path.join(args.output, "ckpts", f"step_{gstep}"))  # local soup ingredient
                print(f"  [ckpt] saved soup ingredient step_{gstep}", flush=True)
        checkpoint_and_eval(f"epoch{ep+1}", ep + 1)
        # also keep the epoch checkpoint as a soup ingredient (local)
        model.save(os.path.join(args.output, "ckpts", f"epoch_{ep+1}"))

    out = os.path.join(args.output, "final_model")
    model.save(out)
    push_ckpt(model, args.hub_repo, token, "final", "final distilled student")
    print(f"Saved distilled student -> {out}", flush=True)
    with open(os.path.join(args.output, "distill_summary.json"), "w") as f:
        json.dump({"teacher_emb": args.teacher_emb, "n_corpus": len(texts),
                   "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                   "final_loss": float(loss.item()), "mid_eval": eval_log}, f, indent=2)


if __name__ == "__main__":
    main()
