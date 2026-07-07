#!/usr/bin/env python3
"""
Distillation step 1: precompute TEACHER embeddings over a PT corpus.

Teacher = Qwen3-Embedding-4B (open, dim 2560), strongest open model on PT
clustering. We only need its embeddings (unit-normalized) so the student can
later learn to reproduce the teacher's pairwise-similarity geometry (relational
KD — dimension-agnostic).

Corpus: diverse PT passages (mMARCO-pt) — external, no MTEB-BR test overlap.

Output: data/distill_corpus.parquet  (columns: text, and a .npy of teacher embs)
"""
from __future__ import annotations
import argparse, os
from itertools import islice
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--corpus-parquet", default=None,
                    help="pre-built corpus parquet with a 'text' column (skips mMARCO gathering)")
    ap.add_argument("--limit", type=int, default=0, help="truncate corpus to first N rows (smoke)")
    ap.add_argument("--emb-out", default="distill_teacher_emb.npy",
                    help="output filename for this teacher's embeddings (per-teacher for multi-teacher)")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. gather corpus
    if args.corpus_parquet:
        print(f"Loading pre-built corpus from {args.corpus_parquet} ...")
        df = pd.read_parquet(args.corpus_parquet)
        if args.limit:
            df = df.head(args.limit)
        texts = df["text"].astype(str).tolist()  # parquet order preserved -> emb aligns
        print(f"corpus: {len(texts)} passages ({df['domain'].value_counts().to_dict() if 'domain' in df else 'no domain col'})")
    else:
        from datasets import load_dataset
        print("Loading mMARCO-pt passages...")
        ds = load_dataset("andreribeiro87/mmarco-more-hard-negatives", split="train",
                          streaming=True, token=token).shuffle(seed=args.seed, buffer_size=50000)
        seen, texts = set(), []
        for ex in islice(ds, args.n * 3):
            p = (ex.get("positive") or "").strip()
            if p and p not in seen:
                seen.add(p); texts.append(p)
            if len(texts) >= args.n:
                break
        print(f"corpus: {len(texts)} unique passages")

    # 2. teacher embeddings
    from sentence_transformers import SentenceTransformer
    print(f"Loading teacher {args.teacher} ...")
    teacher = SentenceTransformer(args.teacher, token=token, trust_remote_code=True)
    teacher.max_seq_length = args.max_seq_length  # cap: long passages otherwise OOM the 4B/8B teacher
    print(f"Encoding with teacher (max_seq_length={teacher.max_seq_length}, bs={args.batch_size})...")
    emb = teacher.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)
    print(f"teacher embeddings: {emb.shape}")

    np.save(os.path.join(args.out_dir, args.emb_out), emb.astype(np.float16))
    if args.corpus_parquet:
        # corpus already on disk (with domain cols); only the aligned embeddings are new.
        # for smoke --limit, also write the truncated corpus so train stays aligned.
        if args.limit:
            df.to_parquet(os.path.join(args.out_dir, "distill_corpus_smoke.parquet"), index=False)
        print(f"Saved teacher embeddings ({emb.shape}) aligned to {args.corpus_parquet}")
    else:
        pd.DataFrame({"text": texts}).to_parquet(os.path.join(args.out_dir, "distill_corpus.parquet"), index=False)
        print(f"Saved corpus + teacher embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
