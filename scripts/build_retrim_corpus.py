#!/usr/bin/env python3
"""Rich token-selection corpus for a domain-aware 64k RE-TRIM, from the FULL SOURCE
datasets (disjoint from the tiny MTEB eval subsets). Over-represents the domains the
plain-PT trim hurt (medical=MedPTcl, wiki=WikiCat) so their vocabulary survives the
64k cut — WITHOUT enlarging the model. Output: data/retrim_corpus.parquet (col 'text').
Note: vocab selection is unsupervised (token frequencies), so this is not benchmark
fitting; eval-row exclusion matters for DISTILLATION (later), not for the trim.
"""
import os
from itertools import islice
import pandas as pd
from datasets import load_dataset

TOK = os.getenv("HF_TOKEN")
OUT = "./data/retrim_corpus.parquet"
parts = []


def add(texts, tag):
    texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 80]
    if texts:
        parts.append(pd.DataFrame({"text": texts, "domain": tag}))
    print(f"  + {tag}: {len(texts)}", flush=True)


def stream_texts(ds, n, keys):
    out = []
    for ex in islice(ds, n * 3):
        for k in keys:
            v = ex.get(k)
            if isinstance(v, str) and len(v) >= 80:
                out.append(v); break
        if len(out) >= n:
            break
    return out


# general PT — keep general coverage dominant
print("fineweb-2 por ...", flush=True)
fw = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True,
                  token=TOK).shuffle(seed=7, buffer_size=20000)
add(stream_texts(fw, 80000, ["text"]), "general-fineweb")

# WikiCat domain — full Wikipedia-PT
print("wikipedia pt ...", flush=True)
try:
    wk = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True,
                      token=TOK).shuffle(seed=7, buffer_size=20000)
    add(stream_texts(wk, 50000, ["text"]), "wikipedia-pt")
except Exception as e:
    print("  wikimedia failed:", str(e)[:120], flush=True)

# MedPTcl domain — AKCIT/MedPT (384k), question+answer text
print("AKCIT/MedPT ...", flush=True)
try:
    md = load_dataset("AKCIT/MedPT", split="train", streaming=True, token=TOK).shuffle(seed=7, buffer_size=20000)
    rows = []
    for ex in islice(md, 40000):
        t = " ".join(str(ex.get(k, "")) for k in ("question", "answer")).strip()
        if len(t) >= 80:
            rows.append(t)
    add(rows, "medical-medpt")
except Exception as e:
    print("  MedPT failed:", str(e)[:120], flush=True)

# SciELO scientific abstracts (general boost)
print("scielo ...", flush=True)
try:
    sc = load_dataset("eduagarcia/scielo_abstracts", split="train", streaming=True, token=TOK).shuffle(seed=7, buffer_size=20000)
    ex0 = next(iter(sc))
    keys = [k for k, v in ex0.items() if isinstance(v, str) and "abstract" in k.lower()] or \
           [k for k, v in ex0.items() if isinstance(v, str)]
    add(stream_texts(sc, 20000, keys), "scientific-scielo")
except Exception as e:
    print("  scielo failed:", str(e)[:120], flush=True)

# legal — reuse from distill corpus if present
try:
    d = pd.read_parquet("./data/distill_corpus.parquet")
    add(d[d.domain == "legal"]["text"].tolist()[:12000], "legal")
except Exception as e:
    print("  legal reuse failed:", str(e)[:120], flush=True)

df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
os.makedirs("./data", exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\nRICH RE-TRIM CORPUS: {len(df)} texts -> {OUT}", flush=True)
print(df["domain"].value_counts().to_string(), flush=True)
