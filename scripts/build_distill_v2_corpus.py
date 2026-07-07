#!/usr/bin/env python3
"""Rich multi-teacher distillation corpus (v2), from FULL SOURCE datasets with the
MTEB(por) eval rows held out where loadable. Weighted toward Colibri's headroom
(clustering + retrieval domains). Output: data/distill_v2_corpus.parquet (col text, domain).
"""
import os, re
from itertools import islice
import pandas as pd
from datasets import load_dataset

TOK = os.getenv("HF_TOKEN")
OUT = "./data/distill_v2_corpus.parquet"
_ws = re.compile(r"\s+")
def norm(t): return _ws.sub(" ", (t or "")).strip()
def key(t): return norm(t).lower()[:200]

# ---- eval rows to hold out (load the MTEB-BR eval subsets, exclude by text) ----
EVAL = set()
for rid, split in [("MTEB-BR/wikipedia-categories", "train"),      # WikiCat evals on TRAIN
                   ("MTEB-BR/stackoverflow-clustering", "test"),
                   ("MTEB-BR/scielo-clustering", "test")]:
    try:
        ds = load_dataset(rid, split=split, token=TOK)
        col = "sentences" if "sentences" in ds.column_names else ds.column_names[0]
        for v in ds[col]:
            if isinstance(v, list):
                for x in v: EVAL.add(key(x))
            else:
                EVAL.add(key(v))
        print(f"  held-out {rid}[{split}]: +{len(ds)}")
    except Exception as e:
        print(f"  eval load {rid} skip: {str(e)[:80]}")
print(f"eval texts to exclude: {len(EVAL)}")

parts = []
def add(texts, tag, cap):
    out, seen = [], set()
    for t in texts:
        t = norm(t)
        k = key(t)
        if len(t) < 80 or k in EVAL or k in seen:
            continue
        seen.add(k); out.append(t)
        if len(out) >= cap: break
    parts.append(pd.DataFrame({"text": out, "domain": tag}))
    print(f"  + {tag}: {len(out)}", flush=True)


def stream_take(ds, keys, cap):
    out = []
    for ex in islice(ds, cap * 4):
        for k in keys:
            v = ex.get(k)
            if isinstance(v, str) and len(v) >= 80:
                out.append(v); break
        if len(out) >= cap * 2: break
    return out

# medical — AKCIT/MedPT (question+answer)
print("MedPT ...", flush=True)
md = load_dataset("AKCIT/MedPT", split="train", streaming=True, token=TOK).shuffle(seed=11, buffer_size=20000)
add([" ".join(str(ex.get(k, "")) for k in ("question", "answer")) for ex in islice(md, 60000)], "medical", 20000)

# scientific — SciELO abstracts
print("SciELO ...", flush=True)
sc = load_dataset("eduagarcia/scielo_abstracts", split="train", streaming=True, token=TOK).shuffle(seed=11, buffer_size=20000)
ex0 = next(iter(load_dataset("eduagarcia/scielo_abstracts", split="train", streaming=True, token=TOK)))
akeys = [k for k, v in ex0.items() if isinstance(v, str) and "abstract" in k.lower()] or [k for k, v in ex0.items() if isinstance(v, str)]
add(stream_take(sc, akeys, 15000), "scientific", 15000)

# tech — Stack Overflow PT (local)
print("StackO ...", flush=True)
add(pd.read_parquet("./data/stack_pt.parquet")["text"].tolist(), "tech", 15000)

# wiki — Wikipedia-PT (eval articles excluded via EVAL set)
print("Wikipedia-PT ...", flush=True)
wk = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True, token=TOK).shuffle(seed=11, buffer_size=20000)
add(stream_take(wk, ["text"], 15000), "wiki", 15000)

# general web + legal + banking + fiscal — reuse the independent distill corpus
print("reuse distill_corpus (general/legal/banking/fiscal) ...", flush=True)
d = pd.read_parquet("./data/distill_corpus.parquet")
add(d[d.domain == "web"]["text"].tolist(), "general", 15000)
add(d[d.domain == "legal"]["text"].tolist(), "legal", 12000)
add(d[d.domain == "banking"]["text"].tolist(), "banking", 5000)
add(d[d.domain == "fiscal"]["text"].tolist(), "fiscal", 3000)

df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
os.makedirs("./data", exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\nDISTILL-V2 CORPUS: {len(df)} passages -> {OUT}", flush=True)
print(df["domain"].value_counts().to_string(), flush=True)
