#!/usr/bin/env python3
"""GPU: MTEB(por) mean_22 for Colibri variants — fp32 at Matryoshka dims + fp16.
Writes /root/variant_quality.json incrementally."""
import json, os, sys
sys.path.insert(0, "/root")
from interpolate_eval import eval_mteb
from sentence_transformers import SentenceTransformer

RID = "tardellirs/colibri-embed-ptbr"
TOK = os.getenv("HF_TOKEN")
OUT = "/root/variant_quality.json"
res = {}


def run(tag, model):
    # mean_21 (drop slow Quati) — relative dim/precision trade-off; fast (~7min/eval).
    mean, scores = eval_mteb(model, 128, exclude=["Quati"])
    res[tag] = {"mean21": mean, "scores": scores}
    print(f"[{tag}] mean_21={mean:.4f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)


# fp32 at Matryoshka dims
for dim in [768, 512, 256, 128]:
    run(f"fp32_d{dim}", SentenceTransformer(RID, truncate_dim=dim, token=TOK))
# fp16 (full dim) — confirm ~= fp32
run("fp16_d768", SentenceTransformer(RID, revision="fp16", token=TOK))

print("=== QUALITY SUMMARY (mean_21, no Quati) ===", flush=True)
for k in sorted(res):
    print(f"  {k}: {res[k]['mean21']:.4f}", flush=True)
print("VARIANT-QUALITY DONE", flush=True)
