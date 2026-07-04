#!/usr/bin/env python3
"""GPU: head-to-head MTEB(por) mean_21 of the domain-aware RE-TRIM vs the current
64k trim (and optionally the full 300m). Prints per-task deltas (did WikiCat/MedPTcl
recover?). Writes /root/compare_trims.json."""
import json, os, sys
sys.path.insert(0, "/root")
from interpolate_eval import eval_mteb
from sentence_transformers import SentenceTransformer

TOK = os.getenv("HF_TOKEN")
MODELS = {
    "old_trim_64k": "tardellirs/embeddinggemma-pt-br",
    "retrim_nostack": "tardellirs/embeddinggemma-pt-br-64k-retrim-test",
    "retrim_stack": "tardellirs/embeddinggemma-pt-br-64k-retrim-stack-test",
}
res = {}
for tag, mid in MODELS.items():
    print(f"=== {tag} ({mid}) ===", flush=True)
    m = SentenceTransformer(mid, token=TOK)
    mean, scores = eval_mteb(m, 128, exclude=["Quati"])  # mean_21
    res[tag] = {"model": mid, "mean21": mean, "scores": scores}
    print(f"[{tag}] mean_21={mean:.4f}", flush=True)
    json.dump(res, open("/root/compare_trims.json", "w"), indent=2)

old = res["old_trim_64k"]["scores"]
ns = res["retrim_nostack"]["scores"]
st = res["retrim_stack"]["scores"]
TARGETS = ("StackoverflowPtClustering", "WikipediaPTCategoriesClusteringP2P", "MedPTClustering")
print("\n=== per-task (old | nostack Δ | stack Δ) ===", flush=True)
for t in sorted(old, key=lambda k: st.get(k, 0) - old.get(k, 0)):
    flag = "  <-- target" if t in TARGETS else ""
    print(f"  {t:<34} old {old[t]:.3f}  nostack {ns.get(t,0)-old[t]:+.3f}  stack {st.get(t,0)-old[t]:+.3f}{flag}", flush=True)
print(f"\nmean_21:  old {res['old_trim_64k']['mean21']:.4f}  "
      f"nostack {res['retrim_nostack']['mean21']:.4f} ({res['retrim_nostack']['mean21']-res['old_trim_64k']['mean21']:+.4f})  "
      f"stack {res['retrim_stack']['mean21']:.4f} ({res['retrim_stack']['mean21']-res['old_trim_64k']['mean21']:+.4f})", flush=True)
print("COMPARE-TRIMS DONE", flush=True)
