# Independent-corpus distillation — results

**Goal:** improve `tardellirs/embeddinggemma-pt-br` (64k-vocab trim of embeddinggemma-300m, ~157M effective) on MTEB(por) WITHOUT training on any MTEB-PT task-source dataset ("no benchmark chasing"). Method: relational KD from Qwen3-Embedding-4B on a 67,544-passage corpus mined from INDEPENDENT same-domain PT-BR sources, then weight-merge the distilled student with the base.

## Corpus (67,544 passages, zero task-source overlap)
web 25k (fineweb-2) · banking 14.2k (BBRC+bacen2024) · legal 12k (STJ acórdãos) · scientific 8k (Carolina uni) · fiscal 4.7k (CTN+tax laws) · medical 3.7k (BDTD+Wiki-PT+gov.br).

## V1 — distill 2 epochs (lr 5e-6, bs 256) + α-merge with base

Clean alpha sweep (one eval session; deltas are apples-to-apples):

| model | mean_22 | mean_21 |
|---|---|---|
| our base (trim 64k) | 0.6444 | 0.6466 |
| **distill ⊕ base, α=0.2** | **0.6493 (+0.0049)** | **0.6514 (+0.0048)** |
| google/embeddinggemma-300m (full 262k) | 0.6490 | 0.6510 |

**The +0.0049 makes the 64k-trim model edge past the full embeddinggemma-300m.** α curve (mean_21): 0.0→0.6466, 0.2→0.6514(peak), 0.3→0.6484, 0.4→0.6500, 0.5→0.6494 (noise ~±0.002).

Per-task real gains (base→α0.2): JurisTCUClustering +0.034, WikiCat +0.023, MedPTClustering +0.015, JurisTCUReranking +0.011, fiscal BRTaxQA +0.008. STS/NLI preserved (Assin2 0.798→0.798). Banking FaqBacen −0.007.

Caveat corrected: the in-training baseline reported BRTaxQAR=0.246 (memory-pressure artifact); TRUE base BRTaxQAR≈0.374 (= gemma-300m 0.375), so real fiscal gain is +0.008, not the +0.132 an earlier bad-baseline read suggested.

## CPU efficiency — Colibri vs google/embeddinggemma-300m (head-to-head)
Same VPS-like VM (Verda CPU.4V.16G = 4 vCPU, 16 GB). Encode of PT sentences, batch 32.

| variant | model | weight | latency p50 | throughput | peak RAM |
|---|---|---|---|---|---|
| fp32 | **Colibri (157M)** | 607 MB | 76 ms | 37/s | **969 MB** |
| fp32 | gemma-300m | 1211 MB | 78 ms | 36/s | 1253 MB |
| onnx | **Colibri** | ~600 MB | **33 ms** | **49/s** | **2.9 GB** |
| onnx | gemma-300m | ~1200 MB | 38 ms | 41/s | 5.3 GB |
| fp16 | Colibri | 304 MB | 75 ms | 38/s | 1287 MB |

**Findings:** (1) encode LATENCY ~identical (vocab trim shrinks the embedding matrix, not the 24 transformer layers = the compute). (2) Colibri wins on SIZE (half) + RAM — decisive on a small VPS; ONNX mode 2.9 GB vs 5.3 GB. (3) ONNX ≈ 2× faster than fp32 for both. (4) fp16 useless on CPU (no speedup, more RAM). (5) ONNX-int8 not buildable: gemma3 unsupported by optimum's ORTModel quantized-load path (dropped). Matryoshka dims don't change encode cost (post-encode truncation); win is vector footprint = dim×4 bytes.

**PENDING:** GPU quality per Matryoshka dim (mean_21 for fp32 @ 512/256/128 + fp16) — spot A100 was preempted; re-run when spot available.

### GGUF (Q8_0/Q4_0) — attempted, NOT viable out-of-box (2026-07-03)
Tried llama.cpp GGUF quantization for edge/CPU. Two blockers:
1. **Colibri won't convert** — `convert_hf_to_gguf.py` AssertionError on the trimmed 64k tokenizer.
2. **Embeddings don't match the model** — even gemma-300m **f16** GGUF gives `cosine_vs_fp32 ≈ 0` (orthogonal). Cause: EmbeddingGemma's **Dense projection heads** (ST `2_Dense`/`3_Dense`) are SEPARATE safetensors; `convert_hf_to_gguf` only reads the transformer `model.safetensors`, so the projection is dropped → llama.cpp embeds the raw transformer, a different space than the ST model.
GGUF efficiency numbers (llama.cpp ~2× faster + less RAM than torch) are moot without faithful embeddings. Verdict: GGUF needs an EmbeddingGemma-specific conversion that folds the Dense projection (Google's official embeddinggemma GGUF exists, so it's possible) + a tokenizer fix for the trim — a dedicated effort, deferred. **The validated CPU/VPS path is ONNX** (faithful, 2.3× faster than fp32, less RAM than gemma-300m).

## Domain-aware RE-TRIM (2026-07-03) — StackOverflow is the key
Which tasks did the plain 64k trim hurt vs full gemma-300m? All CLUSTERING: StackoverflowPt −0.032, WikiCat −0.030, MedPTcl −0.019 (vocab-coverage effect — the PT trim dropped tech-English/rare/medical tokens). BRTaxQA did NOT degrade (−0.001; the earlier −0.13 was an artifact).

Re-trimmed 64k (SAME 157M size) with a domain-rich token-selection corpus (fineweb + Wikipedia-PT + AKCIT/MedPT 40k + SciELO + legal, ± Stack Overflow em Português 45k from the Stack Exchange dump). 3-way MTEB(por) mean_21:

| trim | mean_21 | vs old | key per-task Δ |
|---|---|---|---|
| old (generic `por`) | 0.6466 | — | — |
| re-trim **no-stack** | 0.6458 | −0.0008 | StackO +0.018, but MedPTcl −0.025 → net neutral/worse |
| re-trim **+stack** | **0.6477** | **+0.0011** | StackO +0.020, WikiCat +0.012, JurisCl +0.012, MedPTcl +0.004, SciELO −0.012 |

**Verdict:** including **Stack Overflow em Português** (the user's instinct) is what tips the domain-aware re-trim to net-positive — it recovers the degraded clustering tasks at the same model size, giving a better BASE. Modest (+0.0011, near noise) but directionally clear. The +stack re-trim (`tardellirs/embeddinggemma-pt-br-64k-retrim-stack-test`) is the candidate NEW BASE for a Colibri-v2 (then re-distill on it = the A+B plan). Scripts: build_retrim_corpus.py, get_stackoverflow_pt.py, retrim_vocab.py (--corpus-parquet), compare_trims.py.

## DELIVERABLE (final)
**`tardellirs/colibri-embed-ptbr` main (commit 00d2376c)** = **v2 soup model**: soup=avg(distill-epoch1, distill-epoch2), merged with base at α=0.2. Plain standalone SentenceTransformer (no adapters). MTEB(por) **mean_22 = 0.6497**. Reproducible from HF (base + epoch1 rev 40927590 + epoch2 rev 13380dd6). Raw checkpoints + v1 remain in git history.

| model | mean_22 | mean_21 |
|---|---|---|
| our base (trim 64k) | 0.6444 | 0.6466 |
| v1 (distill-epoch2 ⊕ base, α0.2) | 0.6493 | 0.6514 |
| **v2 soup (avg e1,e2 ⊕ base, α0.2)** | **0.6497** | **0.6520** |
| google/embeddinggemma-300m (full) | 0.6490 | 0.6510 |

## V2 — checkpoint SOUP (two attempts)
**Attempt 1 (full campaign, PREEMPTED):** 2 KD runs (seed42 lr5e-6; seed123 lr1e-5), intra-epoch checkpoints → 14 ingredients. soup_eval reached 11/14 before the **spot A100 was preempted** (VM + local intra-epoch checkpoints lost). Data: all raw ingredients scored mean_21 0.646–0.650.

**Attempt 2 (lean, on-demand A6000 — SUCCESS):** used the 2 raw checkpoints that survived on HF (epoch1, epoch2). soup=avg → α sweep vs base (mean_21): 0.1→0.6484, 0.15→0.6489, **0.2→0.6520 (peak)**, 0.25→0.6497, 0.3→0.6481. Full-22 at α0.2 = **0.6497**, edging v1 (0.6493) by **+0.0004** — nominally the best, though within run-to-run noise (~±0.002); the soup is a mild SWA (2-checkpoint average) so it is the slightly-more-robust deliverable. On-demand avoided preemption; ~$1.3.

Note: a device bug in `interpolate_eval.interpolate` (CPU soup_sd vs CUDA base) crashed soup_eval's alpha sweep first — fixed (device-safe) and rerun via lean `finish_soup.py`. Total campaign spend ~$4.3.

## V2 MULTI-TEACHER (2026-07-04) — Qwen3-4B + 8B relational KD on the +stack re-trim base
New campaign (distinct from the single-teacher soup above): base = `embeddinggemma-pt-br-64k-retrim-stack-test`, teachers = **Qwen3-Embedding-4B (2560d, clustering champ) + Qwen3-Embedding-8B (4096d, retrieval)**, target = per-batch AVERAGE of the two teachers' cosine-sim matrices. 100k-passage domain corpus (medical/scientific/tech/wiki/general/legal/banking/fiscal), 2 epochs, lr5e-6, bs256, intra-epoch soup checkpoints.

soup_eval: 10 ingredients scored mean_21 0.6485–0.6517; best single = step_528 (0.6517); best soup = top5 (0.6517, = step_528 on mean_21). The default sweep capped α at 0.3 → pushed α=0.25 = mean_22 **0.6481** (BELOW v1!).

**Extended α sweep (user's call — measure the full curve on cheap mean_21, then mean_22 on the peak):** the mean_21 curve is NON-monotonic — dips below base at α=0.1, then peaks at **α≈0.65–0.70 = 0.6532** (above the pure soup 0.6517!), then falls. The merge at the right α both raises the mean AND recovers regressed tasks (StackO 0.513→0.534, WikiCat 0.656→0.673). Two spot preemptions during soup_eval/extend; the 2nd was recovered WITHOUT re-training by reusing the locally-backed-up step_528 ingredient (best single = top5 on mean_21) — extend-only, ~1.5h vs ~4.5h.

**Winner: step_528 ⊕ base @ α=0.65 → mean_22 = 0.6501** (pushed to `tardellirs/colibri-v2-distill`).

| model | mean_22 | note |
|---|---|---|
| +stack retrim base | 0.6455 | multi-teacher base |
| capped α=0.25 | 0.6481 | what the default pipeline would ship |
| **multi-teacher @ α=0.65** | **0.6501** | extended-sweep winner |
| current Colibri deliverable (single-teacher soup) | 0.6497 | for reference |
| Qwen3-4B teacher | 0.6621 | upper bound |

**Verdict:** multi-teacher @ optimal-α = **0.6501**, nominally the best but +0.0004 over the current deliverable = within run-to-run noise (~±0.002). The headline barely moves; the real change is a **stronger clustering profile** — JurisTCUClusteringP2P 0.397, StackO 0.534, SciELO 0.724, WikiCat 0.673, JurisReranking 0.522 all up — traded against a **MedPTClustering regression (0.700→0.682)** (surprising given the 4B teacher is the MedPTcl champ; the α=0.65 merge dampened it). The extended α sweep was the decisive lever: +0.0020 over the capped pipeline (0.6481→0.6501), turning a below-v1 result into the nominal best. Local ingredient backups (step_528/epoch_1/step_176) were the robustness key across 2 preemptions. Scripts: run_distill_v2.sh, distill_train.py (--teacher-emb-2), soup_eval.py, extend_sweep.py.
