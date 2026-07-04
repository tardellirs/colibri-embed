# MTEB(por) Teacher Survey for Knowledge Distillation into embeddinggemma-pt-br

**Student model:** `tardellirs/embeddinggemma-pt-br` (~157M effective params, embedding dim 768)
**Source leaderboard:** `mteb-pt/mteb-pt-results` (score_matrix.parquet, 93 models × 22 tasks, fetched 2026-07-02)
**Constraint:** Open-weight models only (type = "O"); closed/API models excluded.
**Student overall rank:** #13 of 93 (open rank #9 of 73), mean_22 = 0.6490

---

## 1. Per-Task Analysis (22 MTEB(por) Tasks)

### Notation
- **gap** = best_open_score − gemma_score (positive = teacher is better)
- Weak tasks (large gaps or poor open rank) are marked **★ WEAK**
- Scores are the primary metric reported by MTEB (nDCG@10 for retrieval, V-measure for clustering, accuracy/F1 for classification, Pearson/Spearman for STS, MAP for reranking)

---

### 1.1 Classification Tasks

#### HateBR (binary hate-speech classification)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | harrier-oss-v1-27b | 27.0B | 5376 | 0.8774 |
| 2 | F2LLM-v2-14B | 14.0B | 5120 | 0.8609 |
| 3 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8560 |
| 4 | BOOM_4B_v1 | 4.0B | 2560 | 0.8519 |
| 5 | PIXIE-Rune-v1.0 | 568M | 1024 | 0.8441 |
| **9** | **embeddinggemma-300m** | **308M** | **768** | **0.8304** |

Gap to best open: +0.0470. Gemma open rank: 13/73.

---

#### FactckBrClassification (fact-checking)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | harrier-oss-v1-27b | 27.0B | 5376 | 0.6054 |
| 2 | F2LLM-v2-8B | 7.6B | 4096 | 0.5846 |
| 3 | F2LLM-v2-4B | 4.0B | 2560 | 0.5815 |
| 4 | F2LLM-v2-14B | 14.0B | 5120 | 0.5788 |
| 5 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.5722 |
| **6** | **embeddinggemma-300m** | **308M** | **768** | **0.5691** |

Gap to best open: +0.0363. Gemma open rank: 6/73 — already near top.

---

#### ToxSynPT (toxicity classification)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | Linq-Embed-Mistral | 7.1B | 4096 | 0.9076 |
| 2 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.9008 |
| 3 | BidirLM-1B-Embedding | 1000M | 1152 | 0.8994 |
| 4 | BidirLM-1.7B-Embedding | 1.7B | 2048 | 0.8981 |
| 5 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.8962 |
| **18** | **embeddinggemma-300m** | **308M** | **768** | **0.8593** |

Gap to best open: +0.0483. Gemma open rank: 18/73.

---

#### PortuLexRRIP (legal text classification) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | bert-large-portuguese-cased | 335M | 1024 | 0.5575 |
| 2 | bert-base-portuguese-cased | 110M | 768 | 0.5367 |
| 3 | albertina-900m-portuguese-ptbr-encoder | 900M | ~2048 | 0.4780 |
| 4 | harrier-oss-v1-27b | 27.0B | 5376 | 0.4742 |
| 5 | LaBSE | 471M | 768 | 0.4652 |
| **24** | **embeddinggemma-300m** | **308M** | **768** | **0.4223** |

Gap to best open: **+0.1352**. Gemma open rank: 24/73.
**Note:** This task rewards domain-specific PT-language pretraining (BERTimbau-style). General embedding models struggle. KD from a strong general model will not close this gap much; task-specific fine-tuning or LaBSE-style training data is needed.

---

#### BrighterEmotionMultilabelClassification ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | multilingual-e5-large-instruct | 560M | 1024 | 0.3297 |
| 2 | SFR-Embedding-2_R | 7.1B | 4096 | 0.3263 |
| **3** | **embeddinggemma-300m** | **308M** | **768** | **0.3237** |
| 4 | BidirLM-1B-Embedding | 1000M | 1152 | 0.3186 |
| 5 | gte-Qwen2-7B-instruct | 7.1B | 3584 | 0.3105 |

Gap to best open: **+0.0060**. Gemma open rank: 3/73.
**Note:** Gemma is already rank 3 open on this task. All models score low (best open: 0.33). Not a KD priority.

---

### 1.2 NLI / STS Tasks

#### AssinRTE (textual entailment)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8862 |
| 2 | PwC-Embedding_expr | 560M | 1024 | 0.8814 |
| 3 | multilingual-e5-large-instruct | 560M | 1024 | 0.8771 |
| **4** | **embeddinggemma-300m** | **308M** | **768** | **0.8757** |
| 5 | serafim-900m-portuguese-pt-sentence-encoder | 900M | — | 0.8723 |

Gap to best open: +0.0105. Gemma open rank: 4/73 — excellent.

---

#### InferBR (Portuguese NLI)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | Octen-Embedding-8B | 7.6B | 4096 | 0.9098 |
| 2 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.9079 |
| 3 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8986 |
| 4 | Qwen3-Embedding-4B | 4.0B | 2560 | 0.8939 |
| **5** | **embeddinggemma-300m** | **308M** | **768** | **0.8734** |

Gap to best open: +0.0364. Gemma open rank: 5/73.

---

#### AssinSTS (STS, sentence similarity)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | multilingual-e5-large-instruct | 560M | 1024 | 0.8076 |
| 2 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8065 |
| 3 | PwC-Embedding_expr | 560M | 1024 | 0.8052 |
| 4 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.8044 |
| 5 | Linq-Embed-Mistral | 7.1B | 4096 | 0.8034 |
| **9** | **embeddinggemma-300m** | **308M** | **768** | **0.7886** |

Gap to best open: +0.0190. Gemma open rank: 9/73 — solid.

---

#### Assin2STS (STS v2)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | harrier-oss-v1-27b | 27.0B | 5376 | 0.8349 |
| 2 | serafim-335m-portuguese-pt-sentence-encoder | 335M | — | 0.8323 |
| 3 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.8267 |
| 4 | serafim-900m-portuguese-pt-sentence-encoder | 900M | — | 0.8267 |
| 5 | Octen-Embedding-8B | 7.6B | 4096 | 0.8229 |
| **17** | **embeddinggemma-300m** | **308M** | **768** | **0.7986** |

Gap to best open: +0.0363. Gemma open rank: 17/73.

---

### 1.3 Clustering Tasks

#### MedPTClustering (biomedical PT clustering) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | **Qwen3-Embedding-4B** | **4.0B** | **2560** | **0.8863** |
| 2 | BidirLM-1.7B-Embedding | 1.7B | 2048 | 0.8573 |
| 3 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.8069 |
| 4 | gte-Qwen2-1.5B-instruct | 1.5B | 1536 | 0.8022 |
| 5 | jina-embeddings-v5-text-small | 596M | 1024 | 0.7990 |
| **30** | **embeddinggemma-300m** | **308M** | **768** | **0.7194** |

Gap to best open: **+0.1669** — largest gap in the benchmark.
Qwen3-4B leads by a huge margin; #2 BidirLM-1.7B also far ahead.

---

#### WikipediaPTCategoriesClusteringP2P ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.7992 |
| 2 | Octen-Embedding-8B | 7.6B | 4096 | 0.7919 |
| 3 | multilingual-e5-large-instruct | 560M | 1024 | 0.7882 |
| 4 | Qwen3-Embedding-0.6B | 596M | 1024 | 0.7737 |
| 5 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.7722 |
| **22** | **embeddinggemma-300m** | **308M** | **768** | **0.6861** |

Gap to best open: **+0.1131**. Qwen3 models dominate Wikipedia clustering.

---

#### JurisTCUClusteringP2P (legal clustering) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | harrier-oss-v1-27b | 27.0B | 5376 | 0.4471 |
| 2 | F2LLM-v2-4B | 4.0B | 2560 | 0.4382 |
| 3 | jua-4B-mixed | 4.0B | 2560 | 0.4367 |
| 4 | jua-4B-legal-only | 4.0B | — | 0.4147 |
| 5 | F2LLM-v2-14B | 14.0B | 5120 | 0.4139 |
| **42** | **embeddinggemma-300m** | **308M** | **768** | **0.2942** |

Gap to best open: **+0.1529**. Legal-domain clustering; all models score low (task is hard).

---

#### SciELOClusteringP2P (scientific literature clustering) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8263 |
| 2 | BidirLM-1.7B-Embedding | 1.7B | 2048 | 0.7811 |
| 3 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.7776 |
| 4 | Octen-Embedding-8B | 7.6B | 4096 | 0.7776 |
| 5 | BOOM_4B_v1 | 4.0B | 2560 | 0.7664 |
| **18** | **embeddinggemma-300m** | **308M** | **768** | **0.7002** |

Gap to best open: **+0.1261**.

---

#### StackoverflowPtClustering ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.6526 |
| 2 | SFR-Embedding-2_R | 7.1B | 4096 | 0.6356 |
| 3 | BidirLM-1.7B-Embedding | 1.7B | 2048 | 0.6291 |
| 4 | Qwen3-Embedding-4B | 4.0B | 2560 | 0.6255 |
| 5 | jina-embeddings-v5-text-small | 596M | 1024 | 0.6167 |
| **13** | **embeddinggemma-300m** | **308M** | **768** | **0.5450** |

Gap to best open: **+0.1076**.

---

### 1.4 Retrieval Tasks

#### MedPTRetrieval (biomedical retrieval) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | F2LLM-v2-14B | 14.0B | 5120 | 0.8863 |
| 2 | harrier-oss-v1-27b | 27.0B | 5376 | 0.8809 |
| 3 | KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 0.8741 |
| 4 | F2LLM-v2-8B | 7.6B | 4096 | 0.8712 |
| 5 | F2LLM-v2-4B | 4.0B | 2560 | 0.8628 |
| **20** | **embeddinggemma-300m** | **308M** | **768** | **0.7771** |

Gap to best open: **+0.1092**. F2LLM family dominates biomedical retrieval.

---

#### FaQuADIR (Brazilian FAQ retrieval)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | PIXIE-Rune-v1.0 | 568M | 1024 | 0.8677 |
| 2 | BOOM_4B_v1 | 4.0B | 2560 | 0.8623 |
| **3** | **embeddinggemma-300m** | **308M** | **768** | **0.8464** |
| 4 | multilingual-e5-large | 560M | 1024 | 0.8444 |
| 5 | Linq-Embed-Mistral | 7.1B | 4096 | 0.8403 |

Gap to best open: +0.0213. Gemma open rank: 3/73 — excellent.

---

#### Quati (PT-BR QA retrieval)
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | F2LLM-v2-8B | 7.6B | 4096 | 0.6492 |
| 2 | Octen-Embedding-8B | 7.6B | 4096 | 0.6442 |
| 3 | F2LLM-v2-14B | 14.0B | 5120 | 0.6427 |
| 4 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.6413 |
| 5 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.6260 |
| **9** | **embeddinggemma-300m** | **308M** | **768** | **0.6074** |

Gap to best open: +0.0418. Gemma open rank: 9/73.

---

#### FaqBacenRetrieval (Brazilian Central Bank FAQ) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | **F2LLM-v2-8B** | **7.6B** | **4096** | **0.8229** |
| 2 | F2LLM-v2-14B | 14.0B | 5120 | 0.8189 |
| 3 | F2LLM-v2-4B | 4.0B | 2560 | 0.8144 |
| 4 | F2LLM-v2-1.7B | 1.7B | 2048 | 0.7896 |
| 5 | F2LLM-v2-0.6B | 596M | 1024 | 0.7635 |
| **15** | **embeddinggemma-300m** | **308M** | **768** | **0.6949** |

Gap to best open: **+0.1280**. F2LLM family sweeps all top-5 positions.

---

#### JurisTCU (legal retrieval) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | snowflake-arctic-embed-l-v2.0 | 568M | 1024 | 0.6622 |
| 2 | BOOM_4B_v1 | 4.0B | 2560 | 0.6408 |
| 3 | PIXIE-Rune-v1.0 | 568M | 1024 | 0.6306 |
| 4 | Linq-Embed-Mistral | 7.1B | 4096 | 0.6227 |
| **5** | **embeddinggemma-300m** | **308M** | **768** | **0.6207** |

Gap to best open: +0.0415. Gemma open rank: 5/73 — already near top.

---

#### BRTaxQAR (Brazilian tax law retrieval) ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.4241 |
| 2 | Qwen3-Embedding-4B | 4.0B | 2560 | 0.4195 |
| 3 | SFR-Embedding-2_R | 7.1B | 4096 | 0.3937 |
| 4 | bge-m3 | 568M | 1024 | 0.3772 |
| **5** | **embeddinggemma-300m** | **308M** | **768** | **0.3748** |

Gap to best open: +0.0493. Gemma open rank: 5/73 — already rank 5.
**Note:** All models score low (0.10–0.42); this is the hardest retrieval task. Closed models don't score much higher.

---

### 1.5 Reranking Tasks

#### QuatiReranking
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | F2LLM-v2-14B | 14.0B | 5120 | 0.6350 |
| 2 | Octen-Embedding-8B | 7.6B | 4096 | 0.6344 |
| 3 | Qwen3-Embedding-8B | 7.6B | 4096 | 0.6298 |
| 4 | F2LLM-v2-8B | 7.6B | 4096 | 0.6272 |
| 5 | SFR-Embedding-Mistral | 7.1B | 4096 | 0.6254 |
| **27** | **embeddinggemma-300m** | **308M** | **768** | **0.5693** |

Gap to best open: +0.0657. Gemma open rank: 27/73 — moderate gap.

---

#### JurisTCUReranking ★ WEAK
| Rank (open) | Model | Params | Dim | Score |
|---|---|---|---|---|
| 1 | snowflake-arctic-embed-l-v2.0 | 568M | 1024 | 0.6006 |
| 2 | Linq-Embed-Mistral | 7.1B | 4096 | 0.5810 |
| 3 | SFR-Embedding-2_R | 7.1B | 4096 | 0.5745 |
| 4 | PIXIE-Rune-v1.0 | 568M | 1024 | 0.5709 |
| 5 | F2LLM-v2-8B | 7.6B | 4096 | 0.5645 |
| **38** | **embeddinggemma-300m** | **308M** | **768** | **0.5023** |

Gap to best open: **+0.0983**. Gemma open rank: 38/73.

---

## 2. Summary Table — Weak Tasks

| Task | Gemma Score | Gemma Open Rank | Best Open Model | Best Open Score | Gap | Best Open Dim |
|---|---|---|---|---|---|---|
| MedPTClustering | 0.7194 | 30/73 | Qwen3-Embedding-4B | 0.8863 | **+0.1669** | 2560 |
| JurisTCUClusteringP2P | 0.2942 | 42/73 | harrier-oss-v1-27b | 0.4471 | **+0.1529** | 5376 |
| PortuLexRRIP | 0.4223 | 24/73 | bert-large-portuguese-cased | 0.5575 | **+0.1352** | 1024 |
| FaqBacenRetrieval | 0.6949 | 15/73 | F2LLM-v2-8B | 0.8229 | **+0.1280** | 4096 |
| SciELOClusteringP2P | 0.7002 | 18/73 | KaLM-Embedding-Gemma3-12B-2511 | 0.8263 | **+0.1261** | 3840 |
| WikipediaPTCategoriesClusteringP2P | 0.6861 | 22/73 | Qwen3-Embedding-8B | 0.7992 | **+0.1131** | 4096 |
| MedPTRetrieval | 0.7771 | 20/73 | F2LLM-v2-14B | 0.8863 | **+0.1092** | 5120 |
| StackoverflowPtClustering | 0.5450 | 13/73 | SFR-Embedding-Mistral | 0.6526 | **+0.1076** | 4096 |
| JurisTCUReranking | 0.5023 | 38/73 | snowflake-arctic-embed-l-v2.0 | 0.6006 | **+0.0983** | 1024 |
| BRTaxQAR | 0.3748 | 5/73 | Qwen3-Embedding-8B | 0.4241 | +0.0493 | 4096 |
| JurisTCU | 0.6207 | 5/73 | snowflake-arctic-embed-l-v2.0 | 0.6622 | +0.0415 | 1024 |
| BrighterEmotionMultilabelClassification | 0.3237 | 3/73 | multilingual-e5-large-instruct | 0.3297 | +0.0060 | 1024 |

---

## 3. Candidate Teacher Profile — Key Open Models

| Model | HF ID | Params | Dim | Overall Open Rank | mean_22 | Top-5 / 22 tasks | Top-3 / 12 weak tasks |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | 7.6B | 4096 | 2 | 0.6704 | 7/22 | 3/12 |
| KaLM-Embedding-Gemma3-12B-2511 | tencent/KaLM-Embedding-Gemma3-12B-2511 | 11.8B | 3840 | 3 | 0.6701 | **9/22** | 2/12 |
| Octen-Embedding-8B | Octen/Octen-Embedding-8B | 7.6B | 4096 | 5 | 0.6674 | 6/22 | 1/12 |
| Qwen3-Embedding-4B | Qwen/Qwen3-Embedding-4B | 4.0B | 2560 | 6 | 0.6621 | 4/22 | 2/12 |
| SFR-Embedding-Mistral | Salesforce/SFR-Embedding-Mistral | 7.1B | 4096 | 10 | 0.6523 | 6/22 | 2/12 |
| BidirLM-1.7B-Embedding | BidirLM/BidirLM-1.7B-Embedding | 1.7B | 2048 | 11 | 0.6513 | 4/22 | **3/12** |
| BOOM_4B_v1 | ICT-TIME-and-Querit/BOOM_4B_v1 | 4.0B | 2560 | 12 | 0.6503 | 4/22 | 1/12 |
| SFR-Embedding-2_R | Salesforce/SFR-Embedding-2_R | 7.1B | 4096 | 20 | 0.6397 | 4/22 | **4/12** |
| harrier-oss-v1-27b | microsoft/harrier-oss-v1-27b | 27.0B | 5376 | 22 | 0.6390 | 6/22 | 2/12 |
| F2LLM-v2-8B | codefuse-ai/F2LLM-v2-8B | 7.6B | 4096 | 24 | 0.6368 | 6/22 | 1/12 |
| harrier-oss-v1-0.6b | microsoft/harrier-oss-v1-0.6b | 596M | 1024 | 26 | 0.6342 | — | — |
| F2LLM-v2-14B | codefuse-ai/F2LLM-v2-14B | 14.0B | 5120 | 27 | 0.6339 | 7/22 | 2/12 |
| F2LLM-v2-4B | codefuse-ai/F2LLM-v2-4B | 4.0B | 2560 | 29 | 0.6318 | 4/22 | 2/12 |
| F2LLM-v2-1.7B | codefuse-ai/F2LLM-v2-1.7B | 1.7B | 2048 | 42 | 0.6149 | — | — |
| multilingual-e5-large-instruct | intfloat/multilingual-e5-large-instruct | 560M | 1024 | 19 | 0.6409 | 4/22 | 2/12 |
| jina-embeddings-v5-text-small | jinaai/jina-embeddings-v5-text-small | 596M | 1024 | 17 | 0.6435 | 2/22 | 0/12 |
| snowflake-arctic-embed-l-v2.0 | Snowflake/snowflake-arctic-embed-l-v2.0 | 568M | 1024 | 37 | 0.6201 | — | 2/12 |

---

## 4. Analysis

### 4.1 Is there ONE model that is top-tier across many tasks?

**No single open model dominates all weak tasks.** The task × teacher landscape is fragmented:

- **Retrieval tasks** (FaqBacenRetrieval, MedPTRetrieval): dominated by the **F2LLM family** (codefuse-ai), which sweeps all 5 top positions on FaqBacenRetrieval.
- **Biomedical/scientific clustering** (MedPTClustering, SciELOClusteringP2P): **Qwen3-Embedding-4B** and **BidirLM-1.7B** are the clear leaders.
- **General clustering** (WikipediaPT, StackoverflowPt): **Qwen3-Embedding-8B** and **SFR-Embedding-Mistral** lead.
- **Legal tasks** (JurisTCUClusteringP2P, JurisTCUReranking, JurisTCU): harrier-27B and snowflake-arctic lead, but gaps are smaller and all models score low.
- **PortuLexRRIP**: uniquely dominated by pre-trained Portuguese BERT models—a pattern no KD from a general teacher will fix.

The **closest to a universal teacher** by breadth is **KaLM-Embedding-Gemma3-12B-2511** (9/22 tasks in top-5 among open models) and **Qwen3-Embedding-8B** (7/22 tasks). However, neither leads on the most critical weak tasks (FaqBacenRetrieval: Qwen3-8B is rank 5 open; KaLM is rank 7 open).

### 4.2 Embedding Dimensions — Is dim 768 achievable from a strong open model?

**No.** Across all 73 open models on this leaderboard, the dimension landscape is:

| Dim | Typical size | Best open rank in this dim |
|---|---|---|
| **768** | 110M–335M BERT-based | #76 (bert-large-pt) |
| **1024** | 0.6B–0.7B embedding LLMs | #17 (jina-v5-small), #19 (me5-large-instruct) |
| **1152** | 1B Gemma-3-based | #23 (BidirLM-1B) |
| **1536** | 1.5B Qwen2 | #35 (gte-Qwen2-1.5B) |
| **2048** | 1.7B LLM-based | #11 (BidirLM-1.7B) |
| **2560** | 4B LLM-based | #6 (Qwen3-4B) |
| **3584** | 7B Qwen2 | #21 (gte-Qwen2-7B) |
| **3840** | 12B Gemma3 | #3 (KaLM-12B) |
| **4096** | 7–8B LLM-based | #2 (Qwen3-8B) |
| **5120** | 14B LLM-based | #27 (F2LLM-14B) |
| **5376** | 27B | #22 (harrier-27B) |

**Direct KD** (teacher and student share the same dim 768) is not feasible with any strong open teacher. All practical teachers operate at 1024–4096 dimensions.

**Conclusion: Relational KD is mandatory.** Use similarity-space distillation:
- Teacher encodes a batch → compute pairwise cosine similarity matrix
- Student encodes same batch → compute student similarity matrix
- Minimize KL divergence or MSE between the two similarity matrices

No projection head is required for relational KD. Batch composition (including hard negatives) matters more than dimensional alignment.

### 4.3 Teacher Size vs. Distillation Cost

| Teacher | Params | Inference cost for KD | Teacher strength (mean_22) | Notes |
|---|---|---|---|---|
| BidirLM-1.7B | 1.7B | Very low — fits on single 16 GB GPU, fast | 0.6513 | Best value; 1.7B but rank 11 open |
| Qwen3-Embedding-4B | 4.0B | Low — fits on 24 GB GPU with bf16 | 0.6621 | Best on MedPTClustering by large margin |
| F2LLM-v2-4B | 4.0B | Low — same as above | 0.6318 | Retrieval specialist |
| Qwen3-Embedding-8B | 7.6B | Moderate — 2× A100 40 GB or 1× 80 GB | 0.6704 | Best overall open |
| F2LLM-v2-8B | 7.6B | Moderate | 0.6368 | Retrieval specialist, 1st on FaqBacen |
| KaLM-Embedding-Gemma3-12B | 11.8B | High — 1× A100 80 GB or 2× 40 GB | 0.6701 | Broadest coverage, good on SciELO |
| F2LLM-v2-14B | 14.0B | High | 0.6339 | Best on MedPTRetrieval only |
| harrier-oss-v1-27b | 27.0B | Very high — 2× A100 80 GB + quant | 0.6390 | Only top on JurisTCU clustering |

For KD of a 157M student, the marginal gain from a 14B vs. 8B teacher rarely justifies 2× inference cost. The sweet spot is **4B** (excellent signal, tractable cost).

---

## 5. Ranked Teacher Recommendations

### Tier 1 — Recommended Primary Teacher

**Qwen3-Embedding-4B** (`Qwen/Qwen3-Embedding-4B`)
- Overall open rank: **#6**, mean_22 = 0.6621
- Embedding dim: **2560**
- HF availability: Yes, Apache 2.0
- KD type: **Relational** (cosine similarity distillation)
- Key advantages:
  - **Best open model on MedPTClustering** (0.8863 vs gemma 0.7194, gap +0.1669 — largest gap in the benchmark)
  - Top-2 on BRTaxQAR (0.4195, gap +0.0447)
  - Top-4 on StackoverflowPtClustering (0.6255)
  - Top-4 on InferBR (0.8939)
  - Strong overall — rank 6 among 73 open models
  - 4B fits in ~10 GB at bfloat16 for inference; tractable on a single consumer GPU
- Weak spots: Not in top-3 on FaqBacenRetrieval (rank 9 open, 0.6546) or MedPTRetrieval

---

### Tier 2 — Recommended Retrieval Supplement

**F2LLM-v2-4B** (`codefuse-ai/F2LLM-v2-4B`)
- Overall open rank: #29, mean_22 = 0.6318
- Embedding dim: **2560** (same as Qwen3-4B — enables identical projection if needed)
- HF availability: Yes
- KD type: Relational
- Key advantages:
  - **#3 open on FaqBacenRetrieval** (0.8144, gap vs gemma +0.1195)
  - **#5 open on MedPTRetrieval** (0.8628, gap vs gemma +0.0857)
  - **#2 open on JurisTCUClusteringP2P** (0.4382, gap vs gemma +0.1440)
  - Same dim 2560 as Qwen3-4B → if using learned projection, one head serves both
- Rationale for F2LLM-4B over F2LLM-8B: score delta is small (0.8144 vs 0.8229 on FaqBacen) but cost is half; the 4B variant is sufficient for signal

Two-teacher strategy with Qwen3-4B + F2LLM-4B covers the two largest systematic gaps:
- Clustering (MedPT +0.1669, Wikipedia +0.11, SciELO +0.13) → Qwen3-4B
- Retrieval (FaqBacen +0.13, MedPTRetrieval +0.11) → F2LLM-4B

---

### Tier 3 — Budget Single Teacher

**BidirLM-1.7B-Embedding** (`BidirLM/BidirLM-1.7B-Embedding`)
- Overall open rank: **#11**, mean_22 = 0.6513
- Embedding dim: **2048**
- HF availability: Yes
- KD type: Relational
- Key advantages:
  - **#2 open on MedPTClustering** (0.8573, gap vs gemma +0.1379)
  - **#3 open on StackoverflowPtClustering** (0.6291, gap +0.0841)
  - **#2 open on SciELOClusteringP2P** (0.7811, gap +0.0809)
  - Tied for 4th on ToxSynPT
  - **1.7B → fastest KD training; runs on 8 GB GPU; practical for iterative distillation experiments**
  - Mean_22 = 0.6513 is only 0.0023 below Qwen3-8B (0.6704 gap) but at 4× fewer params than 8B models
- Best for: rapid iteration, compute-constrained environments, or as the first distillation step before upgrading to a larger teacher

---

### Tier 4 — If Retrieval is the Only Priority

**F2LLM-v2-8B** (`codefuse-ai/F2LLM-v2-8B`)
- Overall open rank: #24, mean_22 = 0.6368
- Embedding dim: **4096**
- **#1 open on FaqBacenRetrieval** (0.8229), **#4 on MedPTRetrieval** (0.8712)
- Use only if FaqBacenRetrieval improvement is the sole target; the 8B cost is not justified for a mixed-task student

---

### Do NOT Use (Too Large for the Signal)

| Model | Issue |
|---|---|
| harrier-oss-v1-27b (27B) | 27B inference cost; #1 on JurisTCU clustering but only weakly better than F2LLM-4B |
| F2LLM-v2-14B (14B) | Best on MedPTRetrieval but F2LLM-4B achieves 0.8628 vs 14B's 0.8863; 2× cost for 0.024 delta |
| KaLM-Embedding-Gemma3-12B (12B) | Broadest breadth but Qwen3-4B + F2LLM-4B cover the same weak tasks at 1/3 the cost |

---

## 6. Recommended Distillation Strategy

### Setup A — Single Teacher (Simplest)
**Teacher**: Qwen3-Embedding-4B (`Qwen/Qwen3-Embedding-4B`, dim 2560)
**Method**: Relational KD — pairwise cosine similarity distillation

```
Loss = MSE(sim_matrix_student, sim_matrix_teacher)
     + optional: CKA alignment term
```

Batch: sample hard negatives from PT-BR retrieval corpora (MMARCO-PT, FaQuAD, MedPT, FaqBacen) to create informative similarity matrices.
Expected gains: MedPTClustering, SciELO, BRTaxQAR, WikipediaPT, StackoverflowPt.

### Setup B — Two Teachers (Recommended for Broad Coverage)
**Teacher 1**: Qwen3-Embedding-4B (clustering/classification specialist)
**Teacher 2**: F2LLM-v2-4B (retrieval specialist)

Both at dim 2560 — same architecture family (Qwen2-based and Mistral-based respectively).

Strategy: domain-partition batches or task-weighted loss:
```
Loss = λ₁ · L_relational(student, Qwen3-4B, clustering_batch)
     + λ₂ · L_relational(student, F2LLM-4B, retrieval_batch)
```

Optionally interleave by domain (medical, legal, general) within each batch.

### Note on PortuLexRRIP
The best open models for this task are BERTimbau-based models with no strong embedding LLM teacher available. Recommend **separate fine-tuning** on Portuguese legal classification data (not KD) to close this gap.

---

## 7. Quick Reference — Teacher Candidates by Dim

| Dim | Best open model at this dim | mean_22 | Open rank | Viable for KD? |
|---|---|---|---|---|
| 768 | bert-large-portuguese-cased | 0.4675 | #76 | No — weaker than student |
| 1024 | jina-embeddings-v5-text-small (596M) | 0.6435 | #17 | Marginal — only ~0.005 better overall |
| 1024 | multilingual-e5-large-instruct (560M) | 0.6409 | #19 | Marginal but strong on STS/NLI |
| 2048 | BidirLM-1.7B-Embedding (1.7B) | 0.6513 | **#11** | **Yes — recommended budget teacher** |
| 2560 | Qwen3-Embedding-4B (4.0B) | 0.6621 | **#6** | **Yes — primary recommendation** |
| 2560 | F2LLM-v2-4B (4.0B) | 0.6318 | #29 | **Yes — retrieval supplement** |
| 4096 | Qwen3-Embedding-8B (7.6B) | 0.6704 | **#2** | Yes — best single teacher if compute allows |
| 4096 | F2LLM-v2-8B (7.6B) | 0.6368 | #24 | Yes — retrieval only, if FaqBacen is primary |

---

## 8. Data Provenance

- Scores: `mteb-pt/mteb-pt-results` dataset, file `score_matrix.parquet`, revision `fab798de` (last updated 2026-07-01), 93 models × 27 columns (22 task scores + rank/params/type/mean_22).
- Embedding dims: fetched from `config.json` of each model's HF repository via the HF API on 2026-07-02.
- No scores were interpolated or invented; all values are exactly as stored in the parquet.
- Models marked `type = "C"` (closed/API) were excluded from teacher rankings throughout.

---

*Generated 2026-07-02 from live MTEB-PT leaderboard data.*
