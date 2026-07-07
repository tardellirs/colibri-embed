#!/usr/bin/env bash
# Colibri-v2: multi-teacher relational KD on the +stack re-trim base.
#   precompute Qwen3-4B + Qwen3-8B -> avg similarity target -> KD -> soup + merge -> eval.
# Spot-safe: HF checkpoints (resume); precompute skipped if embs already present.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_TOKEN="${HF_TOKEN:?}"
SMOKE_TEST="${SMOKE_TEST:-0}"
cd /root
CORPUS=data/distill_v2_corpus.parquet
BASE=tardellirs/embeddinggemma-pt-br-64k-retrim-stack-test
HUB=tardellirs/colibri-v2-distill
T4B="Qwen/Qwen3-Embedding-4B"; T8B="Qwen/Qwen3-Embedding-8B"
export COLIBRI_BASE="$BASE"

apt-get install -y python3.12-venv git -qq
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
if [ ! -f venv/.deps ]; then
  pip install -q --upgrade pip
  pip install -q "numpy<2" torch transformers sentence-transformers accelerate mteb
  touch venv/.deps
fi
[ -d mteb-br ] || git clone --depth=1 https://github.com/tardellirs/mteb-br.git
pip install -q -e ./mteb-br/

if [ "$SMOKE_TEST" = "1" ]; then
  echo "=== SMOKE: precompute 512 (4B+8B) + smoke multi-teacher train ==="
  python3 distill_precompute.py --teacher "$T4B" --corpus-parquet "$CORPUS" --limit 512 --batch-size 16 --emb-out emb_4b_smoke.npy --out-dir ./data
  python3 distill_precompute.py --teacher "$T8B" --corpus-parquet "$CORPUS" --limit 512 --batch-size 8  --emb-out emb_8b_smoke.npy --out-dir ./data
  python3 distill_train.py --corpus data/distill_corpus_smoke.parquet \
    --teacher-emb data/emb_4b_smoke.npy --teacher-emb-2 data/emb_8b_smoke.npy --base "$BASE" \
    --output ./ckpt_smoke --smoke-test --batch-size 64 --hub-repo "${HUB}-smoke" \
    --eval-every-epoch --eval-include Assin2STS MedPTClustering
  echo "=== SMOKE OK: precompute(2 teachers)+multi-teacher train+HF push+mid-eval worked ==="
  exit 0
fi

# --- precompute both teachers (skip if present -> resume after preemption) ---
[ -f data/emb_4b.npy ] || { echo "=== precompute 4B ==="; python3 distill_precompute.py --teacher "$T4B" --corpus-parquet "$CORPUS" --batch-size 32 --emb-out emb_4b.npy --out-dir ./data; }
[ -f data/emb_8b.npy ] || { echo "=== precompute 8B ==="; python3 distill_precompute.py --teacher "$T8B" --corpus-parquet "$CORPUS" --batch-size 12 --emb-out emb_8b.npy --out-dir ./data; }

echo "=== train multi-teacher KD (base=retrim-stack) ==="
python3 distill_train.py --corpus "$CORPUS" \
  --teacher-emb data/emb_4b.npy --teacher-emb-2 data/emb_8b.npy --base "$BASE" \
  --output ./ckpt_v2 --epochs 2 --batch-size 256 --lr 5e-6 --save-steps 88 \
  --hub-repo "$HUB" --eval-every-epoch --merge-alpha 0.3 --eval-batch-size 128

echo "=== soup + merge(alpha sweep) + full-22 vs v1(0.6497) ==="
rm -rf /root/.cache/mteb/results/no_model_name* 2>/dev/null || true
python3 soup_eval.py --ckpts-dir ./ckpt_v2/ckpts --alphas 0.1 0.15 0.2 0.25 0.3 --greedy \
  --hub-repo "$HUB" --out ./v2_soup_results
echo "=== DISTILL-V2 DONE ==="
