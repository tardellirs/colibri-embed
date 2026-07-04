#!/usr/bin/env bash
# GPU: eval new-trim vs old-trim on MTEB(por) mean_21 (per-task deltas).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_TOKEN="${HF_TOKEN:?}"
cd /root
apt-get install -y python3.12-venv git -qq
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
if [ ! -f venv/.deps ]; then
  pip install -q --upgrade pip
  pip install -q "numpy<2" torch transformers sentence-transformers accelerate mteb
  touch venv/.deps
fi
[ -d mteb-pt ] || git clone --depth=1 https://github.com/tardellirs/mteb-pt.git
pip install -q -e ./mteb-pt/
rm -rf /root/.cache/mteb/results/no_model_name* 2>/dev/null || true
python3 compare_trims.py
echo "COMPARE ALL DONE"
