#!/usr/bin/env bash
# Parameterized CPU bench: MODEL_ID + MODEL_TAG + VARIANTS. Reuses existing venv.
set -euo pipefail
HF_TOKEN="${HF_TOKEN:?}"
export MODEL_ID="${MODEL_ID:-tardellirs/colibri-embed-ptbr}"
MODEL_TAG="${MODEL_TAG:-colibri}"
VARIANTS="${VARIANTS:-fp32 fp16 onnx onnx-int8}"
export OUT_FILE="/root/cpu_bench_${MODEL_TAG}.jsonl"
export DIMS_FILE="/root/cpu_dims_${MODEL_TAG}.jsonl"
cd /root
source venv/bin/activate
: > "$OUT_FILE"
echo "=== MODEL $MODEL_ID ($MODEL_TAG) | cpu: $(nproc) vCPU, $(free -m | awk '/Mem:/{print $2}') MB ==="
clean() { for p in $(pgrep -f "cpu_bench\.py" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; sleep 2; }
for V in $VARIANTS; do
  clean
  echo "=== [$MODEL_TAG] bench $V | stray python3: $(pgrep -c -x python3 2>/dev/null || echo 0) ==="
  python3 cpu_bench.py --variant "$V" || echo "  $V FAILED"
done
clean
echo "=== [$MODEL_TAG] dims footprint ==="
python3 cpu_bench.py --dims || echo "dims FAILED"
echo "CPU-BENCH ALL DONE ($MODEL_TAG)"
cat "$OUT_FILE"; echo "--- dims ---"; cat "$DIMS_FILE" 2>/dev/null
