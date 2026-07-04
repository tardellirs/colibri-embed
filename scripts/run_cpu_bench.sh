#!/usr/bin/env bash
# CPU (VPS-like): efficiency per variant + Matryoshka footprint. Cleans stray procs before each.
set -euo pipefail
HF_TOKEN="${HF_TOKEN:?}"
cd /root
apt-get install -y python3.12-venv git -qq
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
if [ ! -f venv/.deps ]; then
  pip install -q --upgrade pip
  pip install -q "numpy<2" torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q transformers sentence-transformers "optimum[onnxruntime]" onnxruntime
  touch venv/.deps
fi
: > /root/cpu_bench_results.jsonl
echo "cpu info: $(nproc) vCPU, RAM $(free -m | awk '/Mem:/{print $2}') MB"
clean() { for p in $(pgrep -f "cpu_bench\.py|cpu_int8_quality\.py" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; sleep 2; }
for V in fp32 fp16 onnx onnx-int8; do
  clean
  echo "=== bench $V | stray python3: $(pgrep -c -x python3 2>/dev/null || echo 0) ==="
  python3 cpu_bench.py --variant "$V" || echo "  $V FAILED"
done
clean
echo "=== Matryoshka dims footprint (onnx) ==="
python3 cpu_bench.py --dims || echo "dims FAILED"
clean
echo "=== int8 quality vs fp32 ==="
python3 cpu_int8_quality.py || echo "int8 quality FAILED"
echo "CPU-BENCH ALL DONE"
echo "--- variants ---"; cat /root/cpu_bench_results.jsonl
echo "--- dims ---"; cat /root/cpu_dims.jsonl 2>/dev/null
