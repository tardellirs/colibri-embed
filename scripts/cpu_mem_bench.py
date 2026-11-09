"""CPU RAM / latency / quality of the PyTorch and ONNX paths. One mode per fresh process:
python scripts/cpu_mem_bench.py <mode> <saved_dir>
  fp32     torch model from the Hub
  export   backend="onnx" with export=True (how the first benchmark ran, before the repo had onnx/model.onnx), then save to OUT
  onnx     the saved model.onnx, no export
  noarena  same, with the ORT CPU memory arena off
  int8     onnx/model_qint8.onnx made by quantize (dynamic int8)
  quantize write model_qint8.onnx next to the saved model.onnx (quantize2: per-channel, MatMul only -> int8pc)
  int8pc   onnx/model_qint8_pc.onnx
Prints peak RSS after load, peak after encode, and current RSS at the end; writes embeddings for a quality check."""
import gc, json, os, resource, statistics, sys, time

import numpy as np
import psutil

RID, OUT = "tardellirs/colibri-embed-ptbr", sys.argv[2]
mode = sys.argv[1]
SENTS = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "bench_sents.txt")) if l.strip()]
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 if sys.platform == "darwin" else 1 / 1024) / 2**20
cur = lambda: psutil.Process().memory_info().rss / 2**20

if mode.startswith("quantize"):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    name, kw = ("model_qint8", {}) if mode == "quantize" else ("model_qint8_pc", {"per_channel": True, "op_types_to_quantize": ["MatMul"]})
    quantize_dynamic(f"{OUT}/onnx/model.onnx", f"{OUT}/onnx/{name}.onnx", weight_type=QuantType.QInt8, **kw)
    print(name, os.path.getsize(f"{OUT}/onnx/{name}.onnx") / 1e6, "MB")
    sys.exit()

from sentence_transformers import SentenceTransformer

t0 = time.time()
if mode == "fp32":
    m = SentenceTransformer(RID)
elif mode == "export":
    m = SentenceTransformer(RID, backend="onnx", model_kwargs={"export": True})
elif mode == "onnx":
    m = SentenceTransformer(OUT, backend="onnx")
elif mode == "noarena":
    import onnxruntime as ort
    so = ort.SessionOptions(); so.enable_cpu_mem_arena = False
    m = SentenceTransformer(OUT, backend="onnx", model_kwargs={"session_options": so})
elif mode == "int8":
    m = SentenceTransformer(OUT, backend="onnx", model_kwargs={"file_name": "onnx/model_qint8.onnx"})
elif mode == "int8pc":
    m = SentenceTransformer(OUT, backend="onnx", model_kwargs={"file_name": "onnx/model_qint8_pc.onnx"})
load_s, load_peak = time.time() - t0, peak()
# transformers 4.x rebuilds Gemma's post-processor without <eos>; 5.x (which the model was saved with) and llama.cpp keep it
if hasattr(m.tokenizer, "add_eos_token") and not m.tokenizer("a")["input_ids"][-1] == m.tokenizer.eos_token_id:
    m.tokenizer.add_eos_token = True
assert m.tokenizer("a")["input_ids"][-1] == m.tokenizer.eos_token_id
gc.collect()
load_cur = cur()

for _ in range(3):
    m.encode("aquecimento")
lat = []
for s in SENTS[:50]:
    t = time.perf_counter(); m.encode(s); lat.append((time.perf_counter() - t) * 1000)
batch = (SENTS * 8)[:512]
t = time.perf_counter(); emb = m.encode(batch, batch_size=32, normalize_embeddings=True); thr = len(batch) / (time.perf_counter() - t)
gc.collect()
print(json.dumps({"mode": mode, "load_s": round(load_s, 1), "peak_after_load_MB": round(load_peak), "rss_after_load_MB": round(load_cur),
                  "peak_end_MB": round(peak()), "rss_end_MB": round(cur()), "p50_ms": round(statistics.median(lat), 1), "sent_per_s": round(thr, 1)}), flush=True)
np.save(f"{os.path.dirname(__file__)}/emb_{mode}.npy", m.encode(SENTS, normalize_embeddings=True))
if mode == "export":
    m.save_pretrained(OUT)
