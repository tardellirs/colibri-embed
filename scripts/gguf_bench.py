"""llama-server with a GGUF: peak and current RSS of the server process, latency, cosine vs the fp32 HF embeddings.
python scripts/gguf_bench.py model.gguf  (after cpu_mem_bench.py fp32 wrote emb_fp32.npy; LLAMA_ARGS overrides the server flags)"""
import json, os, statistics, subprocess, sys, time, urllib.request
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
gg = sys.argv[1]
SENTS = [l.strip() for l in open(f"{D}/bench_sents.txt") if l.strip()]
p = subprocess.Popen([os.path.expanduser("~/llama.cpp/build/bin/llama-server"), "-m", gg, "--embeddings", "--pooling", "mean",
                      "-t", "4", *os.environ.get("LLAMA_ARGS", "-c 2048 -ub 2048 -b 2048").split(), "--port", "8091", "--log-disable"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def post(body):
    r = urllib.request.Request("http://127.0.0.1:8091/v1/embeddings", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return [d["embedding"] for d in json.load(urllib.request.urlopen(r))["data"]]
for _ in range(120):
    try:
        post({"input": "aquecimento"}); break
    except Exception:
        time.sleep(0.5)
st = lambda k: int(next(l for l in open(f"/proc/{p.pid}/status") if l.startswith(k)).split()[1]) / 1024
load_rss = st("VmRSS")
lat = []
for s in SENTS:
    t = time.perf_counter(); post({"input": s}); lat.append((time.perf_counter() - t) * 1000)
batch = (SENTS * 18)[:512]
t = time.perf_counter()
for i in range(0, len(batch), 32):
    post({"input": batch[i:i + 32]})
thr = len(batch) / (time.perf_counter() - t)
e = np.array(post({"input": SENTS}))
e /= np.linalg.norm(e, axis=1, keepdims=True)
ref = np.load(f"{D}/emb_fp32.npy")
cos = (e * ref).sum(1)
print(json.dumps({"mode": os.path.basename(gg), "size_MB": round(os.path.getsize(gg) / 1e6), "rss_after_load_MB": round(load_rss),
                  "peak_end_MB": round(st("VmHWM")), "rss_end_MB": round(st("VmRSS")), "p50_ms": round(statistics.median(lat), 1),
                  "sent_per_s": round(thr, 1), "cos_min": round(float(cos.min()), 4), "cos_mean": round(float(cos.mean()), 4)}), flush=True)
p.terminate()
