#!/usr/bin/env python3
"""CPU efficiency benchmark for Colibri variants (run one per fresh process for
clean peak-RSS). Two modes:
  --variant {fp32,fp16,onnx,onnx-int8}  -> weight size, load RAM, latency, throughput, peak RSS
  --dims                                -> Matryoshka footprint: encode latency + vector bytes per dim
Appends JSON lines to /root/cpu_bench_results.jsonl (variants) / cpu_dims.jsonl (dims).
"""
import argparse, glob, json, os, resource, statistics, time
from sentence_transformers import SentenceTransformer

RID = os.getenv("MODEL_ID", "tardellirs/colibri-embed-ptbr")
TOK = os.getenv("HF_TOKEN")
OUT_FILE = os.getenv("OUT_FILE", "/root/cpu_bench_results.jsonl")
DIMS_FILE = os.getenv("DIMS_FILE", "/root/cpu_dims.jsonl")

SENTS = [
    "Como declarar imposto de renda de aluguel recebido de pessoa física?",
    "O Banco Central regula as instituições financeiras do país.",
    "Sintomas de dengue incluem febre alta, dores no corpo e manchas na pele.",
    "O Supremo Tribunal Federal julgou a constitucionalidade da lei.",
    "A fotossíntese converte luz solar em energia química nas plantas.",
    "Contratos de trabalho regidos pela CLT garantem direitos ao empregado.",
    "Pesquisadores publicaram um artigo sobre inteligência artificial.",
    "O Pix permite transferências instantâneas a qualquer hora do dia.",
]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux KB -> MB


def _find(pattern):
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/" + pattern), recursive=True)
    return max((os.path.getsize(h) for h in hits), default=0) / 1e6 if hits else 0.0


def load(variant):
    if variant == "fp32":
        return SentenceTransformer(RID, token=TOK), _find("model.safetensors")
    if variant == "fp16":
        try:
            m = SentenceTransformer(RID, revision="fp16", token=TOK)
        except Exception:
            m = SentenceTransformer(RID, token=TOK).half()
        sizes = sorted(os.path.getsize(h) for h in
                       glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/model.safetensors"), recursive=True))
        return m, (sizes[0] / 1e6 if sizes else 0.0)
    if variant == "onnx":
        m = SentenceTransformer(RID, backend="onnx", model_kwargs={"export": True}, token=TOK)
        return m, _find("onnx/model.onnx")
    if variant == "onnx-int8":
        from sentence_transformers import export_dynamic_quantized_onnx_model
        m = SentenceTransformer(RID, backend="onnx", model_kwargs={"export": True}, token=TOK)
        export_dynamic_quantized_onnx_model(m, "avx512_vnni", RID)  # writes qint8 into local onnx dir
        qs = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/onnx/*qint8*.onnx"), recursive=True)
        fn = "onnx/" + os.path.basename(qs[0])
        m2 = SentenceTransformer(RID, backend="onnx", model_kwargs={"file_name": fn}, token=TOK)
        return m2, os.path.getsize(qs[0]) / 1e6
    raise ValueError(variant)


def bench_encode(model):
    for _ in range(3):
        model.encode("aquecimento", normalize_embeddings=True)
    lat = []
    for _ in range(50):
        s = time.perf_counter(); model.encode(SENTS[0], normalize_embeddings=True)
        lat.append((time.perf_counter() - s) * 1000)
    batch = (SENTS * 64)[:512]
    s = time.perf_counter(); model.encode(batch, batch_size=32, normalize_embeddings=True)
    thr = len(batch) / (time.perf_counter() - s)
    return statistics.median(lat), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant"); ap.add_argument("--dims", action="store_true")
    a = ap.parse_args()

    if a.dims:  # Matryoshka footprint on the fast ONNX model
        rows = []
        for dim in [768, 512, 256, 128]:
            m = SentenceTransformer(RID, backend="onnx", model_kwargs={"export": True},
                                    truncate_dim=dim, token=TOK)
            lat, thr = bench_encode(m)
            rows.append({"dim": dim, "latency_ms_p50": round(lat, 1),
                         "throughput_sent_per_s": round(thr, 1), "vector_bytes_fp32": dim * 4})
            print(json.dumps(rows[-1]), flush=True)
        with open(DIMS_FILE, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        return

    v = a.variant
    t0 = time.time(); model, wmb = load(v); load_s = time.time() - t0
    load_rss = rss_mb()
    lat, thr = bench_encode(model)
    row = {"variant": v, "weight_MB": round(wmb, 1), "load_s": round(load_s, 1),
           "load_RSS_MB": round(load_rss, 0), "latency_ms_p50": round(lat, 1),
           "throughput_sent_per_s": round(thr, 1), "peak_RSS_MB": round(rss_mb(), 0)}
    print(json.dumps(row), flush=True)
    with open(OUT_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
