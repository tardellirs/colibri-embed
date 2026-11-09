"""RSS of a bare onnxruntime session on the exported model (no torch, no sentence-transformers), with and without graph optimizations.
python scripts/ort_session_mem.py <saved_dir> {opt,noopt}"""
import os, resource, sys
import psutil
import onnxruntime as ort
base = psutil.Process().memory_info().rss / 2**20
so = ort.SessionOptions(); so.intra_op_num_threads = 4
if sys.argv[2] == "noopt":
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(f"{sys.argv[1]}/onnx/model.onnx", so, providers=["CPUExecutionProvider"])
import numpy as np
ids = np.array([[2, 246, 16547, 1585, 3550, 664, 170, 1]], dtype=np.int64)
feeds = {i.name: (ids if "input_ids" in i.name else np.ones_like(ids)) for i in s.get_inputs()}
for _ in range(5):
    s.run(None, feeds)
print(sys.argv[2], "rss before session", round(base), "MB | after", round(psutil.Process().memory_info().rss / 2**20),
      "| peak", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "| onnx file", round(os.path.getsize(f"{sys.argv[1]}/onnx/model.onnx") / 2**20), "MB")
