"""Native FAISS runs here, isolated from PyTorch's bundled OpenMP runtime.

Only arrays/metadata cross the process boundary; native index objects stay in this
worker. The worker imports no torch/model code. All index operations are serial.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_INDEX = None


def execute(operation, payload):
    global _INDEX
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('FAISS is missing; install ".[search]"') from exc
    if operation == "build":
        vectors = payload["vectors"]
        _INDEX = faiss.IndexFlatIP(vectors.shape[1])
        _INDEX.add(vectors)
    elif operation == "load":
        _INDEX = faiss.read_index(payload["path"])
    elif operation == "save":
        faiss.write_index(_INDEX, payload["path"])
        return None
    elif operation == "search":
        return _INDEX.search(payload["vectors"], payload["k"])
    else:
        raise ValueError(f"unknown FAISS operation: {operation}")
    return {
        "dimension": _INDEX.d,
        "count": _INDEX.ntotal,
        "inner_product": _INDEX.metric_type == faiss.METRIC_INNER_PRODUCT,
        "version": faiss.__version__,
    }


class FaissWorker:
    def __init__(self, operation, **payload):
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
        try:
            info = self.call(operation, **payload)
        except Exception:
            self.close()
            raise
        self.d, self.ntotal = info["dimension"], info["count"]
        self.inner_product, self.version = info["inner_product"], info["version"]

    def call(self, operation, **payload):
        return self.executor.submit(execute, operation, payload).result()

    def search(self, vectors, k):
        return self.call("search", vectors=vectors, k=k)

    def close(self):
        self.executor.shutdown(wait=True)
