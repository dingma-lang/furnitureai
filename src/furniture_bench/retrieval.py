from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .data import digest, load_image, load_manifest, read_json, write_json
from .faiss_worker import FaissWorker


def unit_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("expected a nonempty 2D embedding matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("embeddings must have finite positive norms")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class CatalogIndex:
    def __init__(self, index, metadata):
        self.index, self.metadata = index, metadata

    @classmethod
    def build(cls, vectors, products, embedding):
        vectors = unit_vectors(vectors)
        ids = [row["product_id"] for row in products]
        if len(products) != len(vectors) or len(ids) != len(set(ids)):
            raise ValueError("one embedding per unique product_id is required")
        index = FaissWorker("build", vectors=vectors)
        return cls(
            index,
            {
                "schema_version": 1,
                "metric": "cosine",
                "embedding": embedding,
                "products": products,
                "dimension": vectors.shape[1],
                "faiss_version": index.version,
                "execution": "isolated persistent CPU worker; search seconds include IPC",
            },
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.index.call("save", path=str(path / "index.faiss"))
        self.metadata["index_sha256"] = digest(path / "index.faiss")
        write_json(path / "metadata.json", self.metadata)

    @classmethod
    def load(cls, path):
        """Load locally built trusted indexes only; FAISS files are native binary data."""
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        if metadata.get("schema_version") != 1 or metadata.get("metric") != "cosine":
            raise ValueError("unsupported index metadata")
        if digest(path / "index.faiss") != metadata.get("index_sha256"):
            raise ValueError("index checksum mismatch")
        index = FaissWorker("load", path=str(path / "index.faiss"))
        if index.d != metadata["dimension"] or index.ntotal != len(metadata["products"]):
            index.close()
            raise ValueError("index and product metadata disagree")
        if not index.inner_product:
            index.close()
            raise ValueError("expected an inner-product index")
        return cls(index, metadata)

    def close(self):
        if self.index is not None:
            self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def assert_compatible(self, embedding):
        original = self.metadata["embedding"]
        for key in ("model_id", "requested_revision", "feature", "preprocessing"):
            if original.get(key) != embedding.get(key):
                raise ValueError(f"catalog/query embedding mismatch: {key}; rebuild the index")
        # If both hub commit hashes are known, require the same actual weights.
        old, new = original.get("resolved_revision"), embedding.get("resolved_revision")
        if old and new and old != new:
            raise ValueError("catalog/query resolved model revisions differ")

    def search(self, vectors, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        vectors = unit_vectors(vectors)
        if vectors.shape[1] != self.index.d:
            raise ValueError("query embedding dimension differs from catalog")
        start = perf_counter()
        scores, indices = self.index.search(vectors, min(k, self.index.ntotal))
        results = []
        for row_scores, row_indices in zip(scores, indices):
            matches = []
            for score, index in zip(row_scores, row_indices):
                if index < 0:
                    continue
                product = self.metadata["products"][int(index)]
                matches.append(
                    {
                        "product_id": product["product_id"],
                        "label": product["label"],
                        "file": product["file"],
                        "similarity": float(score),
                        "identity_verified": False,
                    }
                )
            results.append(matches)
        return results, perf_counter() - start


def build_catalog(manifest, output, embedder):
    if Path(output).exists():
        raise ValueError("index output already exists; choose a new directory")
    rows = load_manifest(manifest, "catalog")
    products, vector_batches = [], []
    total_seconds = 0.0
    # Bound image and tensor memory; fail explicitly if any catalog image cannot load.
    for offset in range(0, len(rows), embedder.batch_size):
        batch = rows[offset : offset + embedder.batch_size]
        images = [load_image(row["resolved_file"]) for row in batch]
        vectors, seconds = embedder.encode(images)
        total_seconds += seconds
        vector_batches.append(vectors)
        products.extend(
            {
                **{key: value for key, value in row.items() if key != "resolved_file"},
                "image_sha256": digest(row["resolved_file"]),
            }
            for row in batch
        )
    index = CatalogIndex.build(np.concatenate(vector_batches), products, embedder.metadata())
    index.metadata.update(
        {
            "catalog_manifest_sha256": digest(manifest),
            "embedding_seconds": total_seconds,
            "product_count": len(products),
        }
    )
    try:
        index.save(output)
    finally:
        index.close()
    return index.metadata
