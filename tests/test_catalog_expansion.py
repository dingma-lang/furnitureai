import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from furniture_bench.catalog_expansion import (
    BudgetExceeded,
    DownloadStore,
    incremental_embeddings,
    validate_catalog,
    validate_nested_galleries,
    validate_source,
)
from furniture_bench.data import digest
from furniture_bench.product_catalog import MultiImageCatalog


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), "ETag": '"fixture"', **(headers or {})}

    def getcode(self):
        return self.status

    def geturl(self):
        return "https://example.test/object.png"


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = {
            "name": "UNIT TEST",
            "license": "test fixture",
            "license_evidence": "fixture",
            "attribution": "fixture",
            "commercial_training": True,
            "commercial_evaluation": True,
            "retain_images": True,
            "allowed_hosts": ["example.test"],
        }
        output = io.BytesIO()
        Image.new("RGB", (20, 20), "red").save(output, format="PNG")
        self.payload = output.getvalue()

    def test_license_requires_all_intended_uses(self):
        for field in [
            "commercial_training",
            "commercial_evaluation",
            "retain_images",
            "license_evidence",
        ]:
            source = {**self.source, field: False}
            with self.assertRaises(ValueError):
                validate_source(source)

    def test_incremental_download_reuses_verified_bytes(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            return Response(self.payload)

        store = DownloadStore(self.root, self.source, opener=opener)
        first = store.fetch("id", "https://example.test/object.png")
        second = store.fetch("id", "https://example.test/object.png")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(store.state["transferred_bytes"], len(self.payload))

    def test_interrupted_download_resumes_exact_range(self):
        cut = len(self.payload) // 2
        calls = []

        def opener(request, timeout):
            calls.append(request.headers)
            if len(calls) == 1:
                return Response(
                    self.payload[:cut], headers={"Content-Length": str(len(self.payload))}
                )
            self.assertEqual(request.get_header("Range"), f"bytes={cut}-")
            return Response(
                self.payload[cut:],
                206,
                {"Content-Range": f"bytes {cut}-{len(self.payload) - 1}/{len(self.payload)}"},
            )

        store = DownloadStore(self.root, self.source, opener=opener)
        with self.assertRaises(OSError):
            store.fetch("id", "https://example.test/object.png")
        resumed = DownloadStore(self.root, self.source, opener=opener)
        result = resumed.fetch("id", "https://example.test/object.png")
        self.assertEqual(Path(result["file"]).read_bytes(), self.payload)
        self.assertEqual(result["resumed_from_bytes"], cut)

    def test_server_ignores_range_restarts_without_concatenating(self):
        count = []

        def opener(request, timeout):
            count.append(1)
            return (
                Response(self.payload[:10], headers={"Content-Length": str(len(self.payload))})
                if len(count) == 1
                else Response(self.payload)
            )

        store = DownloadStore(self.root, self.source, opener=opener)
        with self.assertRaises(OSError):
            store.fetch("id", "https://example.test/object.png")
        result = store.fetch("id", "https://example.test/object.png")
        self.assertEqual(Path(result["file"]).read_bytes(), self.payload)
        self.assertTrue(result["server_restarted_full_transfer"])
        self.assertEqual(store.state["transferred_bytes"], len(self.payload) + 10)

    def test_invalid_resume_range_preserves_partial(self):
        count = []

        def opener(request, timeout):
            count.append(1)
            return (
                Response(self.payload[:10], headers={"Content-Length": str(len(self.payload))})
                if len(count) == 1
                else Response(self.payload[10:], 206, {"Content-Range": "bytes 11-30/40"})
            )

        store = DownloadStore(self.root, self.source, opener=opener)
        with self.assertRaises(OSError):
            store.fetch("id", "https://example.test/object.png")
        with self.assertRaises(ValueError):
            store.fetch("id", "https://example.test/object.png")
        self.assertEqual(
            next((self.root / "images").glob("*.part")).read_bytes(), self.payload[:10]
        )

    def test_budget_prevents_oversized_payload_and_extra_file(self):
        store = DownloadStore(
            self.root,
            self.source,
            max_files=1,
            max_bytes=len(self.payload) - 1,
            opener=lambda *a, **k: Response(self.payload),
        )
        with self.assertRaises(BudgetExceeded):
            store.fetch("id", "https://example.test/object.png")
        self.assertEqual(store.state["transferred_bytes"], 0)
        with self.assertRaises(BudgetExceeded):
            store.fetch("second", "https://example.test/second.png")

    def test_unknown_length_budget_never_exceeds_cap(self):
        def opener(*a, **k):
            response = Response(self.payload)
            del response.headers["Content-Length"]
            return response

        store = DownloadStore(self.root, self.source, max_bytes=20, opener=opener)
        with self.assertRaises(BudgetExceeded):
            store.fetch("id", "https://example.test/object.png")
        self.assertEqual(store.state["transferred_bytes"], 20)

    def test_unapproved_source_and_changed_file_rejected(self):
        store = DownloadStore(self.root, self.source, opener=lambda *a, **k: Response(self.payload))
        with self.assertRaises(ValueError):
            store.fetch("id", "https://retailer.test/unauthorized.png")
        result = store.fetch("id", "https://example.test/object.png")
        Path(result["file"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            store.fetch("id", "https://example.test/object.png")

    def test_incremental_embeddings_bind_hash_and_model_and_resume(self):
        rows = []
        for n in range(2):
            path = self.root / f"image-{n}.png"
            Image.new("RGB", (10, 10), (n * 50, 30, 50)).save(path)
            rows.append({"image_id": str(n), "file": str(path), "sha256": digest(path)})
        calls = []

        def encoder(paths):
            calls.append(len(paths))
            return np.tile(np.array([[1.0, 0.0]]), (len(paths), 1)), 0.01

        first, a = incremental_embeddings(rows[:1], self.root / "features", "model1", encoder)
        second, b = incremental_embeddings(rows, self.root / "features", "model1", encoder)
        self.assertEqual(b["reused"], 1)
        self.assertEqual(b["new_encodings"], 1)
        self.assertEqual(calls, [1, 1])
        _, c = incremental_embeddings(rows, self.root / "features", "model2", encoder)
        self.assertEqual(c["new_encodings"], 2)
        Path(rows[0]["file"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            incremental_embeddings(rows, self.root / "features", "model1", encoder)

    def test_invalid_cached_or_nonfinite_features_rejected(self):
        path = self.root / "image.png"
        Image.new("RGB", (10, 10)).save(path)
        rows = [{"image_id": "id", "file": str(path), "sha256": digest(path)}]
        with self.assertRaises(ValueError):
            incremental_embeddings(
                rows, self.root / "features", "m", lambda p: (np.array([[np.nan, 0.0]]), 0)
            )

    def test_interrupted_embedding_job_reuses_completed_batches(self):
        rows = []
        for n in range(2):
            path = self.root / f"resume-{n}.png"
            Image.new("RGB", (10, 10), (n * 50, 20, 30)).save(path)
            rows.append({"image_id": str(n), "file": str(path), "sha256": digest(path)})
        calls = []

        def interrupted(paths):
            calls.append(len(paths))
            if len(calls) == 2:
                raise RuntimeError("simulated interruption; no model used")
            return np.array([[1.0, 0.0]], dtype=np.float32), 0

        with self.assertRaises(RuntimeError):
            incremental_embeddings(
                rows, self.root / "resume-features", "model", interrupted, batch_size=1
            )
        _, report = incremental_embeddings(
            rows,
            self.root / "resume-features",
            "model",
            lambda p: (np.array([[1.0, 0.0]], dtype=np.float32), 0),
            batch_size=1,
        )
        self.assertEqual(report["reused"], 1)
        self.assertEqual(report["new_encodings"], 1)

    def test_nested_identity_and_query_leakage_checks(self):
        def product(pid):
            return {
                "product_id": pid,
                "images": [{"image_id": pid + "-ref", "sha256": pid + "-hash"}],
            }

        queries = [
            {
                "product_id": "A",
                "image_id": "query",
                "sha256": "qhash",
                "true_product_in_catalog": True,
            },
            {
                "product_id": "U",
                "image_id": "unknown",
                "sha256": "uhash",
                "true_product_in_catalog": False,
            },
        ]
        validate_nested_galleries({1: [product("A")], 2: [product("A"), product("B")]}, queries)
        for wrong in [
            {1: [product("B")]},
            {2: [product("A"), product("U")]},
            {2: [product("A"), product("A")]},
        ]:
            with self.assertRaises(ValueError):
                validate_nested_galleries(wrong, queries)
        leaked = product("A")
        leaked["images"][0]["sha256"] = "qhash"
        with self.assertRaises(ValueError):
            validate_nested_galleries({1: [leaked]}, queries)

    def test_faiss_rebuild_retains_variants_and_unique_products(self):
        path = self.root / "image.png"
        Image.new("RGB", (10, 10)).save(path)
        products = [
            {
                "product_id": pid,
                "label": "sofa",
                "brand": "same",
                "model_number": "reported-same",
                "rights": {"license": "fixture", "allowed_for_benchmark": True},
                "images": [{"image_id": pid, "file": str(path), "sha256": digest(path)}],
            }
            for pid in ["variant-A", "variant-B"]
        ]
        catalog = {"schema_version": 2, "products": products}
        self.assertEqual(validate_catalog(catalog)["products"], 2)
        index = MultiImageCatalog.build(
            np.eye(2, dtype=np.float32), catalog, {"fixture": True}, self.root
        )
        try:
            index.save(self.root / "index")
        finally:
            index.close()
        loaded = MultiImageCatalog.load(self.root / "index")
        try:
            result, _ = loaded.search(np.array([[1.0, 0.0]]), k=5)
            self.assertEqual(result[0][0]["product_id"], "variant-A")
            self.assertEqual(len({r["product_id"] for r in result[0]}), 2)
            self.assertIsNone(result[0][0]["manufacturer"])
        finally:
            loaded.close()


if __name__ == "__main__":
    unittest.main()
