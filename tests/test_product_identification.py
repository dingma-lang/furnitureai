import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from furniture_bench.product_catalog import MultiImageCatalog, aggregate_products
from room_viewer.photo_projects import ProjectStore
from room_viewer.product_matching import ProductMatching, observation_signature
from scripts.audit_abo_queries import fingerprint


def products():
    return [{"product_id": "a", "label": "sofa", "retailer": "Amazon", "images": [{"image_id": "a1", "file": "a1.png", "sha256": "a"}, {"image_id": "a2", "file": "a2.png", "sha256": "b"}], "specifications": {}}, {"product_id": "b", "label": "chair", "retailer": "IKEA", "images": [{"image_id": "b1", "file": "b.png", "sha256": "c"}], "name": None}]


class ProductAggregationTests(unittest.TestCase):
    def test_white_margin_crop_audit_detects_resized_export_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import ImageDraw
            original = Image.new("RGB", (140, 120), "white")
            draw = ImageDraw.Draw(original)
            draw.rectangle([30, 20, 110, 100], fill=(50, 80, 110))
            draw.line([40, 30, 100, 90], fill=(190, 120, 60), width=7)
            a, b = Path(tmp)/"a.png", Path(tmp)/"b.png"
            original.save(a)
            original.crop((30, 20, 111, 101)).save(b)
            ha, ra = fingerprint(a)
            hb, rb = fingerprint(b)
            self.assertEqual(int(np.count_nonzero(ha != hb)), 0)
            self.assertLess(float(np.mean(abs(ra-rb))), 1)

    def test_multiple_image_hits_collapse_to_one_identity(self):
        hits = [{"dataset_product_id": "a", "image_id": "a1", "similarity": .9}, {"dataset_product_id": "a", "image_id": "a2", "similarity": .95}, {"dataset_product_id": "b", "image_id": "b1", "similarity": .94}]
        result = aggregate_products(hits, products())
        self.assertEqual([r["product_id"] for r in result], ["a", "b"])
        self.assertEqual(result[0]["best_image_id"], "a2")
        self.assertFalse(result[0]["identity_verified"])
        self.assertIsNone(result[1]["name"])
        self.assertIsNone(result[1]["manufacturer"])
        self.assertIsNone(result[1]["price"])
        self.assertEqual(aggregate_products(hits, products(), category="bed"), [])
        self.assertEqual(aggregate_products(hits, products(), ["Target"], retailer_only=True), [])
        preferred = aggregate_products(hits, products(), ["IKEA"])
        self.assertEqual(preferred[0]["product_id"], "b")
        self.assertEqual(preferred[0]["similarity"], .94)

    def test_real_isolated_faiss_roundtrip_preserves_product_image_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = {"products": products()}
            vectors = np.array([[1, 0], [.9, .1], [0, 1]], np.float32)
            index = MultiImageCatalog.build(vectors, catalog, {"model_id": "synthetic-test-only"}, tmp)
            try:
                matches, _ = index.search(np.array([[1, 0]], np.float32))
                self.assertEqual([r["product_id"] for r in matches[0]], ["a", "b"])
                index.save(Path(tmp) / "index")
            finally:
                index.close()
            restored = MultiImageCatalog.load(Path(tmp) / "index")
            try:
                result, _ = restored.search(np.array([[0, 1]], np.float32))
                self.assertEqual(result[0][0]["product_id"], "b")
            finally:
                restored.close()

    def test_checkpoint_declared_architecture_is_used_without_large_model(self):
        from furniture_bench.embedding.siglip2 import Siglip2Embedder
        model, processor = Mock(), Mock()
        model.to.return_value.eval.return_value = model
        with patch("transformers.AutoModel.from_pretrained", return_value=model) as load, patch("transformers.AutoImageProcessor.from_pretrained", return_value=processor), patch("furniture_bench.embedding.siglip2.resolve_device", return_value="cpu"):
            embedder = Siglip2Embedder("synthetic-checkpoint", allow_download=False)
            embedder.load()
            load.assert_called_once_with("synthetic-checkpoint", use_safetensors=True, local_files_only=True, revision="main")


class ProductProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ProjectStore(self.tmp.name)
        self.pid = self.store.create("Synthetic product integration")["id"]
        memory = io.BytesIO()
        Image.new("RGB", (64, 48)).save(memory, "PNG")
        p = self.store.upload(self.pid, memory.getvalue(), "room.png", True)
        self.photo_id = p["photos"][0]["id"]
        p = self.act(type="correct", operation="add", photo_id=self.photo_id, label="sofa", box=[1, 2, 40, 40])
        self.did = p["photos"][0]["reviewed"][0]["detection_id"]
        self.act(type="correct", operation="confirm", photo_id=self.photo_id, detection_id=self.did)
        p = self.store.read(self.pid)
        photo, detection = p["photos"][0], p["photos"][0]["reviewed"][0]
        self.candidate = aggregate_products([{"dataset_product_id": "a", "image_id": "a1", "similarity": .8}], products())[0]
        photo["product_matches"] = {self.did: {"state": "complete", "observation_signature": observation_signature(photo, detection), "candidates": [self.candidate]}}
        self.store.write(p)

    def act(self, **body):
        return self.store.action(self.pid, body)

    def choice(self, decision):
        return self.act(type="product_choice", photo_id=self.photo_id, detection_id=self.did, product_id="a", decision=decision)

    def test_confirmation_persists_but_is_not_an_exact_identity_claim(self):
        before = copy.deepcopy(self.store.read(self.pid)["photos"][0]["reviewed"])
        p = self.choice("confirm")
        entry = p["inventory"][self.photo_id + ":" + self.did]
        self.assertFalse(entry["identity_verified"])
        self.assertFalse(entry["geometry_applied"])
        self.assertEqual(p["photos"][0]["reviewed"], before)
        reopened = self.store.import_bundle(self.store.export(self.pid))
        self.assertEqual(reopened["inventory"], p["inventory"])
        rejected = self.choice("reject")
        self.assertEqual(rejected["inventory"], {})

    def test_stale_or_unconfirmed_observation_cannot_save_product(self):
        self.act(type="correct", operation="update", photo_id=self.photo_id, detection_id=self.did, box=[2, 2, 40, 40])
        with self.assertRaisesRegex(ValueError, "Confirm"):
            self.choice("confirm")
        self.act(type="correct", operation="confirm", photo_id=self.photo_id, detection_id=self.did)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.choice("confirm")

    def test_product_details_attach_to_placeholder_without_changing_geometry(self):
        self.choice("confirm")
        self.act(type="setup_room", layout="rectangular", dimensions=[6, 4, 3], dimension_status="unknown", source="Test example", approximate_acknowledged=True)
        p = self.act(type="create_placeholder", observation_id=self.photo_id + ":" + self.did, size=[2, 1, 1], position=[3, 2, 0], dimension_status="unknown", source="Example", approximate_acknowledged=True)
        item = p["room"]["furniture"][0]
        self.assertEqual(item["product_details"]["candidate"]["product_id"], "a")
        self.assertEqual(item["size"], [2, 1, 1])
        p = self.choice("reject")
        self.assertEqual(p["room"]["furniture"][0]["product_details"]["status"], "user_rejected")

    def test_worker_failure_preserves_recognition_and_current_room(self):
        p = self.store.read(self.pid)
        photo = p["photos"][0]
        photo["product_matches"][self.did].update(cache_key="synthetic", settings={})
        self.store.write(p)
        matching = ProductMatching(self.store, SimpleNamespace(pool=None))
        with patch("room_viewer.product_matching.subprocess.run", return_value=SimpleNamespace(returncode=2)):
            matching.execute(self.pid, self.photo_id, copy.deepcopy(photo["reviewed"][0]), "synthetic", "signature", {})
        p = self.store.read(self.pid)
        self.assertEqual(p["photos"][0]["product_matches"][self.did]["state"], "error")
        self.assertTrue(p["photos"][0]["reviewed"][0]["confirmed"])
        self.assertEqual(p["room"]["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
