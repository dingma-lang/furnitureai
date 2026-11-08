"""Offline, CPU-only structural tests. Do not inspect official holdout features."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("furniture_auto_train", HERE / "auto_train.py")
agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = agent
spec.loader.exec_module(agent)


def fixtures(groups_per_class=4):
    rows = []
    for cls in agent.CLASSES:
        for gid in range(groups_per_class):
            for view in range(2):
                rows.append({"image_id": f"{cls}-{gid}-{view}",
                    "abo_product_id": f"product-{cls}-{gid}",
                    "group_id": f"group-{cls}-{gid}",
                    "family_group_id": None,
                    "capture_cluster_id": None,
                    "decoded_pixel_sha256": f"pixel-{cls}-{gid}-{view}",
                    "proposed_split": "train", "weak_train_eligible": True,
                    "furniture_class": cls})
    return rows


class GroupingTests(unittest.TestCase):
    def test_fold_coverage_product_isolation(self):
        rows = fixtures()
        folds, groups = agent.create_folds(rows)
        self.assertEqual(len(groups), 7 * 4)
        self.assertEqual({i for fold in folds for i in fold}, set(range(len(rows))))
        self.assertTrue(all(len(x) > 0 for x in folds))
        product_fold = {}
        for fold_no, f in enumerate(folds):
            for i in f:
                pid = rows[i]["abo_product_id"]
                self.assertEqual(product_fold.setdefault(pid, fold_no), fold_no)

    def test_cross_product_shared_photo_is_connected(self):
        rows = fixtures()
        # Two distinct product IDs but the same decoded pixels => one connected component.
        rows[0]["decoded_pixel_sha256"] = rows[2]["decoded_pixel_sha256"]
        folds, groups = agent.create_folds(rows)
        f0 = next(j for j, f in enumerate(folds) if 0 in f)
        self.assertIn(2, folds[f0])
        self.assertEqual(len(groups), 27)

    def test_conflicting_group_category_fails(self):
        rows = fixtures()
        rows[0]["group_id"] = rows[8]["group_id"]
        with self.assertRaisesRegex(ValueError, "Conflicting classes"):
            agent.create_folds(rows)

    def test_unapproved_image_fails(self):
        rows = fixtures()
        rows[0]["proposed_split"] = "test"
        with self.assertRaisesRegex(ValueError, "Non-approved"):
            agent.create_folds(rows)

    def test_known_holdout_overlap_rejected(self):
        rows = fixtures()
        holdout = [{**rows[0], "image_id": "holdout-0", "proposed_split": "test"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            agent.check_cross_partition_group_overlap(rows, holdout)

    def test_metric_scope(self):
        rows = fixtures()
        sample = [i for i, r in enumerate(rows) if r["abo_product_id"] in
                  {"product-sofa-0", "product-chair-0"}]
        logits = np.zeros((len(sample), len(agent.CLASSES)), np.float32)
        for j, i in enumerate(sample):
            logits[j, agent.CLASSES.index(rows[i]["furniture_class"])] = 10
        result = agent.score_logits(rows, sample, logits)
        self.assertEqual(result["product_count"], 2)
        self.assertEqual(result["image_count"], 4)
        self.assertEqual(result["image"]["accuracy"], 1.0)
        self.assertEqual(result["product"]["accuracy"], 1.0)

    def test_json_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.json"
            agent.write_json(p, {"epoch": 5, "done": False})
            self.assertEqual(agent.read_json(p)["epoch"], 5)
            self.assertFalse((Path(d) / "state.json.tmp").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
