"""Phase10E.12 regressions; synthetic files only, no model or source-data access."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from furniture_bench.data import digest, load_manifest, write_json
from furniture_bench.evaluation import in_ignored_region, match_objects
from furniture_bench.geometry import intersection_area, iou, nms
from furniture_bench.postprocessing import containment


class ManifestFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="furniture-e12-hash-")
        self.root = Path(self.temp.name)
        self.actual = self.root / "actual.bin"
        self.decoy = self.root / "decoy.bin"
        self.actual.write_bytes(b"synthetic actual file")
        self.decoy.write_bytes(b"synthetic different file")
        self.manifest = self.root / "rooms.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_room(self, fingerprint, resolved=None, file="actual.bin"):
        row = {
            "image_id": "synthetic-room",
            "file": file,
            "image_sha256": fingerprint,
            "rights": {
                "source": "synthetic CPU fixture",
                "license": "CC0",
                "allowed_for_benchmark": True,
            },
            "objects": None,
        }
        if resolved is not None:
            row["resolved_file"] = str(resolved)
        write_json(self.manifest, {"schema_version": 1, "images": [row]})

    def test_decoy_resolved_file_cannot_validate_other_image(self):
        self.write_room(digest(self.decoy), resolved=self.decoy)
        with self.assertRaisesRegex(ValueError, "image bytes differ"):
            load_manifest(self.manifest, "rooms")

    def test_actual_hash_wins_over_untrusted_resolved_file(self):
        self.write_room(digest(self.actual), resolved=self.decoy)
        row = load_manifest(self.manifest, "rooms")[0]
        self.assertEqual(row["resolved_file"], str(self.actual.resolve()))
        self.assertEqual(digest(row["resolved_file"]), row["image_sha256"])

    def test_cached_resolved_file_still_works_when_consistent(self):
        self.write_room(digest(self.actual), resolved=self.actual)
        self.assertEqual(load_manifest(self.manifest, "rooms")[0]["resolved_file"], str(self.actual))

    def test_missing_consumed_file_cannot_hide_behind_existing_decoy(self):
        self.write_room(digest(self.decoy), resolved=self.decoy, file="missing.bin")
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.manifest, "rooms")

    def test_relative_and_absolute_file_names_resolve_to_same_hash(self):
        self.write_room(digest(self.actual))
        relative = load_manifest(self.manifest, "rooms")[0]
        self.write_room(digest(self.actual), file=str(self.actual))
        absolute = load_manifest(self.manifest, "rooms")[0]
        self.assertEqual(relative["resolved_file"], absolute["resolved_file"])
        self.assertEqual(relative["image_sha256"], absolute["image_sha256"])
        self.assertEqual(relative["file"], "actual.bin")
        self.assertEqual(absolute["file"], str(self.actual))

    def test_symlink_file_uses_actual_target(self):
        alias = self.root / "alias.bin"
        alias.symlink_to(self.actual)
        self.write_room(digest(self.actual), file="alias.bin", resolved=self.decoy)
        row = load_manifest(self.manifest, "rooms")[0]
        self.assertEqual(row["resolved_file"], str(self.actual))

    def test_manifest_and_source_bytes_are_unchanged_by_validation(self):
        self.write_room(digest(self.actual))
        before = [p.read_bytes() for p in (self.manifest, self.actual, self.decoy)]
        load_manifest(self.manifest, "rooms")
        self.assertEqual(before, [p.read_bytes() for p in (self.manifest, self.actual, self.decoy)])


class GeometryPolicyTests(unittest.TestCase):
    def test_disjoint_touching_and_nested_boxes(self):
        a = [0, 0, 10, 10]
        self.assertEqual(intersection_area(a, [10, 0, 20, 10]), 0)
        self.assertEqual(intersection_area(a, [2, 2, 4, 4]), 4)
        self.assertEqual(containment(a, [2, 2, 4, 4]), 1)
        self.assertEqual(iou(a, [2, 2, 4, 4]), 0.04)

    def test_intersection_retains_integer_precision(self):
        edge = 2**54 + 1
        self.assertEqual(intersection_area([0, 0, edge, 2], [0, 0, edge, 2]), edge * 2)

    def test_ignore_coverage_boundary_still_inclusive(self):
        pred = {"box": [0, 0, 10, 10]}
        self.assertTrue(in_ignored_region(pred, [{"box": [0, 0, 5, 10]}]))
        self.assertFalse(in_ignored_region(pred, [{"box": [0, 0, 4.999, 10]}]))

    def test_nms_threshold_still_strict_and_categories_separate(self):
        a = {"label": "chair", "score": 0.8, "box": [0, 0, 10, 10]}
        b = {"label": "chair", "score": 0.8, "box": [0, 0, 5, 10]}
        other = {**a, "label": "table"}
        self.assertEqual(nms([a, b, other], 0.5), [a, b, other])

    def test_matching_tie_order_and_one_to_one_policy_unchanged(self):
        pred = {"label": "chair", "score": 0.8, "box": [0, 0, 10, 10]}
        gt = {"label": "chair", "box": [0, 0, 10, 10]}
        matches, extra, missed = match_objects([pred, pred], [gt, gt], 0.5)
        self.assertEqual(matches, [(0, 1, 1.0), (1, 0, 1.0)])
        self.assertEqual((extra, missed), ([], []))


if __name__ == "__main__":
    unittest.main()
