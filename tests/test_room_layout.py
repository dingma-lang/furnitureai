"""Offline synthetic geometry/reference checks; never download model weights."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from furniture_bench.data import digest, read_json, write_json
from room_layout.corrections import apply_corrections
from room_layout.scene import photo_scene, require_metric_scene, validate_photo_scene
from room_reconstruction.demo import demo_scene, replacement_sofa
from room_reconstruction.model import propose_replacement
from room_reconstruction.photo_estimates import attach_photo_estimate

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from room_layout.evaluation import (
        boundary_metrics,
        corner_metrics,
        evaluate_layout,
        mask_counts,
    )
    from room_layout.geometry import (
        boundary_mask,
        corner_candidates,
        extract_structure,
        fit_segments,
        infer_occlusion_bridges,
        mask_rings,
    )
    from room_layout.pipeline import _cached_records, run_layout


def partial_scene():
    return photo_scene(
        "synthetic",
        "a" * 64,
        160,
        100,
        {"regions": [], "camera_perspective": {"status": "unknown"}},
        {},
        [],
    )


def correction():
    return {
        "view_id": "synthetic",
        "image_sha256": "a" * 64,
        "reviewer": "synthetic-test",
        "reason": "synthetic fixture only",
    }


def segment(identifier, points, kind="wall_floor"):
    return {
        "id": identifier,
        "endpoints": points,
        "kind": kind,
        "visibility": "visible_evidence",
        "confidence": 0.9,
    }


class PartialRoomTests(unittest.TestCase):
    def test_unknown_geometry_cannot_fit(self):
        scene = partial_scene()
        self.assertIsNone(scene["floor"]["world_polygon"])
        self.assertEqual(scene["floor"]["status"], "unknown_no_visible_region")
        self.assertIsNone(scene["coordinate_frames"]["transforms"]["world_to_metres"])
        with self.assertRaisesRegex(ValueError, "not a metric room"):
            require_metric_scene(scene)

    def test_detection_candidates_reuse_observation_format_without_3d(self):
        d = {"detection_id": "owl-1", "label": "sofa", "score": 0.7, "box": [1, 2, 90, 80]}
        s = photo_scene(
            "test", "b" * 64, 100, 100, {"regions": [], "camera_perspective": {}}, {}, [d]
        )
        self.assertEqual(s["furniture_observations"][0]["image_regions"][0]["box_xyxy"], d["box"])
        self.assertIsNone(s["furniture_entities"][0]["physical_dimensions"])
        self.assertIsNone(s["furniture_entities"][0]["world_position"])

    def test_corrections_preserve_original_and_do_not_enable_fit(self):
        s, c = partial_scene(), correction()
        before = copy.deepcopy(s)
        c.update(
            corner_corrections=[{"pixel": [10, 20]}],
            boundary_corrections=[{"points": [[0, 10], [100, 20]]}],
            geometry_verdict="accepted_as_estimate",
            measurements={
                "camera_height": {
                    "status": "measured",
                    "value": 1.6,
                    "units": "m",
                    "source": "test tape",
                    "evidence": "synthetic",
                    "confidence": 0.9,
                }
            },
        )
        changed = apply_corrections(s, c)
        self.assertEqual(s, before)
        self.assertEqual(changed["measurements"]["camera_height"]["value"], 1.6)
        self.assertFalse(changed["fit_readiness"]["ready"])
        self.assertFalse(changed["physical_accuracy_verified"])

    def test_invalid_corrections_rejected(self):
        for point in ([float("nan"), 1], [-1, 2], [160, 2], [True, 1]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                apply_corrections(
                    partial_scene(), dict(correction(), corner_corrections=[{"pixel": point}])
                )
        with self.assertRaisesRegex(ValueError, "different view"):
            apply_corrections(partial_scene(), dict(correction(), image_sha256="b" * 64))

    def test_measurement_provenance_units_and_confidence(self):
        valid = {
            "status": "measured",
            "value": 2.8,
            "units": "m",
            "source": "tape",
            "evidence": "synthetic fixture",
            "confidence": 0.9,
        }
        for override in (
            {"value": -1},
            {"value": float("inf")},
            {"evidence": None},
            {"source": None},
            {"confidence": 2},
            {"units": "pixels"},
        ):
            s = partial_scene()
            s["measurements"]["ceiling_height"] = dict(valid, **override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_photo_scene(s)

    def test_known_wall_length_retains_unknown_projection(self):
        c = dict(
            correction(),
            wall_measurements=[
                {
                    "reference": "user wall A",
                    "status": "measured",
                    "value": 4.2,
                    "units": "m",
                    "source": "tape",
                    "evidence": "synthetic",
                    "confidence": 0.95,
                }
            ],
        )
        s = apply_corrections(partial_scene(), c)
        self.assertEqual(s["wall_measurements"][0]["value"], 4.2)
        self.assertIsNone(s["floor"]["world_polygon"])

    def test_camera_matrix_does_not_imply_physical_verification(self):
        c = dict(
            correction(),
            camera={
                "intrinsics": [[500, 0, 80], [0, 500, 50], [0, 0, 1]],
                "source": "synthetic camera",
            },
        )
        s = apply_corrections(partial_scene(), c)
        self.assertEqual(
            s["views"][0]["camera"]["calibration_status"], "externally_supplied_unverified"
        )
        self.assertFalse(s["fit_readiness"]["ready"])
        c["camera"]["intrinsics"] = [[0] * 3] * 3
        with self.assertRaises(ValueError):
            apply_corrections(partial_scene(), c)

    def test_multiple_views_have_unique_ids(self):
        s = partial_scene()
        s["views"].append(copy.deepcopy(s["views"][0]))
        with self.assertRaisesRegex(ValueError, "unique view"):
            validate_photo_scene(s)
        s["views"][1]["view_id"] = "second"
        validate_photo_scene(s)

    def test_metric_replacement_preserves_photo_evidence_and_geometry(self):
        room = demo_scene()
        with self.assertRaisesRegex(ValueError, "demo_only"):
            attach_photo_estimate(room, partial_scene(), "test", "synthetic")
        attached = attach_photo_estimate(room, partial_scene(), "test", "synthetic", demo_only=True)
        after, result = propose_replacement(attached, "sofa-original", replacement_sofa())
        self.assertTrue(result["accepted"])
        for key in ("floor", "walls", "openings", "dimensions"):
            self.assertEqual(after[key], room[key])
        self.assertEqual(after["photo_structure_estimates"], attached["photo_structure_estimates"])
        self.assertNotIn("photo_structure_estimates", room)


@unittest.skipUnless(HAS_SCIPY, "optional scipy layout dependency not installed")
class LayoutGeometryTests(unittest.TestCase):
    def test_concave_polygon_and_hole_remain(self):
        mask = np.ones((12, 12), bool)
        mask[7:, 7:] = False
        mask[2:4, 2:4] = False
        rings = mask_rings(mask, epsilon=0)
        self.assertEqual(sum(r["pixel_area_signed"] for r in rings), int(mask.sum()))
        self.assertEqual({r["ring_role"] for r in rings}, {"outer", "hole"})
        self.assertGreater(len(next(r for r in rings if r["ring_role"] == "outer")["points"]), 4)

    def test_boundary_needs_adjacent_classes_and_confidence(self):
        labels = np.ones((30, 80), np.uint8)
        labels[15:] = 2
        confidence, occlusion = np.ones_like(labels, float), np.zeros_like(labels, bool)
        mask = boundary_mask(labels, 1, 2, confidence, 0.55, occlusion)
        self.assertEqual(mask.sum(), 80)
        occlusion[:, 20:50] = True
        confidence[:, :10] = 0.1
        mask = boundary_mask(labels, 1, 2, confidence, 0.55, occlusion)
        self.assertEqual(mask.sum(), 40)
        lines = fit_segments(mask, confidence)
        self.assertEqual(len(lines), 1)  # 10px fragment is below minimum length.
        self.assertLessEqual(max(p[0] for p in lines[0]["endpoints"]), 80)

    def test_corner_needs_two_supported_ends(self):
        lines = [segment("a", [[10, 20], [50, 50]]), segment("b", [[50, 50], [90, 20]])]
        self.assertEqual(len(corner_candidates(lines, 100, 100)), 1)
        lines[1]["endpoints"] = [[70, 35], [90, 20]]
        self.assertEqual(corner_candidates(lines, 100, 100, tolerance=5), [])

    def test_inferred_bridge_requires_occlusion_and_two_sides(self):
        lines = [segment("a", [[0, 10], [30, 10]]), segment("b", [[60, 10], [90, 10]])]
        hints = np.zeros((50, 100), bool)
        self.assertEqual(infer_occlusion_bridges(lines, hints), [])
        hints[8:13, 30:61] = True
        inferred = infer_occlusion_bridges(lines, hints)
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["visibility"], "inferred_occluded")
        self.assertEqual(infer_occlusion_bridges(lines[:1], hints), [])

    def test_no_structure_does_not_fabricate_room(self):
        image = Image.new("RGB", (100, 80))
        structure, _ = extract_structure(
            image, np.zeros((80, 100), np.uint8), np.ones((80, 100)), {"0": "sofa"}
        )
        self.assertEqual(structure["regions"], [])
        self.assertEqual(structure["boundaries"], [])
        self.assertIsNone(structure["topology"]["closed_world_floor_polygon"])
        self.assertIsNone(structure["camera_perspective"]["horizon_line"])


@unittest.skipUnless(HAS_SCIPY, "optional scipy layout dependency not installed")
class LayoutEvaluationTests(unittest.TestCase):
    def test_metrics_on_synthetic_references(self):
        self.assertEqual(mask_counts(np.array([[2, 2, 1]]), np.array([[2, 255, 2]]), 2), (1, 2))
        result = boundary_metrics([[[0, 10], [30, 10]]], [[[0, 13], [30, 13]]], (40, 30))
        self.assertAlmostEqual(result["symmetric_alignment_error_pixels"], 3)
        self.assertEqual(result["recall_within_tolerance"], 1)
        result = corner_metrics([[1, 0], [0, 1]], [[0, 0]])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(corner_metrics([], [[0, 0]])["recall"], 0)
        self.assertEqual(
            boundary_metrics([], [[[0, 1], [5, 1]]], (10, 10))["recall_within_tolerance"], 0
        )

    def fixture(self, root):
        Image.new("RGB", (100, 80), "white").save(root / "image.png")
        manifest = root / "rooms.json"
        write_json(
            manifest,
            {
                "schema_version": 1,
                "images": [
                    {
                        "image_id": "synthetic",
                        "file": "image.png",
                        "rights": {
                            "allowed_for_benchmark": True,
                            "source": "test generated",
                            "license": "CC0",
                        },
                    }
                ],
            },
        )
        for stage, key, arrays in (
            (
                "segmentation",
                "seg",
                {"labels": np.ones((80, 100), np.uint8), "confidence": np.ones((80, 100))},
            ),
            ("depth", "depth", {"relative_depth": np.ones((80, 100), np.float32)}),
        ):
            folder = root / key
            folder.mkdir()
            np.savez_compressed(folder / "000.npz", **arrays)
            write_json(
                folder / "report.json",
                {
                    "stage": stage,
                    "model": {"path": str(folder), "id2label": {"1": "floor"}},
                    "images": [
                        {
                            "image_id": "synthetic",
                            "status": "ok",
                            "file": "000.npz",
                            "image_sha256": digest(root / "image.png"),
                        }
                    ],
                },
            )
        return manifest

    def test_cached_pipeline_no_model_download_and_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            with patch(
                "room_layout.pipeline.subprocess.run",
                side_effect=AssertionError("no model worker expected"),
            ):
                run = run_layout(
                    manifest,
                    root / "out",
                    root / "seg",
                    root / "depth",
                    reuse_segmentation=[root / "seg"],
                    reuse_depth=[root / "depth"],
                )
            self.assertEqual(run["images"][0]["status"], "ok")
            self.assertEqual(
                run["images"][0]["structure_status"], "insufficient_floor_or_wall_evidence"
            )
            report = evaluate_layout(manifest, root / "out/run.json", root / "metrics.json")
            self.assertIsNone(report["floor_iou"])
            self.assertEqual(report["unreviewed_images"], 1)
            with self.assertRaisesRegex(ValueError, "human-reviewed"):
                evaluate_layout(manifest, root / "out/run.json", root / "strict.json", True)
            self.assertFalse((root / "strict.json").exists())

    def test_cache_rejects_wrong_checkpoint_stage_or_image(self):
        from furniture_bench.data import load_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = load_manifest(self.fixture(root), "rooms")
            for stage, model in (("depth", root / "seg"), ("segmentation", root / "other")):
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    _cached_records([root / "seg"], rows, stage, model)
            Image.new("RGB", (100, 80), "black").save(root / "image.png")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                _cached_records([root / "seg"], rows, "segmentation", root / "seg")

    def test_failed_inference_counts_missed_verified_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            d = read_json(manifest)
            mask = np.ones((80, 100), np.uint8)
            mask[40:] = 2
            Image.fromarray(mask).save(root / "reference.png")
            d["images"][0]["layout_annotations"] = {
                "status": "human_reviewed",
                "human_reviewed": True,
                "reviewer": "synthetic fixture",
                "reviewed_at": "synthetic date",
                "evidence": "synthetic only, not a real human annotation",
                "mask_file": "reference.png",
                "corners": [{"pixel": [5, 5]}],
                "boundaries": [{"kind": "wall_floor", "points": [[0, 40], [99, 40]]}],
            }
            write_json(manifest, d)
            write_json(
                root / "run.json",
                {
                    "manifest_sha256": digest(manifest),
                    "images": [
                        {
                            "image_id": "synthetic",
                            "status": "error",
                            "errors": ["synthetic failure"],
                        }
                    ],
                },
            )
            result = evaluate_layout(manifest, root / "run.json", root / "report.json", True)
            self.assertEqual(result["floor_iou"], 0)
            self.assertEqual(result["wall_iou"], 0)
            self.assertEqual(result["reviewed_reference_processing_failures"], 1)
            self.assertEqual(result["images"][0]["corners"]["recall"], 0)


if __name__ == "__main__":
    unittest.main()
