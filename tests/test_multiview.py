"""Offline synthetic geometry tests; genuine experiment results are reported separately."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from furniture_bench.data import digest, write_json
from room_layout.scene import photo_scene
from room_multiview.project import establish_scale, merge_document, photo_guidance
from room_reconstruction.demo import demo_scene, replacement_sofa
from room_reconstruction.model import propose_replacement
from room_reconstruction.photo_estimates import attach_photo_estimate

HAS_GEOMETRY = all(importlib.util.find_spec(m) for m in ("cv2", "scipy", "matplotlib"))
if HAS_GEOMETRY:
    from room_multiview.features import extract_features, match_features
    from room_multiview.geometry import (
        Policy,
        calibration,
        centre,
        coverage,
        estimate_pair,
        project,
        register_camera,
        rotation_error,
        triangulate,
    )
    from room_multiview.pipeline import reconstruct, run_project
    from room_multiview.surfaces import plane_patches


def document(view_id="a"):
    s = photo_scene(
        view_id,
        "a" * 64,
        640,
        480,
        {
            "regions": [],
            "boundaries": [],
            "corner_candidates": [],
            "camera_perspective": {"status": "unknown"},
        },
        {},
        [],
    )
    s["room_id"] = "synthetic-room"
    return s


def measurement(value=3):
    return {
        "status": "measured",
        "value": value,
        "units": "m",
        "source": "synthetic tape",
        "evidence": "synthetic fixture only",
        "confidence": 0.95,
    }


class ProgressiveStateTests(unittest.TestCase):
    def test_editable_metric_room_keeps_multiview_reference_during_replacement(self):
        photo = document()
        photo["shared_geometry"] = {
            "landmarks": [{"id": "p", "position": [1, 2, 3]}],
            "units": "arbitrary",
        }
        original = demo_scene()
        attached = attach_photo_estimate(
            original, photo, "synthetic-test", "compatibility only", demo_only=True
        )
        after, result = propose_replacement(attached, "sofa-original", replacement_sofa())
        self.assertTrue(result["accepted"])
        self.assertEqual(after["walls"], original["walls"])
        self.assertEqual(
            after["photo_structure_estimates"][0]["shared_geometry"], photo["shared_geometry"]
        )
        self.assertFalse(after["photo_structure_estimates"][0]["metric_geometry_applied"])

    def test_measurements_corrections_associations_and_input_preserved(self):
        first = document()
        first["measurements"]["room_width"] = measurement()
        first["views"][0]["user_corrections"] = {"corners": [[10, 20]]}
        first["furniture_entities"] = [
            {"id": "reviewed-sofa", "metric_room_entity_id": "sofa-in-real-room"}
        ]
        before = copy.deepcopy(first)
        merged = merge_document(first, document("b"))
        self.assertEqual(first, before)
        self.assertEqual(merged["measurements"]["room_width"], measurement())
        self.assertEqual(
            merged["views"][0]["user_corrections"], first["views"][0]["user_corrections"]
        )
        self.assertEqual(merged["furniture_entities"], first["furniture_entities"])
        self.assertEqual(len(merged["views"]), 2)
        self.assertFalse(merged["fit_readiness"]["ready"])

    def test_conflicting_measurement_is_flagged_not_merged(self):
        first, incoming = document(), document("b")
        first["measurements"]["room_width"] = measurement(3)
        incoming["measurements"]["room_width"] = measurement(4)
        result = merge_document(first, incoming)
        self.assertEqual(result["measurements"]["room_width"]["value"], 3)
        self.assertEqual(result["evidence_conflicts"][0]["kind"], "measurement_conflict")

    def test_wall_length_conflict_preserves_confirmed_value(self):
        first, incoming = document(), document("b")
        first["wall_measurements"] = [dict(measurement(3), reference="wall A")]
        incoming["wall_measurements"] = [dict(measurement(4), reference="wall A")]
        merged = merge_document(first, incoming)
        self.assertEqual(merged["wall_measurements"], first["wall_measurements"])
        self.assertEqual(len(merged["evidence_conflicts"]), 1)

    def test_changed_image_identity_refused(self):
        a, b = document(), document()
        b["views"][0]["image_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "image bytes"):
            merge_document(a, b)

    def test_source_documents_and_previous_review_are_retained(self):
        first, newer = document(), document()
        first["review"] = {"verdict": "rejected", "reviewer": "synthetic"}
        first["views"][0]["user_corrections"] = {"boundary": [[1, 2], [3, 4]]}
        newer["views"][0]["camera"]["intrinsics"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = merge_document(first, newer)
        self.assertEqual(result["review"], first["review"])
        self.assertEqual(result["views"][0], first["views"][0])
        self.assertIn("a", result["view_documents"])

    def scaled_fixture(self):
        state = document()
        state["shared_geometry"] = {
            "metric_scale": None,
            "landmarks": [{"id": "p", "position": [0, 0, 0]}, {"id": "q", "position": [2, 0, 0]}],
        }
        spec = dict(measurement(4), kind="landmark_distance", landmark_ids=["p", "q"])
        return state, spec

    def test_measured_distance_establishes_transform_without_rewriting_map(self):
        state, spec = self.scaled_fixture()
        original = copy.deepcopy(state)
        result = establish_scale(state, spec)
        self.assertEqual(state, original)
        self.assertEqual(result["shared_geometry"]["metric_scale"]["metres_per_unit"], 2)
        self.assertEqual(
            result["shared_geometry"]["landmarks"], original["shared_geometry"]["landmarks"]
        )
        self.assertIsNone(result["floor"]["world_polygon"])
        self.assertFalse(result["fit_readiness"]["ready"])
        self.assertFalse(result["physical_accuracy_verified"])

    def test_invalid_or_unassociated_scale_refused(self):
        state, spec = self.scaled_fixture()
        for bad in (
            {"value": float("nan")},
            {"value": True},
            {"units": "pixels"},
            {"evidence": ""},
            {"landmark_ids": ["p", "missing"]},
            {"kind": "room_width"},
            {"confidence": 2},
            {"landmark_ids": ["p", "p"]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                establish_scale(state, dict(spec, **bad))

    def test_new_conflicting_scale_does_not_replace_existing_scale(self):
        state, spec = self.scaled_fixture()
        scaled = establish_scale(state, spec)
        before = copy.deepcopy(scaled)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            establish_scale(scaled, dict(spec, value=7))
        self.assertEqual(scaled, before)

    def test_guidance_is_heuristic_and_does_not_count_hidden_walls(self):
        guidance = photo_guidance(document())
        self.assertIsNone(guidance["observations"][0]["world_surface_coverage"])
        self.assertTrue(guidance["unknown_wall_count"])
        self.assertTrue(
            all(r["status"] == "heuristic_not_verified" for r in guidance["recommendations"])
        )


@unittest.skipUnless(HAS_GEOMETRY, "optional multiview dependencies absent")
class MultiViewGeometryTests(unittest.TestCase):
    def test_inconsistent_existing_map_stops_fusion(self):
        _, _, features, assets = self.fixture()
        pairs = {
            "views": ["a", "b"],
            "matches": [[i, i] for i in range(len(features["a"]["pixels"]))],
            "geometry": estimate_pair(
                features["a"]["pixels"],
                features["b"]["pixels"],
                assets["a"]["calibration"],
                assets["b"]["calibration"],
                (640, 480),
                (640, 480),
            ),
        }
        state = reconstruct(
            document(), {v: assets[v] for v in ("a", "b")}, features, [pairs], Policy()
        )
        before = copy.deepcopy(state["shared_geometry"])
        wrong = copy.deepcopy(pairs)
        wrong["geometry"]["camera_b_from_a"][0][3] *= -1
        result = reconstruct(state, assets, features, [wrong], Policy())
        self.assertEqual(result["reconstruction_status"], "inconsistent_geometry_requires_review")
        self.assertEqual(result["shared_geometry"], before)
        self.assertEqual(
            result["geometric_conflicts"][-1]["action"], "preserved_existing_map_and_stopped_fusion"
        )

    def fixture(self):
        cal = calibration(
            {
                "intrinsics": [[520, 0, 320], [0, 520, 240], [0, 0, 1]],
                "distortion": [0] * 5,
                "status": "measured",
                "source": "synthetic camera",
            }
        )
        rng = np.random.default_rng(41)
        points = rng.uniform([-2, -1.5, 4], [2, 1.5, 8], (180, 3))
        poses = {"a": np.eye(4), "b": np.eye(4), "c": np.eye(4)}
        poses["b"][0, 3] = -1
        poses["c"][:3, 3] = [-0.4, -0.15, -0.1]
        features = {v: {"pixels": project(points, p, cal)} for v, p in poses.items()}
        assets = {
            v: {
                "calibration": cal,
                "size": [640, 480],
                "image_sha256": v * 64,
                "_labels": None,
                "_confidence": None,
            }
            for v in poses
        }
        return points, poses, features, assets

    def test_two_view_pose_with_outliers(self):
        _, poses, f, assets = self.fixture()
        second = f["b"]["pixels"].copy()
        second[-30:] = second[-30:][::-1]
        result = estimate_pair(
            f["a"]["pixels"],
            second,
            assets["a"]["calibration"],
            assets["b"]["calibration"],
            (640, 480),
            (640, 480),
        )
        self.assertEqual(result["status"], "accepted")
        self.assertGreater(result["triangulated_inliers"], 100)
        self.assertLess(
            rotation_error(np.array(result["camera_b_from_a"])[:3, :3], poses["b"][:3, :3]), 0.1
        )
        self.assertEqual(result["scale"], "unknown")

    def test_triangulation_known_synthetic_points(self):
        points, poses, f, assets = self.fixture()
        xyz, good, _, errors = triangulate(
            f["a"]["pixels"],
            f["b"]["pixels"],
            poses["a"],
            poses["b"],
            assets["a"]["calibration"],
            assets["b"]["calibration"],
            Policy(),
        )
        self.assertTrue(good.all())
        np.testing.assert_allclose(xyz, points, atol=1e-8)
        self.assertLess(errors.max(), 1e-8)

    def test_tiny_baseline_has_no_usable_parallax(self):
        points, poses, f, assets = self.fixture()
        poses["b"][0, 3] = -0.00001
        second = project(points, poses["b"], assets["b"]["calibration"])
        _, good, _, _ = triangulate(
            f["a"]["pixels"],
            second,
            poses["a"],
            poses["b"],
            assets["a"]["calibration"],
            assets["b"]["calibration"],
            Policy(),
        )
        self.assertFalse(good.any())

    def test_pure_rotation_not_a_room_reconstruction(self):
        points, poses, f, assets = self.fixture()
        theta = 0.12
        p = np.eye(4)
        p[:3, :3] = [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
        result = estimate_pair(
            f["a"]["pixels"],
            project(points, p, assets["a"]["calibration"]),
            assets["a"]["calibration"],
            assets["a"]["calibration"],
            (640, 480),
            (640, 480),
        )
        self.assertEqual(result["status"], "rejected")

    def test_missing_intrinsics_matches_only_no_3d(self):
        _, _, f, _ = self.fixture()
        result = estimate_pair(
            f["a"]["pixels"], f["b"]["pixels"], None, None, (640, 480), (640, 480)
        )
        self.assertEqual(result["reason"], "camera_intrinsics_unknown_no_euclidean_3d")
        self.assertNotIn("points", result)

    def test_calibration_rejects_invalid_or_unsourced_input(self):
        _, _, _, assets = self.fixture()
        valid = assets["a"]["calibration"]
        for bad in (
            {"intrinsics": [[0] * 3] * 3},
            {"distortion": [float("nan")] * 5},
            {"source": ""},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calibration(dict(valid, **bad))

    def test_pnp_registers_existing_frame_and_rejects_mismatches(self):
        points, poses, f, assets = self.fixture()
        result = register_camera(points, f["c"]["pixels"], assets["c"]["calibration"], (640, 480))
        self.assertEqual(result["status"], "accepted")
        np.testing.assert_allclose(
            centre(result["camera_from_world"]), centre(poses["c"]), atol=1e-5
        )
        pixels = np.random.default_rng(10).uniform([0, 0], [640, 480], (len(points), 2))
        bad = register_camera(points, pixels, assets["c"]["calibration"], (640, 480))
        self.assertEqual(bad["status"], "rejected")

    def test_progressive_third_view_keeps_map_frame_and_landmarks(self):
        _, _, f, assets = self.fixture()
        state = document()
        state["measurements"]["ceiling_height"] = measurement(2.8)
        state["views"][0]["user_corrections"] = {"corner": [2, 3]}

        def pair(a, b):
            return {
                "views": [a, b],
                "matches": [[i, i] for i in range(len(f[a]["pixels"]))],
                "geometry": estimate_pair(
                    f[a]["pixels"],
                    f[b]["pixels"],
                    assets[a]["calibration"],
                    assets[b]["calibration"],
                    (640, 480),
                    (640, 480),
                ),
            }

        ab = pair("a", "b")
        state = reconstruct(state, {v: assets[v] for v in ("a", "b")}, f, [ab], Policy())
        frame = state["shared_geometry"]["frame_id"]
        before = copy.deepcopy(state["shared_geometry"]["landmarks"])
        state = reconstruct(state, assets, f, [ab, pair("a", "c"), pair("b", "c")], Policy())
        self.assertEqual(len(state["shared_geometry"]["cameras"]), 3)
        self.assertEqual(state["shared_geometry"]["frame_id"], frame)
        self.assertEqual(
            [p["position"] for p in state["shared_geometry"]["landmarks"][: len(before)]],
            [p["position"] for p in before],
        )
        self.assertEqual(state["measurements"]["ceiling_height"], measurement(2.8))
        self.assertEqual(state["views"][0]["user_corrections"], {"corner": [2, 3]})

    def test_plane_patches_have_no_fabricated_extents(self):
        rng = np.random.default_rng(5)
        points = [
            {"id": str(i), "position": [*rng.uniform(-2, 2, 2), 2], "semantic_surface": "floor"}
            for i in range(60)
        ]
        planes = plane_patches(points)
        self.assertEqual(len(planes), 1)
        self.assertIsNone(planes[0]["boundary_polygon"])
        self.assertEqual(planes[0]["hidden_extent"], "unknown")
        self.assertLess(planes[0]["rms_residual_arbitrary_units"], 1e-8)
        self.assertEqual(plane_patches(points[:5]), [])

    def test_empty_features_and_spatial_coverage(self):
        f = extract_features(Image.new("RGB", (100, 100)))
        self.assertEqual(f["pixels"].shape, (0, 2))
        self.assertEqual(match_features(f, f).shape, (0, 2))
        self.assertEqual(coverage([[0, 0], [1, 1], [2, 2]], (10, 10)), 0)

    def test_single_photo_pipeline_offline_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "a.png"
            Image.new("RGB", (640, 480)).save(image)
            doc = document()
            doc["views"][0]["image_sha256"] = digest(image)
            write_json(root / "photo.json", doc)
            manifest = root / "views.json"
            write_json(
                manifest,
                {
                    "schema_version": 1,
                    "room_id": "synthetic-room",
                    "images": [
                        {
                            "image_id": "a",
                            "file": "a.png",
                            "image_sha256": digest(image),
                            "photo_scene_file": "photo.json",
                            "rights": {
                                "allowed_for_benchmark": True,
                                "source": "synthetic",
                                "license": "CC0",
                            },
                        }
                    ],
                },
            )
            with patch(
                "room_multiview.pipeline.run_layout",
                side_effect=AssertionError("no models should load"),
            ):
                state = run_project(manifest, root / "one")
            self.assertEqual(state["shared_geometry"]["cameras"], {})
            self.assertEqual(state["shared_geometry"]["landmarks"], [])
            self.assertTrue((root / "one/sparse-room.ply").exists())
            with self.assertRaisesRegex(ValueError, "new multiview"):
                run_project(manifest, root / "one")
            Image.new("RGB", (640, 480), "red").save(image)
            # Remove manifest fingerprint to exercise the independent resume check.
            d = __import__("json").loads(manifest.read_text())
            del d["images"][0]["image_sha256"]
            write_json(manifest, d)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                run_project(manifest, root / "two", previous=root / "one/room.json")


if __name__ == "__main__":
    unittest.main()
