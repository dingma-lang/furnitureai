"""Synthetic unit checks; no pretrained downloads or real-room accuracy assumptions."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from furniture_bench.data import digest
from room_multiview.gltf import export_gltf

HAS_GEOMETRY = all(importlib.util.find_spec(m) for m in ("cv2", "scipy"))
if HAS_GEOMETRY:
    from room_multiview.geometry import calibration, project
    from room_multiview.refinement import refine_project
    from room_multiview.rgbd import (
        backproject,
        dense_samples,
        register_rgbd,
        sample_depth,
        transform,
        voxel_fuse,
    )
    from room_multiview.sensor_pipeline import load_sensor
    from room_multiview.sensor_scale import scale_from_anchor_depth
    from room_multiview.structured import fit_observed_planes, intersections, supported_mesh


class GltfTests(unittest.TestCase):
    def test_metric_export_binary_alignment_bounds_and_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.gltf"
            points = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]])
            doc = export_gltf(
                path,
                points,
                np.full((3, 3), 120, np.uint8),
                [[0, 1, 2]],
                {"units": "m", "hidden": "unknown"},
            )
            self.assertEqual(doc["asset"]["version"], "2.0")
            self.assertEqual(doc["meshes"][0]["primitives"][0]["mode"], 4)
            binary = path.with_suffix(".bin").read_bytes()
            self.assertEqual(len(binary), doc["buffers"][0]["byteLength"])
            for v in doc["bufferViews"]:
                self.assertEqual(v["byteOffset"] % 4, 0)
                self.assertLessEqual(v["byteOffset"] + v["byteLength"], len(binary))
            decoded = np.frombuffer(binary, dtype="<f4", count=9).reshape(-1, 3)
            np.testing.assert_array_equal(decoded, points * [1, -1, -1])
            self.assertEqual(doc["extras"]["hidden"], "unknown")
            with self.assertRaises(ValueError):
                export_gltf(path, points, np.zeros((3, 3), np.uint8))

    def test_empty_export_does_not_invent_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = export_gltf(Path(tmp) / "empty.gltf", [], [])
            self.assertEqual(doc["nodes"], [])
            self.assertNotIn("meshes", doc)

    def test_export_rejects_nonfinite_or_bad_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            for points, triangles in (([[float("nan"), 0, 1]], []), ([[0, 0, 1]], [[0, 1, 2]])):
                with self.assertRaises(ValueError):
                    export_gltf(Path(tmp) / "bad.gltf", points, [[0, 0, 0]], triangles)


@unittest.skipUnless(HAS_GEOMETRY, "optional multiview dependencies absent")
class MeasuredDepthTests(unittest.TestCase):
    def test_sensor_scale_preserves_raw_map_and_rejects_disagreement(self):
        pixels = np.array([[x, y] for y in range(10, 25, 3) for x in range(10, 28, 3)])
        k = [[100, 0, 20], [0, 100, 20], [0, 0, 1]]
        xyz = backproject(pixels, np.full(len(pixels), 4.0), k)
        project = {
            "shared_geometry": {
                "anchor_view_id": "a",
                "metric_scale": None,
                "cameras": {"a": {"camera_from_world": np.eye(4).tolist()}},
                "landmarks": [
                    {
                        "id": str(i),
                        "position": p.tolist(),
                        "observations": [{"view_id": "a", "pixel": pixel.tolist()}],
                    }
                    for i, (p, pixel) in enumerate(zip(xyz, pixels))
                ],
            },
            "coordinate_frames": {"transforms": {}, "physical_metres": {}},
            "physical_accuracy_verified": False,
            "fit_readiness": {"ready": False},
        }
        original = copy.deepcopy(project)
        scaled, cal = scale_from_anchor_depth(
            project, np.full((40, 40), 2.0), k, "synthetic sensor", "a" * 64
        )
        self.assertAlmostEqual(cal["metres_per_unit"], 0.5)
        self.assertEqual(project, original)
        self.assertEqual(
            scaled["shared_geometry"]["landmarks"], original["shared_geometry"]["landmarks"]
        )
        self.assertFalse(scaled["fit_readiness"]["ready"])
        with self.assertRaisesRegex(ValueError, "preserve existing"):
            scale_from_anchor_depth(scaled, np.full((40, 40), 2.0), k, "synthetic", "a" * 64)
        for i, p in enumerate(project["shared_geometry"]["landmarks"]):
            p["position"][2] = 2 + i * 0.3
        with self.assertRaisesRegex(ValueError, "dispersion"):
            scale_from_anchor_depth(project, np.full((40, 40), 2.0), k, "synthetic", "a" * 64)

    def test_backprojection_uses_sensor_scale_not_relative_depth(self):
        xyz = backproject([[10, 20], [20, 20]], [2, 3], [[100, 0, 10], [0, 100, 20], [0, 0, 1]])
        np.testing.assert_allclose(xyz, [[0, 0, 2], [0.3, 0, 3]])

    def test_depth_edges_holes_and_unknowns_remain_missing(self):
        depth = np.full((20, 20), 2.0, np.float32)
        depth[:, 10:] = 4
        depth[3:6, 3:6] = 0
        _, valid = sample_depth(
            depth, [[2, 2], [4, 4], [9, 10], [18, 18]], [[100, 0, 10], [0, 100, 10], [0, 0, 1]]
        )
        self.assertEqual(valid.tolist(), [True, False, False, True])
        rgb = Image.new("RGB", (20, 20))
        samples = dense_samples(depth, rgb, [[100, 0, 10], [0, 100, 10], [0, 0, 1]], stride=2)
        self.assertFalse(any(tuple(p) == (4, 4) for p in samples["pixels"]))

    def test_metric_rigid_registration_with_outliers_and_bad_overlap(self):
        rng = np.random.default_rng(13)
        points = rng.uniform([-1, -1, 2], [1, 1, 4], (120, 3))
        pose = np.eye(4)
        pose[:3, 3] = [0.12, -0.04, 0.03]
        second = transform(points, pose)
        second[-20:] += 2
        pixels = rng.uniform([20, 20], [620, 460], (120, 2))
        report = register_rgbd(points, second, pixels, pixels, (640, 480))
        self.assertEqual(report["status"], "accepted")
        np.testing.assert_allclose(report["camera_b_from_a"], pose, atol=1e-7)
        bad = register_rgbd(points, second[::-1], pixels, pixels, (640, 480))
        self.assertEqual(bad["status"], "rejected")
        self.assertEqual(
            register_rgbd(points[:5], second[:5], pixels[:5], pixels[:5], (640, 480))["status"],
            "rejected",
        )

    def test_voxel_support_counts_distinct_views_and_semantic_conflict(self):
        points = np.array([[0.01, 0.01, 0.01], [0.011, 0.011, 0.011], [0.012, 0.012, 0.012]])
        cloud = voxel_fuse(
            points, np.full((3, 3), 100, np.uint8), np.array([1, 1, 2]), np.array([0, 0, 1])
        )
        self.assertEqual(cloud["view_count"].tolist(), [2])
        self.assertEqual(cloud["labels"].tolist(), [0])
        self.assertEqual(voxel_fuse(np.empty((0, 3)), [], [], [])["points"].shape, (0, 3))

    def test_relative_depth_cannot_enter_sensor_pipeline(self):
        with self.assertRaisesRegex(ValueError, "genuine"):
            load_sensor({"sensor_depth": {"status": "estimated_relative_depth"}}, "unused.json")

    def test_sensor_hash_scale_and_timestamp_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.png"
            Image.fromarray(np.full((5, 5), 10000, np.uint16)).save(path)
            spec = {
                "file": "depth.png",
                "sha256": digest(path),
                "status": "sensor_measured",
                "encoding": "uint16_depth_z",
                "source": "synthetic sensor",
                "units_per_metre": 5000,
                "rgb_depth_delta_seconds": 0.001,
                "calibration": {
                    "intrinsics": [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    "status": "measured",
                    "source": "synthetic",
                },
            }
            depth, _ = load_sensor({"sensor_depth": spec}, Path(tmp) / "manifest.json")
            np.testing.assert_allclose(depth, 2)
            for update in (
                {"sha256": "wrong"},
                {"units_per_metre": 0},
                {"rgb_depth_delta_seconds": 0.05},
            ):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_sensor({"sensor_depth": dict(spec, **update)}, Path(tmp) / "manifest.json")

    def test_structural_plane_requires_semantic_and_multiview_support(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
        points = np.c_[x.ravel(), y.ravel(), np.full(x.size, 2.0)]
        cloud = {
            "points": points,
            "labels": np.full(len(points), 2),
            "view_count": np.full(len(points), 2),
        }
        planes = fit_observed_planes(cloud)
        self.assertEqual(len(planes), 1)
        self.assertLess(planes[0]["rms_residual_metres"], 1e-8)
        self.assertIsNone(planes[0]["closed_boundary"])
        cloud["view_count"][:] = 1
        self.assertEqual(fit_observed_planes(cloud), [])
        cloud["view_count"][:] = 2
        cloud["labels"][:] = 0
        self.assertEqual(fit_observed_planes(cloud), [])

    def test_partial_mesh_does_not_bridge_holes_or_furniture(self):
        x, y = np.meshgrid(np.arange(0, 24, 4), np.arange(0, 24, 4))
        pixels = np.c_[x.ravel(), y.ravel()]
        points = np.c_[pixels * 0.01, np.full(len(pixels), 2.0)]
        samples = {"pixels": pixels, "points": points, "labels": np.full(len(points), 2)}
        plane = {"id": "p", "surface_hypothesis": "floor", "normal": [0, 0, 1], "offset_metres": -2}
        _, full, _ = supported_mesh(samples, np.eye(4), [plane])
        samples["labels"][14] = 0
        _, cut, _ = supported_mesh(samples, np.eye(4), [plane])
        self.assertLess(len(cut), len(full))
        self.assertFalse(np.any(cut == 14))
        inferred = intersections(
            [
                plane,
                {"id": "w", "surface_hypothesis": "wall", "normal": [1, 0, 0], "offset_metres": 0},
            ]
        )
        self.assertFalse(inferred[0]["usable_as_room_boundary"])


@unittest.skipUnless(HAS_GEOMETRY, "optional multiview dependencies absent")
class BundleRefinementTests(unittest.TestCase):
    def test_refinement_preserves_source_gauge_and_user_provenance(self):
        rng = np.random.default_rng(7)
        xyz = rng.uniform([-1, -1, 3], [1, 1, 5], (30, 3))
        poses = {"a": np.eye(4), "b": np.eye(4), "c": np.eye(4)}
        poses["b"][0, 3] = -1
        poses["c"][0, 3] = -0.5
        cal = calibration(
            {
                "intrinsics": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                "status": "measured",
                "source": "synthetic",
            }
        )
        points = []
        for i, p in enumerate(xyz):
            observations = [
                {"view_id": v, "pixel": project([p], pose, cal)[0].tolist()}
                for v, pose in poses.items()
            ]
            points.append(
                {
                    "id": str(i),
                    "position": (p + rng.normal(0, 0.01, 3)).tolist(),
                    "observations": observations,
                    "semantic_surface": "unknown",
                }
            )
        state = {
            "shared_geometry": {
                "cameras": {
                    v: {"camera_from_world": p.tolist(), "centre": (-p[:3, 3]).tolist()}
                    for v, p in poses.items()
                },
                "landmarks": points,
                "metric_scale": None,
            },
            "view_assets": {v: {"calibration": cal} for v in poses},
            "coordinate_frames": {"transforms": {"camera_to_world": {}}},
            "fit_readiness": {"ready": False},
            "measurements": {},
            "user_corrections": {"preserved": True},
        }
        original = copy.deepcopy(state)
        updated, report = refine_project(state, 20)
        self.assertEqual(state, original)
        self.assertEqual(report["status"], "accepted")
        self.assertLess(report["after_median_pixels"], report["before_median_pixels"])
        for v in ["a", "b"]:
            self.assertEqual(
                updated["shared_geometry"]["cameras"][v]["camera_from_world"],
                state["shared_geometry"]["cameras"][v]["camera_from_world"],
            )
        self.assertEqual(updated["user_corrections"], state["user_corrections"])
        state["measurements"] = {"wall": {"status": "measured"}}
        _, blocked = refine_project(state)
        self.assertEqual(
            blocked["reason"], "measured_constraints_require_explicit_constrained_refinement"
        )


if __name__ == "__main__":
    unittest.main()
