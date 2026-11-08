"""Numerical/validation regressions; no model, Open3D or dataset downloads."""

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from furniture_bench.data import write_json
from room_multiview.numeric import apply_pose, points_array, rigid_pose
from room_multiview.odometry_quality import alignment_quality, validate_candidate

HAS_GEOMETRY = all(importlib.util.find_spec(m) for m in ("cv2", "scipy"))
if HAS_GEOMETRY:
    from room_layout.geometry import (
        corner_candidates,
        fit_segments,
        infer_occlusion_bridges,
        line_intersection,
    )
    from room_multiview.geometry import (
        Policy,
        calibration,
        estimate_pair,
        register_camera,
        triangulate,
    )
    from room_multiview.recovery import fuse_recovered_cameras, run_odometry_worker
    from room_multiview.recovery_evaluation import summarize_cameras
    from room_multiview.rgbd import (
        backproject,
        register_rgbd,
        rigid_fit,
        sample_depth,
        voxel_fuse,
    )


class NumericValidationTests(unittest.TestCase):
    def test_finite_bounded_coordinates_and_empty_shape(self):
        self.assertEqual(points_array([]).shape, (0, 3))
        for value in ([[0, np.nan, 1]], [[0, np.inf, 1]], [[1e300, 0, 1]], [1, 2, 3]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                points_array(value)

    def test_nonrigid_reflected_and_singular_pose_rejected(self):
        for matrix in (np.zeros((4, 4)), np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1])):
            with self.subTest(matrix=matrix.tolist()), self.assertRaises(ValueError):
                rigid_pose(matrix)
        pose = np.eye(4)
        pose[0, 3] = float("nan")
        with self.assertRaises(ValueError):
            rigid_pose(pose)

    def test_explicit_contraction_matches_elementwise_without_warnings(self):
        rng = np.random.default_rng(41)
        points = rng.normal(size=(5000, 3))
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="raise"):
                actual = apply_pose(points, pose)
        np.testing.assert_allclose(actual, points + [0.1, 0.2, 0.3])


class DenseOdometryValidationTests(unittest.TestCase):
    def fixture(self):
        y, x = np.mgrid[:120, :160]
        depth = np.full((120, 160), 2.0, float)
        gray = (x + y) / 280
        k = np.array([[130.0, 0, 80], [0, 130, 60], [0, 0, 1]])
        return depth, gray, k

    def test_identity_has_valid_bidirectional_support(self):
        depth, gray, k = self.fixture()
        before = depth.copy()
        q = alignment_quality(depth, depth, gray, gray, k, np.eye(4))
        self.assertTrue(q["accepted"])
        self.assertEqual(q["overlap"], 1)
        self.assertEqual(q["median_depth_error_metres"], 0)
        self.assertTrue(validate_candidate(np.eye(4), np.eye(4), q, q)["accepted"])
        np.testing.assert_array_equal(depth, before)

    def test_low_overlap_and_inconsistent_depth_rejected(self):
        depth, gray, k = self.fixture()
        target = np.zeros_like(depth)
        target[:20] = 2
        q = alignment_quality(depth, target, gray, gray, k, np.eye(4))
        self.assertFalse(q["accepted"])
        q = alignment_quality(depth, depth + 0.2, gray, gray, k, np.eye(4))
        self.assertFalse(q["accepted"])
        self.assertIsNone(q["median_depth_error_metres"])

    def test_low_depth_error_does_not_override_bad_cycle(self):
        depth, gray, k = self.fixture()
        q = alignment_quality(depth, depth, gray, gray, k, np.eye(4))
        wrong = np.eye(4)
        wrong[0, 3] = 0.05
        check = validate_candidate(np.eye(4), wrong, q, q)
        self.assertFalse(check["accepted"])
        self.assertIn("cycle", check["reason"])

    def test_nonfinite_depth_and_invalid_transform_fail_explicitly(self):
        depth, gray, k = self.fixture()
        depth[0, 0] = np.inf
        with self.assertRaises(ValueError):
            alignment_quality(depth, depth, gray, gray, k, np.eye(4))
        check = validate_candidate(
            np.zeros((4, 4)), np.eye(4), {"accepted": True}, {"accepted": True}
        )
        self.assertFalse(check["accepted"])


@unittest.skipUnless(HAS_GEOMETRY, "optional geometry packages absent")
class GeometryRegressionTests(unittest.TestCase):
    def test_worker_launcher_preserves_virtual_environment_symlink_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "system-python"
            binary.write_text("fixture")
            virtual = root / "venv/bin/python"
            virtual.parent.mkdir(parents=True)
            virtual.symlink_to(binary)

            def completed(command, **kwargs):
                self.assertEqual(command[0], str(virtual.absolute()))
                self.assertNotEqual(command[0], str(binary.resolve()))
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                write_json(output / "report.json", {"fixture": True})
                return SimpleNamespace(returncode=0)

            with patch("room_multiview.recovery.subprocess.run", side_effect=completed):
                report = run_odometry_worker(root / "manifest.json", virtual, root / "experiment")
            self.assertTrue(report["fixture"])

    def test_recovery_rejects_heldout_depth_alias_before_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker"
            worker.mkdir()
            write_json(
                worker / "report.json",
                {
                    "reference_poses_used": False,
                    "sources": {"tracking": {"image_sha256": "a" * 64, "depth_sha256": "d" * 64}},
                },
            )
            write_json(worker / "hybrid.json", {"cameras": {}, "anchor_view_id": "tracking"})
            write_json(
                root / "heldout.json",
                {
                    "images": [
                        {
                            "image_id": "heldout",
                            "image_sha256": "b" * 64,
                            "sensor_depth": {"sha256": "d" * 64},
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(ValueError, "held-out"):
                fuse_recovered_cameras(
                    root / "nonexistent-baseline.json",
                    worker,
                    "hybrid",
                    root / "heldout.json",
                    root / "output",
                )
            self.assertFalse((root / "output").exists())

    def test_camera_evaluation_does_not_fit_away_metric_error(self):
        refs = [
            {
                "image_id": "a",
                "timestamp_delta_seconds": 0,
                "position_metres": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            {
                "image_id": "b",
                "timestamp_delta_seconds": 0,
                "position_metres": [1, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
        ]
        b = np.eye(4)
        b[0, 3] = -2
        cameras = {
            "a": {"camera_from_sensor_world": np.eye(4).tolist()},
            "b": {"camera_from_sensor_world": b.tolist()},
        }
        report = summarize_cameras(cameras, "a", refs, {"a", "b"}, {"a", "b"})
        self.assertEqual(report["common_with_phase5_baseline"]["translation_rmse_metres"], 1)
        self.assertIsNone(report["evaluation_scale_fit"])

    def cal(self):
        return calibration(
            {
                "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "source": "synthetic",
                "status": "measured",
            }
        )

    def test_singular_huge_or_nonfinite_calibration_refused(self):
        for k in (
            np.zeros((3, 3)),
            np.diag([1e300, 1, 1]),
            [[500, 0, 0], [0, np.nan, 0], [0, 0, 1]],
            np.diag([1e-14, 1, 1]),
        ):
            with self.subTest(k=np.asarray(k).tolist()), self.assertRaises(ValueError):
                calibration({"intrinsics": k, "source": "synthetic", "status": "measured"})

    def test_identical_and_collinear_matches_do_not_create_pose(self):
        line = np.c_[np.arange(50), np.arange(50)]
        result = estimate_pair(line, line, self.cal(), self.cal(), (640, 480), (640, 480))
        self.assertEqual(result["status"], "rejected")
        with self.assertRaises(ValueError):
            estimate_pair([[0, np.nan]], [[0, 0]], self.cal(), self.cal(), (640, 480), (640, 480))

    def test_zero_baseline_and_near_parallel_rays_remain_invalid(self):
        rng = np.random.default_rng(3)
        p = rng.uniform([50, 50], [600, 400], (50, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="raise"):
                xyz, valid, angles, errors = triangulate(
                    p, p, np.eye(4), np.eye(4), self.cal(), self.cal(), Policy()
                )
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(xyz).all())
        self.assertTrue(np.isfinite(angles).all())
        self.assertTrue(np.isfinite(errors).all())

    def test_bad_depth_voxel_and_rigid_fit_inputs_refused(self):
        for points in (np.zeros((3, 3)), [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[np.inf, 0, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                rigid_fit(points, points)
        with self.assertRaises(ValueError):
            backproject([[0, 0]], [np.nan], np.eye(3))
        with self.assertRaises(ValueError):
            sample_depth(np.full((5, 5), np.nan), [[1, 1]], np.eye(3))
        with self.assertRaises(ValueError):
            voxel_fuse([[np.nan, 0, 0]], [[0, 0, 0]], [0], [0])
        with self.assertRaises(ValueError):
            voxel_fuse([[0, 0, 0]], [[0, 0, 0]], [0], [0], voxel_size=0)

    def test_degenerate_rgbd_and_pnp_reject_without_native_error(self):
        points = np.zeros((40, 3))
        pixels = np.zeros((40, 2))
        self.assertEqual(
            register_rgbd(points, points, pixels, pixels, (640, 480))["status"], "rejected"
        )
        self.assertEqual(
            register_camera(points, pixels, self.cal(), (640, 480))["reason"],
            "degenerate_3d_correspondences",
        )

    def test_line_warning_regression_and_invalid_inputs(self):
        mask = np.zeros((100, 200), bool)
        mask[50, :] = True
        confidence = np.ones_like(mask, float)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(all="raise"):
                lines = fit_segments(mask, confidence)
        self.assertTrue(lines)
        self.assertTrue(np.isfinite(lines[0]["endpoints"]).all())
        self.assertIsNone(line_intersection([[0, 0], [1, 0]], [[0, 1], [1, 1]]))
        with self.assertRaises(ValueError):
            line_intersection([[0, 0], [np.inf, 1]], [[0, 1], [1, 1]])
        confidence[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_segments(mask, confidence)
        lines = [
            {
                "id": "a",
                "kind": "wall_floor",
                "visibility": "visible_evidence",
                "endpoints": [[1, 1], [1, 1]],
                "confidence": 1.0,
            },
            {
                "id": "b",
                "kind": "wall_floor",
                "visibility": "visible_evidence",
                "endpoints": [[0, 0], [10, 10]],
                "confidence": 1.0,
            },
        ]
        with np.errstate(all="raise"):
            self.assertEqual(corner_candidates(lines, 100, 100), [])
            self.assertEqual(infer_occlusion_bridges(lines, np.ones((100, 100), bool)), [])


if __name__ == "__main__":
    unittest.main()
