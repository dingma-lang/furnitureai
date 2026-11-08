"""Synthetic ARKit-format adapter checks; no real assets, weights or network."""

import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from furniture_bench.data import read_json, write_json

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from room_multiview.arkitscenes import (
        orient_calibration,
        prepare_scene,
        resample_intrinsics,
        trajectory_references,
    )


@unittest.skipUnless(HAS_SCIPY, "optional SciPy dependency absent")
class ArkitAdapterTests(unittest.TestCase):
    def test_quarter_turn_calibration_matches_rotated_camera_rays(self):
        k = np.array([[200.0, 0, 128], [0, 210, 96], [0, 0, 1]])
        pixel = np.array([50.0, 20.0, 1.0])
        ray = np.linalg.inv(k) @ pixel
        expected = {
            "Up": [50, 20],
            "Left": [191 - 20, 50],
            "Right": [20, 255 - 50],
            "Down": [255 - 50, 191 - 20],
        }
        for sky, target in expected.items():
            new_k, rotation, _, _ = orient_calibration(k, (256, 192), sky)
            projected = new_k @ rotation @ ray
            np.testing.assert_allclose(projected[:2] / projected[2], target, atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_same_intrinsics_preserve_depth_without_interpolation(self):
        values = np.arange(20, dtype=np.uint16).reshape(4, 5) * 100
        k = np.array([[100.0, 0, 2], [0, 100, 2], [0, 0, 1]])
        result = resample_intrinsics(values, k, k, (5, 4), True)
        np.testing.assert_array_equal(result, values)

    def test_reference_inverse_and_upright_rotation_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.traj"
            path.write_text("0.0 0 0 0 0 0 0\n0.1 0 0 0 -0.1 0 0\n")
            rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
            refs = trajectory_references(path, [("frame", 0.05)], rotation)
            np.testing.assert_allclose(refs[0]["position_metres"], [0.05, 0, 0])
            self.assertIn("not_external_motion_capture", refs[0]["reference_kind"])
            self.assertEqual(trajectory_references(path, [("outside", 2)], rotation), [])

    def test_approval_required_before_assets_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "approval.json", {"approved": False, "scene_id": "synthetic"})
            with self.assertRaisesRegex(ValueError, "approval"):
                prepare_scene(
                    root / "missing", root / "output", "synthetic", "Up", root / "approval.json"
                )
            self.assertFalse((root / "output").exists())

    def test_small_synthetic_zip_prepares_disjoint_roles_and_rgb_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()

            def png(array):
                memory = io.BytesIO()
                Image.fromarray(array).save(memory, format="PNG")
                return memory.getvalue()

            for asset in ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics"):
                with zipfile.ZipFile(assets / (asset + ".zip"), "w") as archive:
                    for i in range(3):
                        timestamp = i / 10
                        basename = f"synthetic_{timestamp:.3f}"
                        if asset.endswith("intrinsics"):
                            archive.writestr(f"{asset}/{basename}.pincam", "4 3 100 100 2 1")
                        elif asset == "lowres_wide":
                            archive.writestr(
                                f"{asset}/{basename}.png", png(np.full((3, 4, 3), 30 + i, np.uint8))
                            )
                        elif asset == "confidence":
                            confidence = np.full((3, 4), 2, np.uint8)
                            confidence[0, 0] = 0
                            archive.writestr(f"{asset}/{basename}.png", png(confidence))
                        else:
                            archive.writestr(
                                f"{asset}/{basename}.png", png(np.full((3, 4), 2000 + i, np.uint16))
                            )
            (assets / "lowres_wide.traj").write_text(
                "0.0 0 0 0 0 0 0\n0.1 0 0 0 -0.1 0 0\n0.2 0 0 0 -0.2 0 0\n"
            )
            write_json(
                root / "approval.json",
                {
                    "approved": True,
                    "scene_id": "synthetic",
                    "note": "synthetic test fixture, not real authorization",
                },
            )
            out = prepare_scene(
                assets, root / "prepared", "synthetic", "Left", root / "approval.json", 0, 2, 1, 2
            )
            tracking = read_json(out / "tracking.json")
            heldout = read_json(out / "heldout.json")
            rgb = read_json(out / "rgb-only.json")
            self.assertEqual(len(tracking["images"]), 2)
            self.assertEqual(len(heldout["images"]), 1)
            self.assertFalse(
                set(r["image_id"] for r in tracking["images"])
                & set(r["image_id"] for r in heldout["images"])
            )
            self.assertTrue(all("sensor_depth" not in r for r in rgb["images"]))
            first = tracking["images"][0]
            self.assertFalse(first["rights"]["commercial_training_authorized"])
            self.assertEqual(first["sensor_depth"]["units_per_metre"], 1000)
            with Image.open(out / first["sensor_depth"]["file"]) as depth:
                self.assertEqual(depth.size, (3, 4))
                self.assertEqual(np.count_nonzero(np.asarray(depth) == 0), 1)
            refs = read_json(out / "reference-poses.json")
            self.assertFalse(refs["used_for_inference"])


if __name__ == "__main__":
    unittest.main()
