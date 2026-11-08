"""Deterministic synthetic checks, with no model inference or dataset decisions."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from photo_simulator.compositor import composite, extract_asset, make_mask


def fixture():
    room = Image.new("RGB", (240, 180), (210, 205, 198))
    draw = ImageDraw.Draw(room)
    draw.rectangle((65, 55, 165, 135), fill=(90, 65, 40))
    asset = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(asset)
    draw.rectangle((5, 8, 74, 50), fill=(30, 90, 180, 255))
    draw.rectangle((8, 48, 14, 59), fill=(35, 35, 35, 255))
    draw.rectangle((65, 48, 71, 59), fill=(35, 35, 35, 255))
    mask = Image.new("L", room.size, 0)
    ImageDraw.Draw(mask).rectangle((65, 55, 165, 135), fill=255)
    placement = dict(x=115, y=95, width=110, height=90, rotation=0, skew=0)
    return room, mask, asset, placement


class CompositorTests(unittest.TestCase):
    def test_semantic_mask_cannot_remove_objects_outside_selection(self):
        room, _, _, _ = fixture()
        initial = Image.new("L", room.size, 255)
        mask, _ = make_mask(room, [60, 50, 170, 140], initial_mask=initial)
        a = np.asarray(mask)
        self.assertEqual(a[:50].sum(), 0)
        self.assertEqual(a[:, :60].sum(), 0)
        self.assertEqual(a[140:].sum(), 0)
        self.assertGreater(a[50:140,60:170].sum(), 0)

    def test_mask_is_bounded_and_manual_edits_override(self):
        room, _, _, _ = fixture()
        mask, info = make_mask(room, [60, 50, 170, 140], [{"mode": "subtract", "points": [[80, 70], [100, 70], [100, 90], [80, 90]]}])
        a = np.asarray(mask)
        self.assertEqual(a[80, 90], 0)
        self.assertEqual(a[:50].sum(), 0)
        self.assertEqual(a[:, :60].sum(), 0)
        self.assertGreater(a.sum(), 0)
        self.assertEqual(info["manual_corrections"], 1)

    def test_semantic_seed_uses_corrected_full_selection_and_edits_win(self):
        room, _, _, _ = fixture()
        seed = Image.new("L", room.size, 0)
        ImageDraw.Draw(seed).rectangle((60, 50, 180, 145), fill=255)
        mask, info = make_mask(room, [55, 45, 185, 150], [{"mode":"subtract", "points":[[80,70],[100,70],[100,90],[80,90]]}], initial_mask=seed)
        self.assertEqual(info["method"], "provided_semantic_mask")
        self.assertEqual(np.asarray(mask)[80,90], 0)
        self.assertGreater(np.asarray(mask)[130,170], 0)

    def test_empty_grabcut_uses_reported_fallback(self):
        room, _, _, _ = fixture()
        with patch("photo_simulator.compositor.cv2.grabCut", side_effect=cv2.error("test mask failure")):
            mask, info = make_mask(room, [60, 50, 170, 140])
        self.assertEqual(info["method"], "rectangle_fallback")
        self.assertTrue(info["warnings"])
        self.assertEqual(np.count_nonzero(mask), 110 * 90)

    def test_supplied_alpha_is_preserved_exactly(self):
        _, _, asset, _ = fixture()
        a = np.asarray(asset).copy()
        a[15, 15, 3] = 125
        extracted, info = extract_asset(Image.fromarray(a))
        bounds = info["source_bbox"]
        self.assertEqual(info["method"], "existing_alpha_preserved")
        np.testing.assert_array_equal(np.asarray(extracted), a[bounds[1]:bounds[3], bounds[0]:bounds[2]])

    def test_plain_background_is_removed(self):
        image = Image.new("RGB", (120, 100), "white")
        ImageDraw.Draw(image).rectangle((30, 20, 90, 80), fill=(45, 70, 120))
        extracted, info = extract_asset(image)
        self.assertNotEqual(info["method"], "opaque_background_retained")
        self.assertLess(extracted.width, image.width)
        self.assertGreater(np.asarray(extracted)[:, :, 3].sum(), 0)

    def test_unseparable_asset_keeps_original_with_warning(self):
        image = Image.new("RGB", (100, 100), "white")
        result, info = extract_asset(image)
        self.assertEqual(result.size, image.size)
        self.assertEqual(info["method"], "opaque_background_retained")
        self.assertTrue(info["warnings"])

    def test_composite_changes_pixels_and_preserves_unrelated_region(self):
        room, mask, asset, placement = fixture()
        result, info = composite(room, mask, asset, placement)
        self.assertGreater(info["changed_pixels"], 100)
        np.testing.assert_array_equal(np.asarray(result)[:30], np.asarray(room)[:30])
        self.assertFalse(info["geometry"]["physical_dimensions_verified"])
        self.assertEqual(info["inpaint_method"], "opencv_telea")

    def test_manual_foreground_is_preserved(self):
        room, mask, asset, placement = fixture()
        protection = Image.new("L", room.size, 0)
        ImageDraw.Draw(protection).rectangle((100, 80, 125, 150), fill=255)
        result, _ = composite(room, mask, asset, placement, {"shadow": True, "color_match": True}, protected_mask=protection)
        selected = np.asarray(protection) > 0
        np.testing.assert_array_equal(np.asarray(result)[selected], np.asarray(room)[selected])

    def test_missing_or_invalid_depth_is_explicit_fallback(self):
        room, mask, asset, placement = fixture()
        _, info = composite(room, mask, asset, placement, {"depth_assist": True})
        self.assertEqual(info["depth_status"], "unavailable")
        for value in [np.ones((180, 240)), np.full((180, 240), np.nan), np.ones((5, 5))]:
            _, info = composite(room, mask, asset, placement, {"depth_assist": True}, depth=value)
            self.assertEqual(info["depth_status"], "invalid_skipped")

    def test_depth_nearer_foreground_is_protected(self):
        room, mask, asset, placement = fixture()
        depth = np.tile(np.linspace(.1, .8, room.height)[:, None], (1, room.width))
        depth[70:120, 170:200] = 1.5
        moved = {**placement, "x": 160, "width": 160}
        result, info = composite(room, mask, asset, moved, {"depth_assist": True}, depth=depth)
        self.assertEqual(info["depth_status"], "relative_depth_occlusion_heuristic")
        self.assertGreater(info["depth_protected_pixels"], 0)
        np.testing.assert_array_equal(np.asarray(result)[70:120, 170:200], np.asarray(room)[70:120, 170:200])

    def test_pose_controls_change_actual_output(self):
        room, mask, asset, placement = fixture()
        baseline, _ = composite(room, mask, asset, placement)
        moved, _ = composite(room, mask, asset, {**placement, "x": 160, "rotation": 17, "skew": .3, "flip": True})
        self.assertFalse(np.array_equal(baseline, moved))

    def test_degenerate_and_huge_inputs_reject(self):
        room, mask, asset, placement = fixture()
        for change in [{"width": float("nan")}, {"height": float("inf")}, {"skew": 4}, {"width": 1e20}, {"x": 1e50}]:
            with self.assertRaises(ValueError):
                composite(room, mask, asset, {**placement, **change})
        with self.assertRaises(ValueError):
            make_mask(room, [0, 0, room.width, room.height])
        with self.assertRaises(ValueError):
            make_mask(room, [10, 10, 0, 0])
        with self.assertRaises(ValueError):
            composite(room, Image.new("L", room.size, 255), asset, placement)
        with self.assertRaises(ValueError):
            composite(room, Image.new("L", room.size, 0), asset, placement)
        with self.assertRaises(ValueError):
            composite(room, mask, asset, placement, {"inpaint_radius": float("nan")})

    def test_no_mutation_and_json_safe_diagnostics(self):
        import json
        room, mask, asset, placement = fixture()
        old = [np.asarray(x).copy() for x in (room, mask, asset)]
        _, info = composite(room, mask, asset, placement, {"shadow": True, "color_match": True})
        json.dumps(info, allow_nan=False)
        for actual, expected in zip((room, mask, asset), old):
            np.testing.assert_array_equal(np.asarray(actual), expected)


if __name__ == "__main__":
    unittest.main()
