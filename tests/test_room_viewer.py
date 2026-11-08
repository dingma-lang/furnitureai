import copy
import unittest

from room_reconstruction.demo import demo_scene
from room_layout.scene import validate_photo_scene
from room_viewer.service import edit_scene, load_preset, presentation


class RoomViewerTests(unittest.TestCase):
    def test_replacement_preserves_geometry_and_original_and_reports_restricted_gap(self):
        original = demo_scene()
        saved = copy.deepcopy(original)
        result = edit_scene(original, {"type": "replace", "id": "sofa-original", "size": [2.8, 1.05, .9]})
        self.assertEqual(original, saved)
        for field in ("walls", "floor", "openings", "dimensions"):
            self.assertEqual(result["scene"][field], original[field])
        self.assertTrue(result["replacement_proposal"]["accepted"])
        self.assertEqual(result["status"], "restricted")
        self.assertFalse(result["physical_placement_verified"])

    def test_invalid_replacement_is_an_explicit_draft_not_accepted_geometry(self):
        result = edit_scene(demo_scene(), {"type": "replace", "id": "sofa-original", "size": [9, 1, 1]})
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["replacement_proposal"]["accepted"])
        self.assertEqual(next(f for f in result["scene"]["furniture"] if f["id"] == "sofa-original")["size"][0], 9)

    def test_move_rotate_remove_and_resize_use_existing_checks(self):
        original = demo_scene()
        moved = edit_scene(original, {"type": "update", "id": "sofa-original", "position": [3, 2.45, 0], "rotation_degrees": 30})
        self.assertEqual(moved["status"], "invalid")
        self.assertTrue(moved["assessment"]["geometry_check"]["violations"])
        removed = edit_scene(original, {"type": "remove", "id": "chair"})
        self.assertEqual(len(removed["scene"]["furniture"]), 4)
        resized = edit_scene(original, {"type": "resize", "dimensions": [7, 5, 3]})
        self.assertEqual(resized["scene"]["floor"]["boundary"], [[0, 0], [7, 0], [7, 5], [0, 5]])
        self.assertEqual(resized["scene"]["furniture"], original["furniture"])
        self.assertEqual(resized["scene"]["openings"][0]["width"], .9)

    def test_bad_dimensions_and_nonfinite_inputs_leave_input_untouched(self):
        scene = demo_scene()
        for action in ({"type": "resize", "dimensions": [-1, 2, 3]}, {"type": "update", "id": "chair", "size": [float("nan"), 1, 1]}):
            with self.assertRaises(ValueError):
                edit_scene(scene, action)
        self.assertEqual(scene, demo_scene())

    def test_unverified_v1_document_is_viewable_but_has_no_metric_checks(self):
        scene = demo_scene()
        scene["provenance"] = {"kind": "photographic_estimate"}
        result = presentation(scene)
        self.assertFalse(result["editable"])
        self.assertEqual(result["status"], "disabled")
        self.assertIsNone(result["assessment"])
        with self.assertRaisesRegex(ValueError, "disabled"):
            edit_scene(scene, {"type": "remove", "id": "chair"})

    def test_partial_presets_never_enable_placement_even_with_flag(self):
        result = load_preset("single-photo")
        validate_photo_scene(result["scene"])
        forged = copy.deepcopy(result["scene"])
        forged["physical_accuracy_verified"] = True
        self.assertEqual(presentation(forged)["status"], "disabled")
        with self.assertRaisesRegex(ValueError, "disabled"):
            edit_scene(result["scene"], {"type": "resize", "dimensions": [6, 5, 3]})

    def test_import_does_not_read_untrusted_external_asset_paths(self):
        scene = load_preset("single-photo")["scene"]
        scene["sensor_reconstruction"] = {"point_cloud": {"file": "/etc/passwd"}}
        self.assertEqual(presentation(scene)["observed"], {})


if __name__ == "__main__":
    unittest.main()
