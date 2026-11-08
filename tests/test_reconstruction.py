from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from room_reconstruction.cli import main
from room_reconstruction.collision import bodies_overlap, check_scene, footprint, polygons_overlap
from room_reconstruction.demo import demo_scene, replacement_sofa, run_demo
from room_reconstruction.model import (
    propose_replacement,
    remove_furniture,
    save_json,
    validate_scene,
)


def body(position, size, rotation=0, object_id="test"):
    return {
        "id": object_id,
        "category": "chair",
        "position": position,
        "size": size,
        "rotation_degrees": rotation,
    }


class PlacementTests(unittest.TestCase):
    def test_demonstration_has_valid_known_geometry(self):
        scene = demo_scene()
        self.assertEqual(scene["dimensions"], [6, 4.5, 2.8])
        self.assertFalse(scene["provenance"]["derived_from_photo"])
        self.assertTrue(check_scene(scene)["fits"])
        self.assertEqual({o["kind"] for o in scene["openings"]}, {"door", "window"})

    def test_removal_and_replacement_do_not_mutate_original(self):
        scene = demo_scene()
        snapshot = copy.deepcopy(scene)
        removed = remove_furniture(scene, "sofa-original")
        self.assertEqual(len(removed["furniture"]), 4)
        replacement = replacement_sofa()
        updated, report = propose_replacement(scene, "sofa-original", replacement)
        self.assertTrue(report["accepted"])
        self.assertEqual(scene, snapshot)
        ids = {item["id"] for item in updated["furniture"]}
        self.assertNotIn("sofa-original", ids)
        self.assertIn("sofa-replacement", ids)
        for original in scene["furniture"][1:]:
            self.assertIn(original, updated["furniture"])
        self.assertEqual(report["original_size"], [2.1, 0.85, 0.85])
        self.assertEqual(report["replacement_size"], [2.8, 1.05, 0.9])

    def test_oversized_replacement_is_rejected_and_original_preserved(self):
        scene = demo_scene()
        candidate = replacement_sofa()
        candidate["size"][0] = 4.9
        updated, report = propose_replacement(scene, "sofa-original", candidate)
        self.assertFalse(report["accepted"])
        self.assertEqual(scene, updated)
        self.assertIn(
            "furniture_overlap", {v["type"] for v in report["placement_check"]["violations"]}
        )

    def test_wall_overlap_and_floor_boundary_are_detected(self):
        scene = demo_scene()
        candidate = replacement_sofa()
        candidate["position"][0] = 5.4
        _, report = propose_replacement(scene, "sofa-original", candidate)
        types = {v["type"] for v in report["placement_check"]["violations"]}
        self.assertIn("outside_floor", types)
        self.assertIn("wall_overlap", types)

    def test_rotation_is_used_for_fit(self):
        scene = demo_scene()
        scene["furniture"] = [body([5.6, 2, 0], [1.4, 0.4, 1], 90)]
        self.assertTrue(check_scene(scene)["fits"])
        scene["furniture"][0]["rotation_degrees"] = 0
        self.assertFalse(check_scene(scene)["fits"])

    def test_sat_does_not_use_axis_aligned_bounds(self):
        first = body([0, 0, 0], [2, 0.2, 1], 45)
        second = body([0, 0.5, 0], [2, 0.2, 1], 45)
        # Their axis-aligned envelopes overlap, but their actual parallel strips do not.
        self.assertFalse(polygons_overlap(footprint(first), footprint(second)))
        second["position"][1] = 0.1
        self.assertTrue(bodies_overlap(first, second))

    def test_vertical_separation_and_touching_are_allowed(self):
        first = body([1, 1, 0], [1, 1, 1])
        above = body([1, 1, 1], [1, 1, 1])
        beside = body([2, 1, 0], [1, 1, 1])
        self.assertFalse(bodies_overlap(first, above))
        self.assertFalse(bodies_overlap(first, beside))
        beside["position"][0] = 1.99
        self.assertTrue(bodies_overlap(first, beside))

    def test_floor_ceiling_and_door_clearance(self):
        scene = demo_scene()
        scene["furniture"] = [body([0.95, 0.45, 0], [0.5, 0.5, 0.5])]
        self.assertIn("opening_clearance", {v["type"] for v in check_scene(scene)["violations"]})
        scene["furniture"] = [body([3, 2, -0.1], [0.5, 0.5, 3.1])]
        types = {v["type"] for v in check_scene(scene)["violations"]}
        self.assertIn("below_floor", types)
        self.assertIn("above_ceiling", types)

    def test_wall_openings_are_subtracted_in_height_and_distance(self):
        scene = demo_scene()
        scene["walls"].append(
            {
                "id": "partition",
                "start": [3, 0],
                "end": [3, 4.5],
                "height": 2.8,
                "thickness": 0.2,
                "interior_side": "left",
            }
        )
        scene["openings"].extend(
            [
                {
                    "id": "partition-door",
                    "kind": "door",
                    "wall_id": "partition",
                    "offset": 1.5,
                    "width": 1,
                    "bottom": 0,
                    "height": 2.1,
                },
                {
                    "id": "partition-window",
                    "kind": "window",
                    "wall_id": "partition",
                    "offset": 3,
                    "width": 1,
                    "bottom": 1,
                    "height": 1,
                },
            ]
        )
        scene["furniture"] = [body([3, 2, 0], [0.4, 0.5, 1])]
        self.assertTrue(check_scene(scene)["fits"])
        scene["furniture"][0]["size"][2] = 2.3
        self.assertFalse(check_scene(scene)["fits"])
        scene["furniture"] = [body([3, 3.5, 1.1], [0.4, 0.5, 0.8])]
        self.assertTrue(check_scene(scene)["fits"])
        scene["furniture"][0]["position"][2] = 0
        self.assertFalse(check_scene(scene)["fits"])


class SceneValidationTests(unittest.TestCase):
    def test_invalid_units_dimensions_and_rotation(self):
        for change in (
            lambda s: s.update(units="cm"),
            lambda s: s.update(dimensions=[0, 4.5, 2.8]),
            lambda s: s["furniture"][0].update(rotation_degrees=float("nan")),
            lambda s: s["furniture"][0].update(size=[-1, 1, 1]),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scene = demo_scene()
                change(scene)
                validate_scene(scene)

    def test_unknown_wall_duplicate_ids_and_out_of_bounds_openings(self):
        for change in (
            lambda s: s["openings"][0].update(wall_id="unknown"),
            lambda s: s["furniture"][0].update(id="floor"),
            lambda s: s["openings"][0].update(width=100),
            lambda s: s["floor"].update(z=1),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scene = demo_scene()
                change(scene)
                validate_scene(scene)

    def test_overlapping_openings_are_rejected(self):
        scene = demo_scene()
        duplicate = copy.deepcopy(scene["openings"][0])
        duplicate["id"] = "overlapping-door"
        scene["openings"].append(duplicate)
        with self.assertRaises(ValueError):
            validate_scene(scene)


class ExportAndCliTests(unittest.TestCase):
    def test_exported_demo_and_geometry_are_editable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            report = run_demo(output)
            self.assertTrue(report["accepted_replacement"]["accepted"])
            self.assertFalse(report["rejected_furniture_collision"]["accepted"])
            self.assertFalse(report["rejected_wall_collision"]["accepted"])
            gaps = report["accepted_clearances_metres"]
            self.assertAlmostEqual(gaps["sofa_to_north_interior_face"], 0.255)
            self.assertAlmostEqual(gaps["sofa_to_coffee_table"], 0.445)
            self.assertAlmostEqual(gaps["sofa_to_floor_lamp"], 0.075)
            for name in ("before", "sofa-removed", "after"):
                room = json.loads((output / f"{name}.json").read_text())
                self.assertTrue(check_scene(room)["fits"])
                self.assertEqual(
                    ElementTree.parse(output / f"{name}.svg").getroot().tag,
                    "{http://www.w3.org/2000/svg}svg",
                )
                obj = (output / f"{name}.obj").read_text().splitlines()
                vertex_count = sum(line.startswith("v ") for line in obj)
                for line in obj:
                    if line.startswith("f "):
                        self.assertTrue(all(1 <= int(v) <= vertex_count for v in line.split()[1:]))
                objects = [line[2:] for line in obj if line.startswith("o ")]
                self.assertIn("floor", objects)
                self.assertEqual(
                    sum(v.startswith("furniture_") for v in objects), len(room["furniture"])
                )
            after_mesh = (output / "after.obj").read_text()
            self.assertIn("o furniture_sofa-replacement", after_mesh)
            self.assertNotIn("o furniture_sofa-original", after_mesh)

    def test_cli_accepts_valid_replacement_and_rejects_invalid_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "room.json", demo_scene())
            save_json(root / "sofa.json", replacement_sofa())
            args = [
                "replace",
                "--room",
                str(root / "room.json"),
                "--remove-id",
                "sofa-original",
                "--replacement",
                str(root / "sofa.json"),
                "--output",
                str(root / "accepted.json"),
            ]
            self.assertEqual(main(args), 0)
            self.assertTrue((root / "accepted.json").exists())
            invalid = replacement_sofa()
            invalid["position"][0] = 5.4
            save_json(root / "sofa.json", invalid)
            args[-1] = str(root / "rejected.json")
            self.assertEqual(main(args), 2)
            self.assertFalse((root / "rejected.json").exists())

    def test_reconstruction_imports_no_ml_libraries(self):
        code = (
            "import sys; from room_reconstruction.demo import demo_scene; "
            "from room_reconstruction.collision import check_scene; "
            "assert check_scene(demo_scene())['fits']; "
            "assert not any(m in sys.modules for m in ['torch','transformers','faiss','furniture_bench'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
