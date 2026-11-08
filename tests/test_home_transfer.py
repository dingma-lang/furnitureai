import copy
import io
import tempfile
import unittest
import zipfile

from PIL import Image

from room_reconstruction.polygon import contains, polygon
from room_viewer.home_transfer import TransferStore, room_scene, scale_check, validate_home
from room_viewer.inventory import InventoryStore
from room_viewer.photo_projects import ProjectStore
from room_viewer.service import edit_scene, presentation


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projects = ProjectStore(self.tmp.name)
        self.inventory = InventoryStore(self.projects)
        self.store = TransferStore(self.projects, self.inventory)
        self.project = self.store.create({"name": "Destination fixture"})
        self.pid = self.project["id"]
        self.item = self.inventory.action(
            {"operation": "create", "display_name": "Fixture sofa", "category": "sofa"}
        )["item"]
        self.iid = self.item["inventory_item_id"]

    def action(self, op, **kwargs):
        self.project = self.store.action(self.pid, {"operation": op, **kwargs})
        return self.project["home_transfer"]

    def image(self, fmt="PNG"):
        b = io.BytesIO()
        Image.new("RGB", (1000, 600), "white").save(b, format=fmt)
        return b.getvalue()

    def plan(self):
        self.project = self.store.upload(self.pid, self.image(), "fixture.png", True)
        return self.action(
            "calibrate",
            p1=[100, 100],
            p2=[700, 100],
            value=6,
            unit="m",
            source="conceptual_drawing",
            description="Generated fixture, not actual apartment",
        )

    def room(self, boundary=None):
        h = self.action(
            "room",
            name="Living",
            boundary=boundary or [[1, 1], [7, 1], [7, 5], [1, 5]],
            coordinate_space="metric",
            height=None,
            evidence_status="estimated",
        )
        return list(h["rooms"])[-1]

    def place(self, rid, **extra):
        return self.action(
            "place",
            inventory_item_id=self.iid,
            room_id=rid,
            position=[3, 3, 0],
            approximate_acknowledged=True,
            **extra,
        )

    def test_destination_identity_does_not_modify_library(self):
        before = self.inventory.path.read_bytes()
        self.plan()
        self.place(self.room())
        self.assertEqual(before, self.inventory.path.read_bytes())

    def test_images_preserve_aspect_and_originals(self):
        for fmt in ("PNG", "JPEG"):
            raw = self.image(fmt)
            p = self.store.upload(self.pid, raw, "sample", True)["home_transfer"]["floorplan"]
            self.assertEqual([p["width"], p["height"]], [1000, 600])
            self.assertEqual(self.store.asset(self.pid, p["original"]).read_bytes(), raw)

    def test_private_paths_and_digest(self):
        self.plan()
        for token in ("../../owned-inventory.json", "no.png", "0" * 32 + ".png"):
            with self.assertRaises(ValueError):
                self.store.asset(self.pid, token)
        p = self.project["home_transfer"]["floorplan"]
        path = self.store.asset(self.pid, p["preview"])
        path.write_bytes(b"tamper")
        with self.assertRaises(ValueError):
            self.store.asset(self.pid, p["preview"])

    def test_invalid_uploads(self):
        for raw, rights, page in [
            (self.image(), False, 0),
            (b"bad", True, 0),
            (self.image(), True, 1),
            (self.image(), True, -1),
            (b"%PDF-bad", True, 0),
        ]:
            with self.assertRaises(ValueError):
                self.store.upload(self.pid, raw, "bad", rights, page)

    def test_scale_units(self):
        self.plan()
        for value, unit in [
            (600, "cm"),
            (6, "m"),
            (236.2204724409449, "in"),
            (19.68503937007874, "ft"),
        ]:
            h = self.action(
                "calibrate",
                p1=[100, 100],
                p2=[700, 100],
                value=value,
                unit=unit,
                source="user_calibrated_plan",
                description="Fixture",
            )
            self.assertAlmostEqual(scale_check(h["floorplan"])["metres_per_pixel"], 0.01)

    def test_invalid_calibration_atomic(self):
        self.plan()
        before = self.projects.read(self.pid)
        for p1, p2, v in [
            ([0, 0], [0, 0], 2),
            ([-1, 0], [100, 0], 1),
            ([0, 0], [100, 0], float("nan")),
            ([0, 0], [100, 0], -1),
        ]:
            with self.assertRaises(ValueError):
                self.action("calibrate", p1=p1, p2=p2, value=v, unit="m", description="fixture")
        self.assertEqual(before, self.projects.read(self.pid))

    def test_independent_scale_check_flags_inconsistency(self):
        self.plan()
        h = self.action(
            "check_calibration",
            p1=[100, 100],
            p2=[400, 100],
            value=4,
            unit="m",
            description="Independent inconsistent fixture",
        )
        self.assertTrue(scale_check(h["floorplan"])["inconsistent"])
        self.assertFalse(self.project["transfer_assessment"]["physical_placement_verified"])

    def test_scale_revision_preserves_geometry(self):
        self.plan()
        rid = self.room()
        before = copy.deepcopy(self.project["home_transfer"]["rooms"][rid]["boundary"])
        h = self.action(
            "calibrate",
            p1=[100, 100],
            p2=[700, 100],
            value=5,
            unit="m",
            description="Revised fixture",
        )
        self.assertEqual(before, h["rooms"][rid]["boundary"])
        self.assertTrue(h["rooms"][rid]["calibration_needs_review"])
        self.assertEqual(2, len(h["floorplan"]["calibrations"]))

    def test_trace_requires_scale_and_retains_pixel_relation(self):
        self.store.upload(self.pid, self.image(), "fixture", True)
        with self.assertRaises(ValueError):
            self.action(
                "room",
                boundary=[[100, 100], [700, 100], [700, 500], [100, 500]],
                coordinate_space="image_pixels",
            )
        self.plan()
        h = self.action(
            "room",
            boundary=[[100, 100], [700, 100], [700, 500], [100, 500]],
            coordinate_space="image_pixels",
        )
        r = list(h["rooms"].values())[0]
        self.assertEqual(r["boundary"][0], [1, 1])
        self.assertEqual(r["trace_pixels"][0], [100, 100])
        self.assertIsNone(r["height"])

    def test_concave_polygon_and_validation(self):
        rid = self.room([[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]])
        self.assertEqual(
            6, len(room_scene(self.project["home_transfer"], rid)["floor"]["boundary"])
        )
        for poly in (
            [[0, 0], [2, 2], [0, 2], [2, 0]],
            [[0, 0], [0, 0], [2, 1]],
            [[0, 0], [1, 0], [2, 0]],
            [[0, 0], [float("inf"), 1], [0, 2]],
        ):
            with self.assertRaises(ValueError):
                polygon(poly)
        self.assertFalse(
            contains(
                [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]], [[1, 1], [5, 1], [5, 5], [1, 5]]
            )
        )

    def test_features_and_obstacles(self):
        rid = self.room()
        h = self.action(
            "room",
            room_id=rid,
            name="Living",
            boundary=[[1, 1], [7, 1], [7, 5], [1, 5]],
            features=[
                {"kind": "door", "edge": 0, "offset": 1, "width": 0.9, "swing": "left"},
                {"kind": "window", "edge": 1, "offset": 1, "width": 1},
            ],
            obstacles=[{"boundary": [[5, 3], [6, 3], [6, 4], [5, 4]]}],
        )
        scene = room_scene(h, rid)
        self.assertEqual(2, len(scene["openings"]))
        self.assertIsNone(h["rooms"][rid]["height"])
        self.assertEqual("unknown_visualization_only", scene["provenance"]["height_status"])
        with self.assertRaises(ValueError):
            self.action(
                "room",
                room_id=rid,
                boundary=[[1, 1], [7, 1], [7, 5], [1, 5]],
                features=[{"kind": "door", "edge": 0, "offset": 6, "width": 1}],
            )

    def test_alternative_independence_and_identity(self):
        h = self.place(self.room())
        a = h["active_layout"]
        old = copy.deepcopy(h["layouts"][a])
        h = self.action("alternative", duplicate=True, name="B")
        b = h["active_layout"]
        p = h["layouts"][b]["placements"][0]
        self.assertNotEqual(old["placements"][0]["id"], p["id"])
        self.assertEqual(self.iid, p["inventory_item_id"])
        self.action("update", id=p["id"], position=[4, 4, 0], rotation_degrees=90)
        self.assertEqual(old, self.project["home_transfer"]["layouts"][a])
        self.assertEqual(1, len(self.inventory.read()["items"]))

    def test_no_duplicate_owned_placement(self):
        rid = self.room()
        self.place(rid)
        with self.assertRaises(ValueError):
            self.place(rid)
        self.assertEqual(
            1,
            len(
                self.project["home_transfer"]["layouts"][
                    self.project["home_transfer"]["active_layout"]
                ]["placements"]
            ),
        )

    def test_unknown_dimensions_are_explicit_and_unverified(self):
        h = self.place(self.room())
        p = h["layouts"][h["active_layout"]]["placements"][0]
        self.assertEqual("unknown_approximate_placeholder", p["dimension_status"])
        self.assertFalse(p["physical_placement_verified"])
        self.assertFalse(self.project["transfer_assessment"]["physical_placement_verified"])

    def test_collision_clearance_and_obstacles(self):
        rid = self.room()
        self.place(rid)
        item = self.inventory.action(
            {"operation": "create", "display_name": "Fixture table", "category": "table"}
        )["item"]
        h = self.action(
            "place",
            inventory_item_id=item["inventory_item_id"],
            room_id=rid,
            position=[3, 3, 0],
            approximate_acknowledged=True,
        )
        self.assertIn(
            "furniture_overlap",
            [w["type"] for w in self.project["transfer_assessment"]["warnings"]],
        )
        p = h["layouts"][h["active_layout"]]["placements"][-1]
        self.action("update", id=p["id"], position=[8, 8, 0])
        self.assertIn(
            "outside_room_or_across_wall",
            [w["type"] for w in self.project["transfer_assessment"]["warnings"]],
        )

    def test_3d_sync_keeps_owned_reference(self):
        rid = self.room()
        h = self.place(rid)
        scene = room_scene(h, rid)
        scene["furniture"][0]["position"] = [4, 3, 0]
        scene["furniture"][0]["rotation_degrees"] = 45
        h = self.action("sync_scene", scene=scene)
        p = h["layouts"][h["active_layout"]]["placements"][0]
        self.assertEqual(p["position"], [4, 3, 0])
        self.assertEqual(self.iid, p["inventory_item_id"])
        self.assertFalse(presentation(room_scene(h, rid))["physical_placement_verified"])
        with self.assertRaises(ValueError):
            edit_scene(room_scene(h, rid), {"type": "resize", "dimensions": [8, 6, 3]})

    def test_undo_redo_and_save_reopen(self):
        h = self.place(self.room())
        p = h["layouts"][h["active_layout"]]["placements"][0]
        self.action("update", id=p["id"], position=[4, 4, 0])
        self.action("undo")
        self.assertEqual(
            [3, 3, 0],
            self.project["home_transfer"]["layouts"][h["active_layout"]]["placements"][0][
                "position"
            ],
        )
        self.action("redo")
        self.assertEqual(
            [4, 4, 0],
            self.store.read(self.pid)["home_transfer"]["layouts"][h["active_layout"]]["placements"][
                0
            ]["position"],
        )

    def test_decisions_and_preference(self):
        self.action("decision", inventory_item_id=self.iid, status="consider_selling")
        h = self.action("prefer")
        self.assertEqual(h["active_layout"], h["preferred_layout"])
        self.assertEqual(
            "consider_selling", h["layouts"][h["active_layout"]]["decisions"][self.iid]
        )

    def test_portable_export_import_and_ids(self):
        self.plan()
        self.place(self.room())
        before = self.inventory.path.read_bytes()
        raw = self.store.export(self.pid)
        p = self.store.import_bundle(raw)
        self.assertNotEqual(self.pid, p["id"])
        self.assertEqual(
            self.iid,
            list(p["home_transfer"]["layouts"].values())[0]["placements"][0]["inventory_item_id"],
        )
        self.assertEqual(before, self.inventory.path.read_bytes())
        self.assertEqual(p["id"], p["home_transfer"]["destination_project_id"])

    def test_unsafe_archives_rejected(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("../owned-inventory.json", "bad")
        with self.assertRaises(ValueError):
            self.store.import_bundle(b.getvalue())

    def test_forged_verification_and_schema_rejected(self):
        h = copy.deepcopy(self.project["home_transfer"])
        h["physical_accuracy_verified"] = True
        with self.assertRaises(ValueError):
            validate_home(h)
        h["schema_version"] = 99
        with self.assertRaises(ValueError):
            validate_home(h)

    def test_pdf_pages_locally_rendered(self):
        from PyPDF2 import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=500, height=300)
        writer.add_blank_page(width=300, height=500)
        b = io.BytesIO()
        writer.write(b)
        # Real PDFKit renderer, not a mocked rasterization.
        p = self.store.upload(self.pid, b.getvalue(), "generated.pdf", True, page=1)[
            "home_transfer"
        ]["floorplan"]
        self.assertEqual(2, p["pages"])
        self.assertLess(p["width"], p["height"])
        self.assertEqual(1, p["page"])
        with self.assertRaises(ValueError):
            self.store.upload(self.pid, b.getvalue(), "generated.pdf", True, page=2)
        encrypted = PdfWriter()
        encrypted.add_blank_page(width=100, height=100)
        encrypted.encrypt("secret")
        b = io.BytesIO()
        encrypted.write(b)
        with self.assertRaises(ValueError):
            self.store.upload(self.pid, b.getvalue(), "locked.pdf", True)

    def test_exact_wall_length_keeps_measurement_history(self):
        rid = self.room()
        h = self.action(
            "wall_length",
            room_id=rid,
            edge=0,
            value=500,
            unit="cm",
            description="Explicit fixture length",
        )
        self.assertEqual([6, 1], h["rooms"][rid]["boundary"][1])
        self.assertEqual(5, h["rooms"][rid]["wall_measurements"][0]["metres"])
        self.assertTrue(h["rooms"][rid]["calibration_needs_review"])

    def test_aligned_partial_obstacle_overlap_is_detected(self):
        from room_reconstruction.polygon import overlaps

        self.assertTrue(
            overlaps([[0, 0], [2, 0], [2, 1], [0, 1]], [[1, 0], [3, 0], [3, 1], [1, 1]])
        )
        self.assertFalse(
            overlaps([[0, 0], [2, 0], [2, 1], [0, 1]], [[2, 0], [3, 0], [3, 1], [2, 1]])
        )

    def test_recorded_size_transfers_without_changing_evidence(self):
        for field, value in [("width", 210), ("depth", 90), ("height", 82)]:
            self.inventory.action(
                {
                    "operation": "evidence",
                    "inventory_item_id": self.iid,
                    "field": field,
                    "value": value,
                    "unit": "cm",
                    "source": "user_measured",
                    "description": "Software fixture only",
                }
            )
        before = self.inventory.path.read_bytes()
        h = self.place(self.room())
        p = h["layouts"][h["active_layout"]]["placements"][0]
        for expected, actual in zip([2.1, 0.9, 0.82], p["size"]):
            self.assertAlmostEqual(expected, actual)
        self.assertEqual("recorded_unverified", p["dimension_status"])
        self.assertEqual(3, len(p["dimension_evidence"]))
        self.assertEqual(before, self.inventory.path.read_bytes())

    def test_forged_stored_scale_is_rejected(self):
        h = self.plan()
        h["floorplan"]["calibrations"][0]["metres_per_pixel"] = float("inf")
        with self.assertRaises(ValueError):
            validate_home(h)

    def test_destination_requires_own_portable_export(self):
        with self.assertRaises(ValueError):
            self.projects.export(self.pid)
