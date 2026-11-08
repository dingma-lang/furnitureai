import copy
import io
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from room_viewer.home_transfer import TransferStore
from room_viewer.inventory import InventoryStore
from room_viewer.photo_projects import ProjectStore
from room_viewer.workflow import dashboard, project_progress


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projects = ProjectStore(self.tmp.name)
        self.inventory = InventoryStore(self.projects)
        self.transfer = TransferStore(self.projects, self.inventory)

    def photo(self):
        p = self.projects.create("Workflow photo")
        b = io.BytesIO()
        Image.new("RGB", (100, 80)).save(b, format="PNG")
        return self.projects.upload(p["id"], b.getvalue(), "fixture.png", True)

    def test_dashboard_preserves_existing_project_bytes(self):
        p = self.photo()
        file = self.projects.folder(p["id"]) / "project.json"
        before = file.read_bytes()
        result = dashboard(self.projects, self.inventory)
        self.assertEqual(before, file.read_bytes())
        self.assertFalse(result["model_initialization_required"])
        self.assertEqual("Photo uploaded", result["projects"][0]["state"])

    def test_ai_output_is_not_user_review_or_owned_inventory(self):
        p = self.photo()
        doc = self.projects.read(p["id"])
        doc["photos"][0]["reviewed"] = [
            {"label": "sofa", "confirmed": False, "review_status": "unreviewed"}
        ]
        result = project_progress(doc, {"items": {}})
        self.assertEqual("Furniture awaiting review", result["state"])
        self.assertNotIn("Furniture saved", result["steps"])

    def test_confirmed_detection_does_not_claim_saved_furniture(self):
        p = self.photo()
        doc = self.projects.read(p["id"])
        doc["photos"][0]["reviewed"] = [
            {"label": "sofa", "confirmed": True, "review_status": "confirmed"}
        ]
        r = project_progress(doc, {"items": {}})
        self.assertEqual("Furniture confirmed by user", r["state"])
        self.assertFalse(r["physical_placement_verified"])

    def test_progress_reflects_explicit_owned_link(self):
        p = self.photo()
        item = {"links": [{"project_id": p["id"], "observation_id": "fixture"}]}
        r = project_progress(p, {"items": {"fixture": item}})
        self.assertEqual("Furniture saved", r["state"])

    def test_destination_project_type_and_partial_progress(self):
        p = self.transfer.create({"name": "Move"})
        r = dashboard(self.projects, self.inventory)["projects"][0]
        self.assertEqual("moving_plan", r["type"])
        self.assertEqual("Project created", r["state"])
        self.assertEqual([], r["warnings"])
        self.assertIsNone(p["home_transfer"]["floorplan"])

    def test_missing_photo_asset_is_reported_without_deleting_reviews(self):
        p = self.photo()
        photo = p["photos"][0]
        (self.projects.folder(p["id"]) / "photos" / photo["id"] / "processing.png").unlink()
        r = dashboard(self.projects, self.inventory)
        self.assertIn("unavailable", r["projects"][0]["warnings"][0])
        self.assertEqual(1, len(self.projects.read(p["id"])["photos"]))

    def test_missing_floorplan_preview_preserves_geometry(self):
        p = self.transfer.create({"name": "Move"})
        b = io.BytesIO()
        Image.new("RGB", (100, 80)).save(b, format="PNG")
        p = self.transfer.upload(p["id"], b.getvalue(), "fixture.png", True)
        plan = p["home_transfer"]["floorplan"]
        (self.projects.folder(p["id"]) / "floorplans" / plan["preview"]).unlink()
        r = dashboard(self.projects, self.inventory)
        self.assertEqual("Floor plan added", r["projects"][0]["state"])
        self.assertIn("unavailable", r["projects"][0]["warnings"][0])

    def test_invalid_project_file_does_not_hide_other_projects(self):
        p = self.photo()
        folder = self.projects.root / ("0" * 32)
        folder.mkdir()
        (folder / "project.json").write_text("{bad")
        r = dashboard(self.projects, self.inventory)
        self.assertEqual(1, r["unavailable_projects"])
        self.assertEqual(p["id"], r["projects"][0]["id"])

    def test_duplicate_names_preserve_distinct_project_ids(self):
        a = self.projects.create("Same")
        b = self.projects.create("Same")
        r = dashboard(self.projects, self.inventory)
        self.assertEqual({a["id"], b["id"]}, {p["id"] for p in r["projects"]})

    def test_custom_detection_name_preserves_original_prediction(self):
        p = self.photo()
        pid = p["id"]
        photo = p["photos"][0]["id"]
        self.projects.action(
            pid,
            {
                "type": "correct",
                "photo_id": photo,
                "operation": "add",
                "label": "sofa",
                "box": [1, 1, 50, 60],
            },
        )
        doc = self.projects.read(pid)
        did = doc["photos"][0]["reviewed"][0]["detection_id"]
        original = copy.deepcopy(doc["photos"][0].get("predictions"))
        self.projects.action(
            pid,
            {
                "type": "correct",
                "photo_id": photo,
                "detection_id": did,
                "operation": "update",
                "display_name": "My blue sofa",
            },
        )
        r = self.projects.read(pid)
        self.assertEqual("My blue sofa", r["photos"][0]["reviewed"][0]["display_name"])
        self.assertEqual(original, r["photos"][0].get("predictions"))
        self.assertTrue(r["photos"][0]["corrections"])
        self.assertEqual("My blue sofa", r["photos"][0]["corrections"][-1]["display_name"])

    def test_name_validation(self):
        p = self.photo()
        pid = p["id"]
        photo = p["photos"][0]["id"]
        with self.assertRaises(ValueError):
            self.projects.action(
                pid,
                {
                    "type": "correct",
                    "photo_id": photo,
                    "operation": "add",
                    "display_name": "x" * 161,
                    "label": "sofa",
                    "box": [1, 1, 50, 60],
                },
            )

    def test_dashboard_does_not_initialize_ml(self):
        self.photo()
        with (
            patch("subprocess.Popen", side_effect=AssertionError("No model worker allowed")),
            patch("subprocess.run", side_effect=AssertionError("No worker allowed")),
        ):
            dashboard(self.projects, self.inventory)

    def test_all_rejected_suggestions_do_not_claim_review_is_pending(self):
        p = self.photo()
        p["photos"][0]["reviewed"] = [{"confirmed": False, "review_status": "rejected"}]
        r = project_progress(p, {"items": {}})
        self.assertEqual("Photo uploaded", r["state"])
        self.assertNotIn("Furniture awaiting review", r["steps"])
