import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from room_viewer.furniture_budget import BudgetStore
from room_viewer.home_transfer import TransferStore
from room_viewer.inventory import InventoryStore
from room_viewer.photo_projects import ProjectStore
from room_viewer.replacements import ReplacementStore, proposals
from room_viewer.service import ROOT, presentation


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projects = ProjectStore(self.tmp.name)
        self.inventory = InventoryStore(self.projects)
        self.transfer = TransferStore(self.projects, self.inventory)
        self.store = ReplacementStore(self.transfer)
        self.budget = BudgetStore(self.transfer)
        self.project = self.transfer.create({"name": "Replacement fixture"})
        self.pid = self.project["id"]
        self.lid = self.project["home_transfer"]["active_layout"]
        item = self.inventory.action(
            {"operation": "create", "display_name": "Original fixture sofa", "category": "sofa"}
        )["item"]
        self.iid = item["inventory_item_id"]
        p = self.transfer.action(
            self.pid,
            {
                "operation": "room",
                "name": "Room",
                "boundary": [[0, 0], [6, 0], [6, 4], [0, 4]],
                "height": None,
                "evidence_status": "estimated",
            },
        )
        self.rid = next(iter(p["home_transfer"]["rooms"]))
        p = self.transfer.action(
            self.pid,
            {
                "operation": "place",
                "room_id": self.rid,
                "inventory_item_id": self.iid,
                "position": [2, 2, 0],
                "approximate_acknowledged": True,
            },
        )
        self.original = p["home_transfer"]["layouts"][self.lid]["placements"][0]
        self.key = uuid4().hex

    def act(self, operation, **extra):
        return self.store.action(
            self.pid,
            {"operation": operation, "layout_id": self.lid, "proposal_id": self.key, **extra},
        )

    def create(self, **extra):
        return self.act(
            "create",
            original_placement_id=self.original["id"],
            name="Illustrative compact sofa",
            **extra,
        )

    def proposal(self):
        return proposals(self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid])[
            self.key
        ]

    def preview(self, **extra):
        return self.store.preview(
            self.pid, {"layout_id": self.lid, "proposal_id": self.key, **extra}
        )

    def apply(self, **extra):
        preview = self.preview(**extra)
        return self.act(
            "apply",
            preview_token=preview["preview_token"],
            position=preview["candidate"]["position"],
            rotation_degrees=preview["candidate"]["rotation_degrees"],
            acknowledge_approximate=True,
        )

    def test_manual_proposal_is_not_owned_inventory(self):
        before = self.inventory.path.read_bytes()
        self.create()
        p = self.proposal()
        self.assertEqual(self.iid, p["original_inventory_item_id"])
        self.assertFalse(p["ownership_verified"])
        self.assertEqual(before, self.inventory.path.read_bytes())
        self.assertEqual(
            [self.original],
            self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid]["placements"],
        )

    def test_unknown_dimensions_remain_unknown(self):
        self.create()
        v = self.preview()
        self.assertEqual([None, None, None], v["candidate"]["known_dimensions_m"])
        self.assertIsNone(v["candidate"]["inventory_item_id"])
        self.assertEqual("mixed_source_and_approximate", v["candidate"]["dimension_status"])

    def test_partial_and_conflicting_evidence(self):
        self.create()
        entry = {
            "value": 180,
            "unit": "cm",
            "source": "user_entered",
            "description": "Illustrative dimension",
        }
        self.act("update", dimensions={"width": entry})
        p = self.proposal()
        old = p["specifications"]["width"]["active_evidence_id"]
        self.act("update", dimensions={"width": {**entry, "value": 190}})
        p = self.proposal()
        self.assertEqual(2, len(p["specifications"]["width"]["evidence"]))
        self.act("update", select_evidence={"width": old})
        v = self.preview()
        self.assertAlmostEqual(1.8, v["candidate"]["size"][0])
        self.assertIsNone(v["candidate"]["known_dimensions_m"][1])

    def test_repeated_save_does_not_duplicate_evidence(self):
        self.create()
        d = {"width": {"value": 1.8, "unit": "m", "source": "estimated", "description": "Example"}}
        self.act("update", dimensions=d)
        self.act("update", dimensions=d)
        self.assertEqual(1, len(self.proposal()["specifications"]["width"]["evidence"]))

    def test_preview_has_no_persistence_effect(self):
        self.create()
        path = self.projects.folder(self.pid) / "project.json"
        before = path.read_bytes()
        v = self.preview(position=[3, 2, 0], rotation_degrees=45)
        self.assertEqual(before, path.read_bytes())
        self.assertFalse(v["saved_layout_mutated"])
        self.assertEqual(45, v["candidate"]["rotation_degrees"])

    def test_explicit_apply_and_revert_preserve_original(self):
        self.create()
        self.apply()
        p = self.proposal()
        layout = self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid]
        self.assertEqual(1, len(layout["placements"]))
        self.assertEqual(p["scene_placement_id"], layout["placements"][0]["id"])
        self.assertNotEqual(self.original["id"], p["scene_placement_id"])
        self.act("revert")
        self.assertEqual(
            [self.original],
            self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid]["placements"],
        )
        self.assertEqual(1, len(self.inventory.read()["items"]))

    def test_apply_requires_preview_and_ack(self):
        self.create()
        with self.assertRaises(ValueError):
            self.act("apply", acknowledge_approximate=True, preview_token="bad")
        v = self.preview()
        with self.assertRaises(ValueError):
            self.act("apply", preview_token=v["preview_token"])

    def test_stale_preview_rejected_without_write(self):
        self.create()
        v = self.preview()
        self.act("update", name="Changed source proposal")
        before = self.projects.read(self.pid)
        with self.assertRaises(ValueError):
            self.act("apply", preview_token=v["preview_token"], acknowledge_approximate=True)
        self.assertEqual(before, self.projects.read(self.pid))

    def test_repeated_apply_idempotent(self):
        self.create()
        v = self.preview()
        body = {
            "preview_token": v["preview_token"],
            "position": v["candidate"]["position"],
            "rotation_degrees": 0,
            "acknowledge_approximate": True,
        }
        self.act("apply", **body)
        before = self.projects.read(self.pid)
        self.act("apply", **body)
        self.assertEqual(before, self.projects.read(self.pid))

    def test_create_idempotent_and_duplicate_prevention(self):
        self.create()
        self.create()
        self.assertEqual(
            1, len(proposals(self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid]))
        )
        with self.assertRaises(ValueError):
            self.store.action(
                self.pid,
                {
                    "operation": "create",
                    "layout_id": self.lid,
                    "proposal_id": uuid4().hex,
                    "original_placement_id": self.original["id"],
                },
            )

    def test_layout_independence_and_duplicate_remapping(self):
        self.create()
        self.apply()
        old = copy.deepcopy(self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid])
        new = self.transfer.action(
            self.pid, {"operation": "alternative", "duplicate": True, "name": "B"}
        )
        nlid = new["home_transfer"]["active_layout"]
        alt = new["home_transfer"]["layouts"][nlid]
        newp = next(iter(proposals(alt).values()))
        self.assertNotEqual(self.key, newp["id"])
        self.assertNotEqual(self.proposal()["scene_placement_id"], newp["scene_placement_id"])
        self.assertEqual(old, new["home_transfer"]["layouts"][self.lid])
        self.assertEqual(nlid, newp["layout_id"])

    def test_scene_move_rotate_and_size_do_not_modify_specs(self):
        self.create()
        self.apply()
        p = self.proposal()
        new = self.transfer.action(
            self.pid,
            {
                "operation": "update",
                "id": p["scene_placement_id"],
                "position": [3, 2, 0],
                "rotation_degrees": 60,
                "size": [1.5, 0.7, 0.8],
            },
        )
        placed = new["home_transfer"]["layouts"][self.lid]["placements"][0]
        self.assertEqual(60, placed["rotation_degrees"])
        self.assertEqual("layout_specific_estimate", placed["dimension_status"])
        self.assertEqual(p["specifications"], self.proposal()["specifications"])

    def test_remove_scene_recover_original_after_reopen(self):
        self.create()
        self.apply()
        self.transfer.action(
            self.pid, {"operation": "remove", "id": self.proposal()["scene_placement_id"]}
        )
        self.assertEqual("removed", self.proposal()["status"])
        self.store = ReplacementStore(self.transfer)
        self.act("revert")
        self.assertEqual(
            [self.original],
            self.projects.read(self.pid)["home_transfer"]["layouts"][self.lid]["placements"],
        )

    def test_undo_redo_restores_relationships(self):
        self.create()
        self.apply()
        self.transfer.action(self.pid, {"operation": "undo"})
        self.assertEqual("proposed", self.proposal()["status"])
        self.transfer.action(self.pid, {"operation": "redo"})
        self.assertEqual("applied", self.proposal()["status"])

    def test_link_purchase_is_explicit_atomic_and_idempotent(self):
        self.create()
        self.apply()
        self.assertNotIn("furniture_budget", self.projects.read(self.pid))
        self.act("link_purchase")
        event = self.proposal()["purchase_event_id"]
        self.act("link_purchase")
        s = self.projects.read(self.pid)["furniture_budget"]["scenarios"][self.lid]
        self.assertEqual([event], list(s["purchases"]))
        self.assertIsNone(s["purchases"][event]["unit_price"]["cents"])
        self.assertEqual({}, s["sales"])

    def test_linked_purchase_calculation_keeps_exact_cents(self):
        self.create()
        self.act("link_purchase")
        event = self.proposal()["purchase_event_id"]
        self.budget.action(
            self.pid,
            {
                "operation": "item_action",
                "layout_id": self.lid,
                "inventory_item_id": self.iid,
                "action": "replace",
            },
        )
        self.budget.action(
            self.pid,
            {
                "operation": "purchase",
                "layout_id": self.lid,
                "record": {
                    "id": event,
                    "description": "Example",
                    "replacement_of": self.iid,
                    "unit_price": "100.10",
                    "quantity": 3,
                    "discount": "0",
                    "goods_taxable": False,
                    "delivery": "0",
                    "assembly": "0",
                    "other": "0",
                    "source": "Illustrative only",
                    "price_status": "estimate",
                },
            },
        )
        r = self.budget.get(self.pid)
        self.assertEqual(
            30030, r["comparison"][self.lid]["calculation"]["totals"]["expected_net"]["cents"]
        )
        self.assertEqual(1, len(r["budget"]["scenarios"][self.lid]["purchases"]))

    def test_linked_event_cannot_be_orphaned(self):
        self.create()
        self.act("link_purchase")
        event = self.proposal()["purchase_event_id"]
        with self.assertRaises(ValueError):
            self.budget.action(
                self.pid,
                {
                    "operation": "remove",
                    "layout_id": self.lid,
                    "collection": "purchases",
                    "record_id": event,
                },
            )
        with self.assertRaises(ValueError):
            self.budget.action(self.pid, {"operation": "undo", "layout_id": self.lid})
        self.act("unlink_purchase")
        self.budget.action(
            self.pid,
            {
                "operation": "remove",
                "layout_id": self.lid,
                "collection": "purchases",
                "record_id": event,
            },
        )

    def test_invalid_purchase_and_original_references(self):
        self.create()
        with self.assertRaises(ValueError):
            self.act("link_purchase", purchase_event_id=uuid4().hex)
        self.key = uuid4().hex
        with self.assertRaises(ValueError):
            self.act("create", original_placement_id="not-present")

    def test_export_import_and_revert(self):
        self.create()
        self.apply()
        self.act("link_purchase")
        new = self.transfer.import_bundle(self.transfer.export(self.pid))
        nid = new["id"]
        p = proposals(new["home_transfer"]["layouts"][self.lid])[self.key]
        self.assertEqual(nid, p["destination_project_id"])
        self.store.action(
            nid, {"operation": "revert", "layout_id": self.lid, "proposal_id": self.key}
        )
        self.assertEqual(
            [self.original],
            self.projects.read(nid)["home_transfer"]["layouts"][self.lid]["placements"],
        )

    def test_interrupted_joint_write_keeps_previous_state(self):
        self.create()
        path = self.projects.folder(self.pid) / "project.json"
        before = path.read_bytes()
        with patch(
            "room_viewer.photo_projects.atomic_json",
            side_effect=OSError("Simulated interrupted write"),
        ):
            with self.assertRaises(OSError):
                self.act("link_purchase")
        self.assertEqual(before, path.read_bytes())
        self.assertIsNone(self.proposal()["purchase_event_id"])
        self.assertNotIn("furniture_budget", self.projects.read(self.pid))
        self.act("link_purchase")
        self.assertIsNotNone(self.proposal()["purchase_event_id"])

    def test_partial_temporary_file_is_ignored_and_next_write_recovers(self):
        self.create()
        path = self.projects.folder(self.pid) / "project.tmp"
        path.write_text("{interrupted")
        self.assertEqual("proposed", self.proposal()["status"])
        self.act("link_purchase")
        self.assertFalse(path.exists())

    def test_unverified_geometry_never_claims_fit(self):
        self.create()
        self.apply()
        from room_viewer.home_transfer import room_scene

        r = presentation(room_scene(self.projects.read(self.pid)["home_transfer"], self.rid))
        self.assertFalse(r["physical_placement_verified"])
        self.assertEqual("disabled", r["status"])

    def test_candidate_selection_uses_saved_production_results(self):
        # Unit-only observation fixture; real retrieval/browser evidence is exercised separately.
        catalog = json.loads((ROOT / "data/catalogs/abo-phase6d/catalog.json").read_text())
        product = next(p for p in catalog["products"] if p["label"] == "sofa")
        source = self.projects.create("Candidate source fixture")
        doc = self.projects.read(source["id"])
        photo_id = uuid4().hex
        doc["photos"] = [
            {
                "id": photo_id,
                "product_matches": {
                    "d0": {
                        "state": "complete",
                        "candidates": [{"product_id": product["product_id"], "similarity": 0.7}],
                        "decisions": {},
                    }
                },
            }
        ]
        from room_viewer.product_matching import observation_signature

        photo = doc["photos"][0]
        photo["sha256"] = "0" * 64
        photo["reviewed"] = [
            {
                "detection_id": "d0",
                "label": "sofa",
                "box": [0, 0, 100, 100],
                "confirmed": True,
                "review_status": "confirmed",
            }
        ]
        photo["product_matches"]["d0"]["observation_signature"] = observation_signature(
            photo, photo["reviewed"][0]
        )
        self.projects.write(doc)
        library = self.inventory.read()
        library["items"][self.iid]["links"] = [
            {
                "project_id": source["id"],
                "observation_id": photo_id + ":d0",
                "scene_object_id": None,
            }
        ]
        from room_viewer.photo_projects import atomic_json

        atomic_json(self.inventory.path, library)
        self.create()
        self.act("update", candidate_id=product["product_id"])
        p = self.proposal()
        self.assertEqual(product["product_id"], p["catalog_candidate"]["product_id"])
        self.assertFalse(p["catalog_candidate"]["identity_verified"])
        with self.assertRaises(ValueError):
            self.act("update", candidate_id="unseen-retailer-listing")

    def test_applied_visual_replacement_without_purchase_link_is_unbudgeted(self):
        self.create()
        self.apply()
        self.budget.action(self.pid, {"operation": "ensure", "layout_id": self.lid})
        r = self.budget.action(
            self.pid,
            {
                "operation": "item_action",
                "layout_id": self.lid,
                "inventory_item_id": self.iid,
                "action": "retain",
            },
        )
        c = r["comparison"][self.lid]["calculation"]
        self.assertIsNone(c["totals"]["gross_upfront"]["cents"])
        self.assertTrue(any("replacement_purchase_link" in m for m in c["missing_assumptions"]))
        self.assertEqual({}, r["budget"]["scenarios"][self.lid]["purchases"])

    def test_response_evaluation_failure_does_not_commit_a_joint_update(self):
        self.create()
        before = self.projects.read(self.pid)
        with patch.object(
            self.store, "public", side_effect=ValueError("Simulated invalid result evaluation")
        ):
            with self.assertRaises(ValueError):
                self.act("link_purchase")
        self.assertEqual(before, self.projects.read(self.pid))

    def test_malformed_owned_reference_and_extreme_pose_are_rejected(self):
        self.create()
        before = self.projects.read(self.pid)
        with self.assertRaises(ValueError):
            self.act("update", position=[1e300, 2, 0])
        self.assertEqual(before, self.projects.read(self.pid))
        from room_viewer.replacements import validate_project

        bad = copy.deepcopy(before)
        record = bad["home_transfer"]["layouts"][self.lid]["replacement_planning"]["proposals"][
            self.key
        ]
        record["original_inventory_item_id"] = None
        record["original_placement"]["inventory_item_id"] = None
        with self.assertRaises(ValueError):
            validate_project(bad)

    def test_undo_and_relink_reuse_existing_purchase_without_duplicate_costs(self):
        self.create()
        self.act("link_purchase")
        event = self.proposal()["purchase_event_id"]
        self.transfer.action(self.pid, {"operation": "undo"})
        self.assertIsNone(self.proposal()["purchase_event_id"])
        self.act("link_purchase")
        self.assertEqual(event, self.proposal()["purchase_event_id"])
        self.assertEqual(
            1,
            len(
                self.projects.read(self.pid)["furniture_budget"]["scenarios"][self.lid]["purchases"]
            ),
        )
