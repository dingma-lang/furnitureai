import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from furniture_bench.data import read_json
from room_viewer.inventory import (
    InventoryStore,
    active_dimensions,
    attach_catalog,
    dimension,
    empty_item,
    validate_item,
)
from room_viewer.photo_projects import ProjectStore
from room_viewer.service import ROOT, presentation


class OwnedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.projects = ProjectStore(self.directory.name)
        self.library = InventoryStore(self.projects)
        self.project = self.projects.create("Inventory fixture")
        self.pid = self.project["id"]

    def create(self, **extra):
        return self.library.action(
            {"operation": "create", "display_name": "Fixture sofa", "category": "sofa", **extra}
        )["item"]

    def action(self, item, operation, **extra):
        return self.library.action(
            {"operation": operation, "inventory_item_id": item["inventory_item_id"], **extra}
        )

    def measured(self, item):
        for field, value in [("width", 210), ("depth", 90), ("height", 82)]:
            self.action(
                item,
                "evidence",
                field=field,
                value=value,
                unit="cm",
                source="user_measured",
                description="UNIT TEST tape fixture, not real measurements",
            )
        return self.library.listing()["items"][item["inventory_item_id"]]

    def room(self, pid=None):
        return self.projects.action(
            pid or self.pid,
            {
                "type": "setup_room",
                "layout": "rectangular",
                "dimensions": [6, 4, 2.8],
                "dimension_status": "unknown",
                "source": "Unit test conceptual workspace",
                "approximate_acknowledged": True,
            },
        )

    def observation(self):
        raw = io.BytesIO()
        Image.new("RGB", (64, 48), "tan").save(raw, "PNG")
        project = self.projects.upload(self.pid, raw.getvalue(), "fixture.png", True)
        photo_id = project["photos"][-1]["id"]
        project = self.projects.action(
            self.pid,
            {
                "type": "correct",
                "photo_id": photo_id,
                "operation": "add",
                "label": "sofa",
                "box": [2, 2, 50, 40],
            },
        )
        detection_id = project["photos"][-1]["reviewed"][-1]["detection_id"]
        self.projects.action(
            self.pid,
            {
                "type": "correct",
                "photo_id": photo_id,
                "operation": "confirm",
                "detection_id": detection_id,
            },
        )
        return photo_id + ":" + detection_id

    def test_manual_unknown_creation_no_project_migration(self):
        before = (self.projects.folder(self.pid) / "project.json").read_bytes()
        item = self.create()
        self.assertTrue(item["inventory_item_id"].startswith("inv-"))
        self.assertEqual(item["identification"]["status"], "unknown")
        self.assertIsNone(item["specifications"]["width"]["active_evidence_id"])
        self.assertEqual(before, (self.projects.folder(self.pid) / "project.json").read_bytes())

    def test_length_units_and_invalid_values(self):
        for value, unit in [(100, "cm"), (1000, "mm"), (1, "m")]:
            self.assertEqual(dimension(value, unit), 1)
        self.assertAlmostEqual(dimension(12, "in"), 0.3048)
        self.assertAlmostEqual(dimension(1, "ft"), 0.3048)
        for value in [True, 0, -1, 0.0001, float("nan"), float("inf"), "210", 1000000]:
            with self.assertRaises(ValueError):
                dimension(value, "cm")
        with self.assertRaises(ValueError):
            dimension(1, "yards")

    def test_conflicting_measurements_are_retained_and_selectable(self):
        item = self.create()
        self.action(
            item,
            "evidence",
            field="width",
            value=210,
            unit="cm",
            source="user_measured",
            description="Fixture tape",
        )
        result = self.action(
            item,
            "evidence",
            field="width",
            value=82,
            unit="in",
            source="estimated",
            description="Fixture estimate",
        )["item"]
        record = result["specifications"]["width"]
        self.assertEqual(len(record["evidence"]), 2)
        self.assertEqual(record["active_evidence_id"], record["evidence"][1]["id"])
        result = self.action(
            item, "select_evidence", field="width", evidence_id=record["evidence"][0]["id"]
        )["item"]
        self.assertEqual(
            result["specifications"]["width"]["active_evidence_id"], record["evidence"][0]["id"]
        )

    def test_evidence_requires_source_and_document_references(self):
        item = self.create()
        for source, description in [
            ("verified_by_AI", "fixture"),
            ("user_measured", ""),
            ("manufacturer_document", "fixture"),
        ]:
            with self.assertRaises(ValueError):
                self.action(
                    item,
                    "evidence",
                    field="width",
                    value=210,
                    unit="cm",
                    source=source,
                    description=description,
                )

    def test_catalog_reported_fields_not_verified_or_length_invented_as_depth(self):
        product = read_json(ROOT / "data/catalogs/abo-phase6d/catalog.json")["products"][0]
        item = self.measured(self.create())
        prior = item["specifications"]["width"]["active_evidence_id"]
        attach_catalog(item, product, {"similarity": 0.8})
        validate_item(item)
        self.assertFalse(item["identification"]["catalog"]["exact_identity_verified"])
        self.assertEqual(
            item["identification"]["catalog"]["source_listing_id"], product["product_id"]
        )
        self.assertEqual(item["specifications"]["width"]["active_evidence_id"], prior)
        blank = empty_item("No dimensions", "sofa")
        attach_catalog(blank, product)
        self.assertIsNone(blank["specifications"]["depth"]["active_evidence_id"])

    def test_missing_metadata_remains_missing(self):
        item = empty_item("Unknown", "chair")
        attach_catalog(item, {"product_id": "fixture", "images": [], "rights": {}})
        self.assertIsNone(item["identification"]["catalog"]["reported_model_identifier"])
        self.assertTrue(all(not record["evidence"] for record in item["specifications"].values()))

    def test_clear_association_preserves_item_and_dimension_evidence(self):
        item = self.measured(self.create())
        result = self.action(item, "clear_catalog")["item"]
        self.assertEqual(result["inventory_item_id"], item["inventory_item_id"])
        self.assertEqual(result["specifications"], item["specifications"])

    def test_condition_is_user_recorded_and_persists(self):
        item = self.create()
        self.action(
            item,
            "edit",
            condition={
                "grade": "fair",
                "stains": "Fixture note only",
                "purchase_year": 2020,
                "original_price": {"amount": 500, "currency": "USD"},
            },
        )
        reopened = InventoryStore(ProjectStore(self.directory.name)).listing()["items"][
            item["inventory_item_id"]
        ]
        self.assertEqual(reopened["condition"]["grade"], "fair")
        self.assertEqual(reopened["condition"]["original_price"]["amount"], 500)

    def test_invalid_edit_leaves_saved_library_unchanged(self):
        item = self.create()
        before = self.library.path.read_bytes()
        for body in [
            {"quantity": -1},
            {"condition": {"grade": "AI_verified_damage"}},
            {"display_name": ""},
            {"quantity": True},
        ]:
            with self.assertRaises(ValueError):
                self.action(item, "edit", **body)
            self.assertEqual(self.library.path.read_bytes(), before)

    def test_detection_link_deduplicates_repeated_create(self):
        observation_id = self.observation()
        original = self.projects.read(self.pid)["photos"]
        one = self.create(project_id=self.pid, observation_id=observation_id)
        two = self.create(project_id=self.pid, observation_id=observation_id)
        self.assertEqual(one["inventory_item_id"], two["inventory_item_id"])
        self.assertEqual(self.projects.read(self.pid)["photos"], original)
        self.assertNotEqual(one["inventory_item_id"], observation_id)

    def test_multiple_confirmed_observations_and_explicit_merge_only(self):
        first = self.observation()
        item = self.create(project_id=self.pid, observation_id=first)
        second = self.observation()
        result = self.action(item, "link", project_id=self.pid, observation_id=second)["item"]
        self.assertEqual(len(result["links"]), 2)
        other = self.create()
        with self.assertRaises(ValueError):
            self.action(other, "link", project_id=self.pid, observation_id=first)

    def test_unconfirmed_observation_rejected(self):
        with self.assertRaises(ValueError):
            self.create(project_id=self.pid, observation_id="not-a-confirmed-observation")

    def test_dimensions_apply_keeps_position_and_fit_disabled(self):
        self.room()
        item = self.measured(self.create())
        result = self.action(
            item, "place", project_id=self.pid, acknowledge_approximate_position=True
        )
        entity = result["project"]["room"]["furniture"][0]
        for actual, expected in zip(entity["size"], [2.1, 0.9, 0.82]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(entity["inventory_item_id"], item["inventory_item_id"])
        self.assertIsNone(entity["design_evidence"]["position"]["physical_position"])
        self.assertEqual(presentation(result["project"]["room"])["status"], "disabled")
        self.assertEqual(
            result["project"]["inventory_dimension_history"][-1]["inventory_item_id"],
            item["inventory_item_id"],
        )

    def test_unknown_dimensions_allow_only_labeled_approximate_placeholder(self):
        self.room()
        item = self.create()
        result = self.action(
            item, "place", project_id=self.pid, acknowledge_approximate_position=True
        )
        entity = result["project"]["room"]["furniture"][0]
        self.assertIsNone(entity["design_evidence"]["dimensions"]["value"])
        self.assertEqual(entity["design_evidence"]["dimensions"]["status"], "unknown")
        with self.assertRaises(ValueError):
            self.action(
                item,
                "apply_dimensions",
                project_id=self.pid,
                scene_object_id=entity["id"],
                acknowledge_approximate_position=True,
            )

    def test_cross_project_reuse_preserves_owned_identity_not_location(self):
        item = self.measured(self.create())
        self.room()
        first = self.action(
            item, "place", project_id=self.pid, acknowledge_approximate_position=True
        )
        other = self.projects.create("Future room")["id"]
        self.room(other)
        second = self.action(item, "place", project_id=other, acknowledge_approximate_position=True)
        self.assertEqual(len(self.library.listing()["items"]), 1)
        self.assertNotEqual(
            first["project"]["room"]["furniture"][0]["id"],
            second["project"]["room"]["furniture"][0]["id"],
        )
        self.assertTrue(
            all(not link["physical_location_verified"] for link in second["item"]["links"])
        )

    def test_place_requires_ack_and_prevents_duplicate_scene_reference(self):
        self.room()
        item = self.measured(self.create())
        with self.assertRaises(ValueError):
            self.action(item, "place", project_id=self.pid)
        self.action(item, "place", project_id=self.pid, acknowledge_approximate_position=True)
        with self.assertRaises(ValueError):
            self.action(item, "place", project_id=self.pid, acknowledge_approximate_position=True)

    def test_component_measurement_cannot_define_assembled_size(self):
        item = self.measured(self.create())
        item = self.action(
            item,
            "evidence",
            field="width",
            value=100,
            unit="cm",
            source="user_measured",
            description="Fixture detachable leg",
            scope="component",
        )["item"]
        with self.assertRaises(ValueError):
            active_dimensions(item)

    def test_document_link_not_downloaded_or_independently_verified(self):
        item = self.create()
        result = self.action(
            item,
            "document",
            title="Fixture manual",
            url="https://example.test/manual.pdf",
            permission="unit test fixture",
            supports_identity=True,
            identity_support_note="Fixture model text, unverified",
        )["item"]
        self.assertEqual(result["documents"][0]["retrieval_status"], "link_only_not_retrieved")
        self.assertEqual(result["identification"]["status"], "documentation_supported_unverified")
        self.assertIsNone(result["documents"][0]["date_accessed"])
        for target in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https://user:password@example.test/x",
        ]:
            with self.assertRaises(ValueError):
                self.action(item, "document", url=target)

    def test_pdf_bad_or_unpermitted_bytes_do_not_change_record(self):
        item = self.create()
        before = self.library.path.read_bytes()
        with self.assertRaises(ValueError):
            self.library.extract_document(item["inventory_item_id"], "invalid", b"notpdf", True)
        self.assertEqual(self.library.path.read_bytes(), before)

    @unittest.skipUnless(
        importlib.util.find_spec("PyPDF2"), "Optional document extractor not installed"
    )
    def test_actual_pdf_text_extracts_suggestions_without_applying_measurements(self):
        from PyPDF2 import PdfWriter
        from PyPDF2.generic import DecodedStreamObject, DictionaryObject, NameObject

        writer = PdfWriter()
        writer.add_blank_page(width=300, height=200)
        page = writer.pages[0]
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Width: 210 cm Depth: 90 cm Height: 82 cm) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        raw = io.BytesIO()
        writer.write(raw)
        item = self.create()
        item = self.action(
            item,
            "document",
            title="UNIT TEST synthetic text PDF; not manufacturer evidence",
            url="https://example.test/fixture.pdf",
            permission="Software test fixture",
        )["item"]
        document_id = item["documents"][0]["id"]
        result = self.library.extract_document(
            item["inventory_item_id"], document_id, raw.getvalue(), True
        )["item"]
        self.assertEqual(len(result["documents"][0]["specification_suggestions"]), 3)
        self.assertIn("Width: 210 cm", result["documents"][0]["text_excerpt"])
        self.assertIsNone(result["specifications"]["width"]["active_evidence_id"])
        self.assertEqual(result["identification"]["status"], "unknown")

    def test_journal_completes_interrupted_inventory_and_project_update(self):
        from room_viewer.photo_projects import atomic_json

        item = self.measured(self.create())
        self.room()

        def fail_library(path, value):
            if Path(path) == self.library.path:
                raise OSError("UNIT TEST simulated library write interruption")
            return atomic_json(path, value)

        with (
            patch("room_viewer.inventory.atomic_json", side_effect=fail_library),
            self.assertRaises(OSError),
        ):
            self.action(item, "place", project_id=self.pid, acknowledge_approximate_position=True)
        self.assertTrue(self.library.journal.exists())
        reopened = InventoryStore(ProjectStore(self.directory.name))
        self.assertFalse(reopened.journal.exists())
        self.assertEqual(
            reopened.projects.read(self.pid)["room"]["furniture"][0]["inventory_item_id"],
            item["inventory_item_id"],
        )
        self.assertTrue(reopened.listing()["items"][item["inventory_item_id"]]["links"])

    def test_portable_export_import_idempotent_no_local_photos_or_links(self):
        item = self.measured(self.create())
        bundle = self.library.export()
        self.assertFalse(bundle["items"][item["inventory_item_id"]]["links"])
        self.assertFalse(bundle["items"][item["inventory_item_id"]]["owned_photos"])
        result = self.library.import_data(bundle)
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["existing_ids_preserved"], 1)
        new_projects = ProjectStore(Path(self.directory.name) / "another-owner")
        imported = InventoryStore(new_projects).import_data(bundle)
        self.assertEqual(imported["added"], 1)
        self.assertEqual(next(iter(imported["library"]["items"])), item["inventory_item_id"])

    def test_import_rejects_version_mapping_and_forged_evidence_atomically(self):
        item = self.measured(self.create())
        exported = self.library.export()
        before = self.library.path.read_bytes()
        bad = copy.deepcopy(exported)
        bad["schema_version"] = 9
        with self.assertRaises(ValueError):
            self.library.import_data(bad)
        bad = copy.deepcopy(exported)
        bad["items"][item["inventory_item_id"]]["specifications"]["width"]["evidence"][0][
            "metric_value"
        ] = 100
        with self.assertRaises(ValueError):
            self.library.import_data(bad)
        self.assertEqual(self.library.path.read_bytes(), before)

    def test_private_owned_photo_upload_whitelist_and_hash(self):
        item = self.create()
        raw = io.BytesIO()
        Image.new("RGB", (64, 48), "green").save(raw, "PNG")
        with self.assertRaises(ValueError):
            self.library.upload_photo(item["inventory_item_id"], raw.getvalue(), False)
        uploaded = self.library.upload_photo(item["inventory_item_id"], raw.getvalue(), True)[
            "item"
        ]
        asset = uploaded["owned_photos"][0]
        path = self.library.asset(item["inventory_item_id"], asset["id"])
        self.assertEqual(path.read_bytes(), raw.getvalue())
        with self.assertRaises(ValueError):
            self.library.asset(item["inventory_item_id"], "../../etc/passwd")
        self.assertFalse(self.library.export()["items"][item["inventory_item_id"]]["owned_photos"])

    def test_archive_restore_retains_evidence(self):
        item = self.measured(self.create())
        archived = self.action(item, "edit", archived=True)["item"]
        self.assertTrue(archived["archived"])
        self.assertEqual(archived["specifications"], item["specifications"])
        self.assertFalse(self.action(item, "edit", archived=False)["item"]["archived"])

    def test_project_zip_retains_inventory_entity_id_and_explicit_relink(self):
        self.room()
        item = self.measured(self.create())
        placed = self.action(
            item, "place", project_id=self.pid, acknowledge_approximate_position=True
        )
        scene_id = placed["project"]["room"]["furniture"][0]["id"]
        imported = self.projects.import_bundle(self.projects.export(self.pid))
        self.assertNotEqual(imported["id"], self.pid)
        self.assertEqual(
            imported["room"]["furniture"][0]["inventory_item_id"], item["inventory_item_id"]
        )
        result = self.action(item, "link", project_id=imported["id"], scene_object_id=scene_id)
        self.assertTrue(
            any(link["project_id"] == imported["id"] for link in result["item"]["links"])
        )
        other = self.create()
        with self.assertRaises(ValueError):
            self.action(other, "link", project_id=imported["id"], scene_object_id=scene_id)
        with self.assertRaises(ValueError):
            self.action(other, "link", project_id=imported["id"])


if __name__ == "__main__":
    unittest.main()
