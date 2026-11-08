import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from furniture_bench.data import digest
from furniture_bench.dataset_quality import (
    apply_reviews,
    category_condition,
    fingerprint,
    main,
    near_duplicate,
    validate_manifest,
)


class DatasetQualityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = []
        for n in range(3):
            path = self.root / f"image-{n}.png"
            Image.new("RGB", (30, 20), (40 + n * 45, 30, 70)).save(path)
            self.rows.append(
                {
                    "image_id": f"image-{n}",
                    "product_id": "listing-A",
                    "identity_namespace": "ABO_listing",
                    "file": path.name,
                    "sha256": digest(path),
                    "dimensions": [30, 20],
                    "role": "catalog_reference" if n == 0 else "catalog_query",
                    "split": "development",
                    "manufacturer_model_verified": False,
                    "rights": {
                        "source": "test fixture, not benchmark imagery",
                        "source_url": "https://example.test/fixture",
                        "license": "fixture",
                        "permission_basis": "test_fixture",
                        "allowed_for_evaluation": True,
                    },
                    "quality_annotation": {
                        "label": "ambiguous",
                        "status": "provisional",
                        "provenance": "unit-test fixture",
                    },
                }
            )
        self.manifest = {"schema_version": 1, "images": self.rows}

    def patch(self):
        return {
            "manifest_signature": hashlib.sha256(
                json.dumps(self.manifest, sort_keys=True).encode()
            ).hexdigest(),
            "reviews": [
                {
                    "image_id": "image-1",
                    "sha256": self.rows[1]["sha256"],
                    "label": "whole_object",
                    "reviewer": "UNIT TEST fixture",
                    "attested_human_review": True,
                    "whole_product_visible": "yes",
                    "suitable_query": "yes",
                    "suitable_gallery": "unknown",
                }
            ],
        }

    def test_valid_manifest_aligns_hashes_and_dimensions(self):
        self.assertEqual(validate_manifest(self.manifest, self.root)["images"], 3)

    def test_cli_exports_only_to_new_path(self):
        source = self.root / "manifest.json"
        source.write_text(json.dumps(self.manifest))
        output = self.root / "validated.json"
        args = ["dataset_quality", str(source), "--root", str(self.root), "--output", str(output)]
        with patch("sys.argv", args):
            main()
        self.assertEqual(json.loads(output.read_text()), self.manifest)
        before = output.read_bytes()
        with patch("sys.argv", args), self.assertRaises(ValueError):
            main()
        self.assertEqual(output.read_bytes(), before)

    def test_missing_permission_and_license_rejected(self):
        for key in ["license", "permission_basis", "source_url", "allowed_for_evaluation"]:
            value = copy.deepcopy(self.manifest)
            del value["images"][0]["rights"][key]
            with self.assertRaises(ValueError):
                validate_manifest(value, self.root)

    def test_owner_permission_requires_document(self):
        self.rows[0]["rights"]["permission_basis"] = "owner_permission"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_duplicate_ids_hashes_and_path_escape_rejected(self):
        for field, value in [
            ("image_id", "image-0"),
            ("file", "../escape.png"),
            ("sha256", self.rows[0]["sha256"]),
        ]:
            manifest = copy.deepcopy(self.manifest)
            manifest["images"][1][field] = value
            with self.assertRaises(ValueError):
                validate_manifest(manifest, self.root)

    def test_query_identity_must_be_in_gallery_and_unknown_absent(self):
        self.rows[1]["product_id"] = "missing"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)
        self.rows[1]["role"] = "unknown_query"
        validate_manifest(self.manifest, self.root)
        self.rows[1]["product_id"] = "listing-A"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_household_without_identity_is_pending_not_verified(self):
        self.rows[1]["role"] = "household_query"
        report = validate_manifest(self.manifest, self.root)
        self.assertEqual(report["warnings"][0]["reason"], "household_identity_unverified")
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root, require_evaluable=True)

    def test_verified_household_requires_session_and_preserves_split(self):
        for row in self.rows[1:]:
            row.update(
                {
                    "role": "household_query",
                    "session_id": "session-A",
                    "identity_evidence": {
                        "status": "verified",
                        "verification_method": "fixture receipt",
                        "verified_by": "test fixture",
                        "evidence_references": ["fixture"],
                    },
                }
            )
        validate_manifest(self.manifest, self.root, True)
        self.rows[2]["split"] = "heldout"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_quality_and_human_provenance_cannot_be_implicit(self):
        self.rows[1]["quality_annotation"]["status"] = "human_reviewed"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)
        self.rows[1]["quality_annotation"] = {"label": "made_up", "provenance": "fixture"}
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_review_requires_attestation_and_correct_signature(self):
        patch = self.patch()
        patch["reviews"][0]["attested_human_review"] = False
        with self.assertRaises(ValueError):
            apply_reviews(self.manifest, patch)
        patch = self.patch()
        patch["manifest_signature"] = "wrong"
        with self.assertRaises(ValueError):
            apply_reviews(self.manifest, patch)

    def test_reviews_preserve_identity_and_original_quality(self):
        original = copy.deepcopy(self.manifest)
        reviewed = apply_reviews(self.manifest, self.patch())
        self.assertEqual(self.manifest, original)
        self.assertEqual(reviewed["images"][1]["product_id"], "listing-A")
        self.assertEqual(
            reviewed["images"][1]["quality_history"][0], original["images"][1]["quality_annotation"]
        )
        self.assertEqual(reviewed["images"][1]["quality_annotation"]["status"], "human_reviewed")

    def test_unsupported_manufacturer_verification_rejected(self):
        self.rows[1]["manufacturer_model_verified"] = True
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_synthetic_requires_source_and_transformation(self):
        self.rows[1]["role"] = "synthetic"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.root)

    def test_near_duplicate_screens_changed_exports(self):
        path = self.root / "image-0.png"
        duplicate = self.root / "copy.png"
        Image.open(path).resize((60, 40)).save(duplicate)
        self.assertTrue(near_duplicate(fingerprint(path), fingerprint(duplicate)))

    def test_confirmed_category_requires_confirmation_or_falls_back(self):
        scores = np.array([[0.8, 0.9]])
        confirmed = [{"label": "sofa"}]
        np.testing.assert_equal(
            category_condition(scores, ["sofa", "chair"], [], confirmed, "confirmed"), scores
        )
        confirmed[0].update(confirmed_by="fixture", evidence_source="user_confirmation")
        result = category_condition(scores, ["sofa", "chair"], [], confirmed, "confirmed")
        self.assertTrue(np.isneginf(result[0, 1]))
        # Wrong confirmation excludes truth; this must not be hidden as a successful match.
        confirmed[0]["label"] = "chair"
        self.assertTrue(
            np.isneginf(
                category_condition(scores, ["sofa", "chair"], [], confirmed, "confirmed")[0, 0]
            )
        )

    def test_uncertain_automatic_category_and_missing_metadata_fallback(self):
        scores = np.array([[0.8, 0.9]])
        for prediction in [
            {"label": None, "score": 0},
            {"label": "sofa", "score": 0.1},
            {"label": "missing", "score": 0.9},
        ]:
            np.testing.assert_equal(
                category_condition(scores, ["sofa", "chair"], [prediction], mode="predicted_soft"),
                scores,
            )


if __name__ == "__main__":
    unittest.main()
