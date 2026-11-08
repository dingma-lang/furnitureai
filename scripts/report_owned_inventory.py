"""Summarize saved real integration, regression and preservation evidence."""

from pathlib import Path

from furniture_bench.data import digest, read_json, write_json
from room_viewer.inventory import validate_item

ROOT = Path("reports/phase6e-product-inventory")


def main():
    baseline = (ROOT / "baseline-tests.log").read_text()
    tests = (ROOT / "regression-tests.log").read_text()
    assert "Ran 243 tests" in baseline and "\nOK\n" in baseline
    assert "Ran 269 tests" in tests and "\nOK\n" in tests
    browser = {}
    for directory in ROOT.glob("*-browser"):
        result = read_json(next(directory.glob("*tests.json")))
        browser[directory.name] = {
            "passed": result.get("passed", len(result["checks"])),
            "page_errors": result.get("page_errors", []),
        }
    assert sum(r["passed"] for r in browser.values()) == 71
    integration = read_json(ROOT / "inventory-browser-tests.json")
    details = read_json(ROOT / "inventory-detail-browser-tests.json")
    assert integration["passed"] == 9 and details["passed"] == 6
    assert not any(r["page_errors"] for r in [*browser.values(), integration, details])
    prior = read_json(ROOT / "preservation-before.json")
    changed = [name for name, sha in prior.items() if digest(name) != sha]
    assert not changed, changed
    model = read_json(ROOT / "fresh-inference-model-results.json")
    assert all(
        stage["state"] == "complete" and stage["cached"] is False
        for stage in model["stages"].values()
    )
    sample = read_json(ROOT / "sample-inventory.json")
    for item in sample["items"].values():
        validate_item(item)
    result = {
        "python_baseline": 243,
        "python_final": 269,
        "python_report": str(ROOT / "regression-tests.log"),
        "previous_browser_checks": browser,
        "previous_browser_total": 71,
        "inventory_e2e_checks": integration,
        "inventory_detail_checks": details,
        "browser_total": 86,
        "page_errors": [],
        "actual_model_integration": {
            "source": "Phillip Goldsberry licensed room photo; original retained, separate1440px processing copy",
            "rights_reference": "data/detection-pilot/ATTRIBUTION.md",
            "first_pass_stages": model["stages"],
            "later_passes_reused_real_caches": True,
            "mocked_inference_used_as_integration_evidence": False,
        },
        "persisted_ids": {
            k: integration[k]
            for k in [
                "project_id",
                "second_project_id",
                "inventory_item_id",
                "observation_id",
                "scene_object_id",
                "second_scene_object_id",
            ]
        },
        "persistence": {
            "save_reopen_verified": True,
            "export_import_idempotence_verified": True,
            "cross_project_shared_item_verified": True,
            "project_zip_inventory_id_tested": True,
            "interrupted_joint_update_recovery_tested": True,
            "legacy_projects_automatically_migrated": False,
        },
        "safety": {
            "exact_household_identity_verified": False,
            "physical_placement_verified": False,
            "demonstration_dimensions_are_example_inputs": True,
            "catalog_reports_not_independently_verified": True,
            "unsupported_schema_and_unsafe_asset_paths_rejected": True,
            "private_photos_excluded_from_portable_inventory": True,
            "PDF_text_extraction_tested_on_actual_software_fixture_PDF": True,
            "manufacturer_PDF_identity_authentication_verified": False,
        },
        "preserved_files": len(prior),
        "preserved_file_hashes": prior,
        "preserved_files_changed": changed,
        "production_retrieval_changed": False,
        "experimental_catalog_deployed": False,
        "new_model_weights": 0,
        "training_performed": False,
    }
    write_json(ROOT / "verification-summary.json", result)
    print(
        {
            "python": 269,
            "browser": 86,
            "production_changed": False,
            "real_model_stages": list(model["stages"]),
        }
    )


if __name__ == "__main__":
    main()
