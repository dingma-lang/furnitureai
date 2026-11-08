"""Report executed integrated replacement and unchanged prior-regression evidence."""

import json
import re
from pathlib import Path

from furniture_bench.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase6h-replacement-integration"


def main():
    def count(name):
        value = (OUT / name).read_text()
        assert "\nOK\n" in value, name
        return int(re.search(r"Ran (\d+) tests", value)[1])

    baseline = count("baseline-tests.log")
    final = count("final-python-tests.log")
    new = count("replacement-tests.log")
    previous = {}
    for file in sorted(OUT.glob("*-browser/*tests.json")):
        value = json.loads(file.read_text())
        assert not value.get("page_errors"), str(file)
        previous[str(file.relative_to(OUT))] = {
            "passed": value.get("passed", len(value["checks"])),
            "page_errors": [],
        }
    assert sum(v["passed"] for v in previous.values()) == 115
    status = json.loads((OUT / "browser-regression-status.json").read_text())
    assert len(status) == 12 and all(s["exit_code"] == 0 for s in status)
    workflow = json.loads((OUT / "browser-tests.json").read_text())
    integrity = json.loads((OUT / "integrity-browser-tests.json").read_text())
    assert (
        workflow["passed"] == 12
        and integrity["passed"] == 2
        and not workflow["page_errors"]
        and not integrity["page_errors"]
    )
    manifest = json.loads(
        (ROOT / "reports/phase6e-product-inventory/preservation-before.json").read_text()
    )
    changes = [p for p, h in manifest.items() if digest(ROOT / p) != h]
    assert not changes, changes
    for name in [
        "replacement-workflow.png",
        "replacement-preview.png",
        "replacement-applied-2d.png",
        "replacement-applied-3d.png",
        "replacement-cost-comparison.png",
        "replacement-workflow.webm",
        "replacement-destination-demo.zip",
    ]:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 1000, name
    result = {
        "python_baseline": baseline,
        "python_final": final,
        "new_replacement_tests": new,
        "python_log": str(OUT / "final-python-tests.log"),
        "existing_browser_suites": previous,
        "existing_browser_total": 115,
        "replacement_workflow": workflow,
        "integrity_browser": integrity,
        "browser_total": 129,
        "page_errors": [],
        "source_observations_and_product_decisions_unchanged": True,
        "owned_inventory_unchanged_during_workflow": True,
        "production_retrieval_changed": False,
        "experimental_catalog_deployed": False,
        "models_downloaded": 0,
        "training": False,
        "new_inference_required_for_replacement_demo": False,
        "actual_saved_retrieval_candidates_used": True,
        "mock_inference_used_as_browser_evidence": False,
        "physical_placement_verified": False,
        "actual_purchases_or_sales_processed": False,
        "illustrative_dimensions_m": [1.6, 0.7, 0.8],
        "illustrative_gross_cost_cents": 86500,
        "illustrative_expected_resale_net_cents": 14000,
        "illustrative_expected_net_cost_cents": 72500,
        "failure_recovery_tests": [
            "Write exception preserves last committed scene/proposal/budget and permits later retry",
            "Orphan partial temporary file is ignored and replaced by next successful atomic write",
            "Pre-commit result-evaluation failure leaves saved project unchanged",
        ],
        "durability_limit": "No operating-system power-loss/fsync or multi-writer crash test",
        "preserved_files": manifest,
        "preservation_changes": changes,
        "startup_url": "http://127.0.0.1:8774",
        "limitations": [
            "Geometry, measurements and placement are unverified; placeholders are simplified",
            "Saved production candidates are possible matches, not exact household identity",
            "No live prices, transaction, resale prediction, optimization or marketplace",
            "One proposal per original placement per layout; edit/remove it before making another",
            "Undo/revert geometry does not cancel financial assumptions automatically",
            "Multiple matching purchases require explicit linking; no silent financial reassignment",
            "History is bounded; original snapshot is retained separately for revert",
            "Single-owner localhost and one active server writer",
        ],
        "manual_user_approval_required": False,
    }
    (OUT / "verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"python": final, "browser": 129, "new_tests": new, "physical_fit_verified": False}
        )
    )


if __name__ == "__main__":
    main()
