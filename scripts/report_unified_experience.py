"""Summarize actual unified-journey, regression, performance and provenance evidence."""

import json
import re
from pathlib import Path

from furniture_bench.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase6i-unified-experience"


def main():
    def count(name):
        text = (OUT / name).read_text()
        assert "\nOK\n" in text, name
        return int(re.search(r"Ran (\d+) tests", text)[1])

    baseline, final, workflow_tests = (
        count("baseline-tests.log"),
        count("final-python-tests.log"),
        count("workflow-tests.log"),
    )
    prior = {}
    for file in sorted(OUT.glob("*-browser/*tests.json")):
        value = json.loads(file.read_text())
        assert not value.get("page_errors"), str(file)
        prior[str(file.relative_to(OUT))] = {
            "passed": value.get("passed", len(value["checks"])),
            "page_errors": [],
        }
    old = sum(v["passed"] for v in prior.values())
    assert old == 129, old
    status = json.loads((OUT / "browser-regression-status.json").read_text())
    assert len(status) == 14 and all(r["exit_code"] == 0 for r in status)
    workflow = json.loads((OUT / "end-to-end-browser-tests.json").read_text())
    details = json.loads((OUT / "details-browser-tests.json").read_text())
    recovery = json.loads((OUT / "recovery-browser-tests.json").read_text())
    assert workflow["passed"] == 8 and details["passed"] == 7 and recovery["passed"] == 5
    assert not any(r["page_errors"] for r in (workflow, details, recovery))
    assert (
        workflow["all_workflow_mutations_via_browser_ui"]
        and not workflow["background_requests_started_by_dashboard"]
    )
    analysis = json.loads((OUT / "actual-analysis.json").read_text())
    assert analysis["mocked"] is False and all(
        s["state"] == "complete" for s in analysis["stages"].values()
    )
    assert all(s["cached"] for s in analysis["stages"].values())
    manifest = json.loads(
        (ROOT / "reports/phase6e-product-inventory/preservation-before.json").read_text()
    )
    changed = [p for p, h in manifest.items() if digest(ROOT / p) != h]
    assert not changed, changed
    required = [
        "ux-audit.md",
        "workflow-architecture.md",
        "unified-dashboard.png",
        "photo-review-to-inventory.png",
        "inventory-to-move.png",
        "integrated-replacement-workflow.png",
        "simplified-budget.png",
        "end-to-end-workflow.webm",
    ]
    for name in required:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 200, name
    result = {
        "python_baseline": baseline,
        "python_final": final,
        "new_workflow_unit_tests": workflow_tests,
        "python_log": str(OUT / "final-python-tests.log"),
        "prior_browser_checks": prior,
        "prior_browser_total": old,
        "continuous_end_to_end_workflow": workflow,
        "focused_navigation_and_review_checks": details,
        "focused_recovery_and_draft_checks": recovery,
        "browser_total": old + workflow["passed"] + details["passed"] + recovery["passed"],
        "page_errors": [],
        "actual_analysis": analysis,
        "performance": {
            "timings": workflow["timings_seconds"],
            "dashboard_summary_fetch_ms": workflow["dashboard_fetch_ms"],
            "conditions": "Single local headless Chrome/SwiftShader run; genuine hash-matched CPU analysis caches, not fresh model inference latency",
            "before_after_speed_improvement_claimed": False,
            "model_initializations_required_for_dashboard": False,
            "windows_gpu_dependency": False,
        },
        "safety": {
            "physical_placement_verified": False,
            "illustrative_replacement_dimensions_and_prices": True,
            "exact_product_identity_verified": False,
            "new_owned_id_explicitly_created_by_user": True,
            "hypothetical_replacement_is_not_owned_inventory": True,
            "unknown_dimensions_supported": True,
            "optional_second_photo_supported": True,
            "missing_assets_simulated_without_private_file_deletion": True,
            "new_models_downloaded": 0,
            "neural_training": False,
            "paid_apis": False,
            "cloud_or_account_added": False,
        },
        "compatibility": {
            "destructive_migration": False,
            "same_inventory_destination_layout_proposal_purchase_ids_reused": True,
            "production_matching_changed": False,
            "experimental_catalog_deployed": False,
            "preserved_manifest": manifest,
            "preservation_differences": changed,
            "legacy_assertions_weakened": False,
            "advanced_geometry_test_navigation_updated": True,
        },
        "diagnosed_failures": [
            "Unified navigation clicks during project refresh were initially dropped; actions now queue in order",
            "Save/reload does not increment project revision; corrected new test to check saved reload completion",
            "Concurrent mutating focused/legacy tests invalidated the global inventory equality assertion; remaining legacy suites rerun sequentially with assertions intact",
            "Unavailable-image test initially reused an already-loaded image; now requests a fresh page with a simulated 404",
            "Architecture fixture retries overlapped previous doors; focused tests now use an explicit exported/imported copy and new room",
        ],
        "known_limitations": [
            "Additive navigation, not a full production router; legacy workspace remains below Home",
            "Advanced nonrectangular geometry and some source edits still use explicit forms/JSON",
            "Simple budget is a presentation of saved authoritative calculations; editing advanced assumptions requires Advanced view",
            "Fragment linking preserves separate boxes; no automatic whole-object mask/3D merge",
            "Unsaved browser drafts are not durable and some standalone advanced controls retain explicit-save behavior",
            "No human usability study, mobile-touch acceptance or screen-reader certification",
            "One active localhost writer; concurrent tabs/server conflict resolution unsupported",
            "Software demonstration dimensions/prices are not real room/manufacturer/market evidence",
        ],
        "startup_url": "http://127.0.0.1:8775",
        "manual_user_testing_required_for_this_delivery": False,
    }
    (OUT / "verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "python": final,
                "browser": result["browser_total"],
                "new_browser_checks": 20,
                "analysis_cached": True,
            }
        )
    )


if __name__ == "__main__":
    main()
