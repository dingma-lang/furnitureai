"""Summarize executed Phase 6F evidence, without substituting fixtures for physical accuracy."""

import json
import re
from pathlib import Path

from furniture_bench.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase6f-home-transfer"


def main():
    baseline = (OUT / "baseline-tests.log").read_text()
    final = (OUT / "regression-tests.log").read_text()
    assert "\nOK\n" in baseline and "\nOK\n" in final
    baseline_count = int(re.search(r"Ran (\d+) tests", baseline)[1])
    final_count = int(re.search(r"Ran (\d+) tests", final)[1])
    browser = {}
    for path in sorted(OUT.glob("*-browser/*tests.json")):
        result = json.loads(path.read_text())
        count = result.get("passed", len(result.get("checks", [])))
        assert not result.get("page_errors"), str(path)
        browser[str(path.relative_to(OUT))] = {"passed": count, "page_errors": []}
    old_count = sum(v["passed"] for v in browser.values())
    assert old_count == 86, old_count
    workflow = json.loads((OUT / "browser-tests.json").read_text())
    assert workflow["passed"] == 15 and not workflow["page_errors"]
    assert workflow["inventory_before_sha256"] == workflow["inventory_after_sha256"]
    previous = json.loads(
        (ROOT / "reports/phase6e-product-inventory/preservation-before.json").read_text()
    )
    differences = [name for name, sha in previous.items() if digest(ROOT / name) != sha]
    assert not differences, differences
    # These are genuine model caches exercised by existing end-to-end regressions.
    models = json.loads((OUT / "inventory-browser/actual-model-results.json").read_text())
    for name in [
        "floorplan-import-demo.png",
        "home-transfer-workflow.png",
        "layout-comparison.png",
        "destination-3d-view.png",
        "home-transfer-workflow.webm",
        "destination-demo.zip",
    ]:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 1000, name
    result = {
        "python_baseline": baseline_count,
        "python_final": final_count,
        "new_python_tests": final_count - baseline_count,
        "python_log": str(OUT / "regression-tests.log"),
        "previous_browser_suites": browser,
        "previous_browser_total": old_count,
        "home_transfer_workflow": workflow,
        "browser_total": old_count + workflow["passed"],
        "page_errors": [],
        "test_source": json.loads((OUT / "fixture-provenance.json").read_text()),
        "real_existing_model_regression": {"mock_inference": False, "model_stage_results": models},
        "physical_placement_verified": False,
        "measured_real_apartment_geometry_tested": False,
        "original_inventory_unchanged_during_transfer": True,
        "production_files_and_prior_results_preserved": previous,
        "preservation_differences": differences,
        "models_downloaded": 0,
        "training": False,
        "startup_url": "http://127.0.0.1:8772",
        "limitations": [
            "Generated known test geometry, not actual apartment measurements",
            "One selected room displayed in 3D at a time",
            "Architecture editing uses explicit polygon/feature JSON and simple tracing, not automatic recognition",
            "Unknown heights and thickness use display-only proportions",
            "Door swing is schematic; clearance reserve is conservative",
            "No independently verified physical fit or regulatory/transport certification",
            "Single-owner localhost; no multi-tab conflict resolution",
            "PDF rasterizer currently requires macOS Swift/PDFKit and optional PyPDF2",
        ],
        "resolved_test_failures": [
            "Inventory test initially raced the asynchronous initial library load; fixed its wait, preserved previous items",
            "3D test initially clicked an occluded table in a deliberate overlap; changed the pointer target to the visible sofa",
            "Dataset-review fixture server was stopped; resumed only remaining suites after starting bounded loopback fixture server",
            "Unit fixture used exact equality for a converted binary float; corrected the assertion to approximate equality",
        ],
        "approval_notes": [
            "Local test/browser servers and browser execution used automatic sandbox approval",
            "No user approval, downloads or paid services were needed",
        ],
    }
    (OUT / "verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "python": final_count,
                "browser": result["browser_total"],
                "physical_placement_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
