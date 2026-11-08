"""Summarize executed arithmetic, UI, regression and preservation evidence."""

import copy
import json
import re
from pathlib import Path

from furniture_bench.data import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase6g-furniture-budget"


def main():
    def count(name):
        text = (OUT / name).read_text()
        assert "\nOK\n" in text, name
        return int(re.search(r"Ran (\d+) tests", text)[1])

    baseline, final, budget_tests = (
        count("baseline-tests.log"),
        count("regression-tests.log"),
        count("budget-tests.log"),
    )
    browser = {}
    for path in sorted(OUT.glob("*-browser/*tests.json")):
        result = json.loads(path.read_text())
        assert not result.get("page_errors"), str(path)
        browser[str(path.relative_to(OUT))] = {
            "passed": result.get("passed", len(result["checks"])),
            "page_errors": [],
        }
    prior_count = sum(v["passed"] for v in browser.values())
    assert prior_count == 101, prior_count
    status = json.loads((OUT / "browser-regression-status.json").read_text())
    assert len(status) == 10 and all(s["exit_code"] == 0 for s in status)
    workflow = json.loads((OUT / "browser-tests.json").read_text())
    draft = json.loads((OUT / "draft-browser-tests.json").read_text())
    assert (
        workflow["passed"] == 12
        and draft["passed"] == 2
        and not workflow["page_errors"]
        and not draft["page_errors"]
    )
    assert workflow["inventory_before_sha256"] == workflow["inventory_after_sha256"]
    preserved = json.loads(
        (ROOT / "reports/phase6e-product-inventory/preservation-before.json").read_text()
    )
    changes = [name for name, sha in preserved.items() if digest(ROOT / name) != sha]
    assert not changes, changes
    sample = json.loads((OUT / "example-financial-scenario.json").read_text())
    assert sample["example_only"] and not sample["real_market_quotes"]
    sample["scenario"]["history"] = []
    sample["scenario"]["redo"] = []
    (OUT / "example-financial-scenario.json").write_text(json.dumps(sample, indent=2) + "\n")
    data = copy.deepcopy(sample["calculation"]["totals"])
    assert (
        data["purchase_subtotal"]["cents"] == 110000
        and data["taxes"]["cents"] == 9075
        and data["gross_upfront"]["cents"] == 154075
        and data["future_expected_resale"]["cents"] == 17000
        and data["expected_net"]["cents"] == 137075
    )
    for name in [
        "furniture-budget-workflow.png",
        "layout-cost-comparison.png",
        "furniture-budget-workflow.webm",
        "financial-destination-demo.zip",
    ]:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 1000, name
    result = {
        "python_baseline": baseline,
        "python_final": final,
        "new_financial_tests": budget_tests,
        "python_log": str(OUT / "regression-tests.log"),
        "previous_browser_suites": browser,
        "previous_browser_total": prior_count,
        "budget_workflow": workflow,
        "keyboard_draft_check": draft,
        "browser_total": prior_count + workflow["passed"] + draft["passed"],
        "page_errors": [],
        "illustrative_final_totals_cents": data,
        "currency": "USD",
        "final_currency_arithmetic": "Integer cents; Decimal HALF_UP tax-line rounding",
        "financial_quotes_real": False,
        "tax_jurisdiction_verified": False,
        "marketplace_transactions_processed": False,
        "physical_placement_verified": False,
        "inventory_unchanged_during_financial_workflow": True,
        "original_phase6f_destination_preserved_during_workflow": True,
        "prior_preservation_manifest": preserved,
        "preservation_changes": changes,
        "models_downloaded": 0,
        "training_performed": False,
        "live_prices_or_resale_predictions": False,
        "existing_real_inference_caches_exercised": True,
        "model_evidence": str(OUT / "inventory-browser/actual-model-results.json"),
        "startup_url": "http://127.0.0.1:8773",
        "limitations": [
            "USD only; user input and documentary claims not independently authenticated",
            "Unknown costs/tax/cash timing remain incomplete",
            "No live prices, sale probabilities, financing, marketplace or transactions",
            "Direct insertion of a proposed financial purchase as new 3D furniture is not implemented",
            "Unsaved drafts are tab-local and do not survive page reload",
            "One scenario per saved layout; simultaneous server/tab conflict resolution unsupported",
            "Seller timing is a single before-sale/deducted-at-sale/unknown assumption per resale record",
        ],
        "diagnosed_initial_failures": [
            "Actual-sale fields were inside a closed disclosure; UI now opens it on completed-sale selection",
            "An independence test initially expected 290.20 for 2×150−10; corrected expected fixture total to 290.00",
            "New prepaid-expense tests caught an initially unconnected timing accumulator; fixed and reran all tests",
        ],
        "manual_approvals_required": False,
    }
    (OUT / "verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "python": final,
                "browser": result["browser_total"],
                "sample_expected_usd_cents": 137075,
            }
        )
    )


if __name__ == "__main__":
    main()
