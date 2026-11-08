"""Focused keyboard-close regression; edits remain unsaved browser-local drafts."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("FURNITURE_BUDGET_REPORT_DIR", ROOT / "reports/phase6g-furniture-budget"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8773")


def main():
    saved = json.loads((OUT / "browser-tests.json").read_text())
    pid = saved["destination_id"]
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function("window.homeTransfer && window.furnitureBudget")
        page.locator("#plan-move").click()
        page.locator("#transfer-picker").select_option(pid)
        page.wait_for_function("(pid)=>homeTransfer.snapshot().project?.id===pid", arg=pid)
        page.locator("#open-budget").click()
        page.wait_for_function(
            "furnitureBudget.snapshot().data!==null&&!furnitureBudget.snapshot().busy"
        )
        before = page.request.get(BASE + "/api/budget/" + pid).json()
        page.locator("#budget-purchase-picker").select_option(saved["purchase_id"])
        page.locator("#budget-unit-price").fill("1300.01")
        page.keyboard.press("Escape")
        assert not page.locator("#budget-panel").is_visible()
        page.locator("#open-budget").click()
        page.wait_for_function("document.getElementById('budget-unit-price').value==='1300.01'")
        after = page.request.get(BASE + "/api/budget/" + pid).json()
        assert before == after
        # Capture the saved illustrative scenario, not the unsaved keyboard-test draft.
        page.locator("#budget-unit-price").fill("1200.00")
        page.locator("#budget-item").select_option(saved["sofa_inventory_id"])
        page.locator("#budget-compare").click()
        page.locator("#budget-panel").evaluate("el=>el.scrollTop=0")
        page.screenshot(path=str(OUT / "layout-cost-comparison.png"))
        # Preserve the reviewed example; check contradictory decisions on an explicit design copy.
        bundle = page.request.get(BASE + "/api/transfer/" + pid + "/export")
        response = page.request.post(
            BASE + "/api/transfer/import",
            data=bundle.body(),
            headers={"Content-Type": "application/zip"},
        )
        assert response.ok
        copied = response.json()["id"]
        page.locator("#budget-close").click()
        page.locator("#transfer-close").click()
        page.locator("#plan-move").click()
        page.locator("#transfer-picker").select_option(copied)
        page.wait_for_function("(id)=>homeTransfer.snapshot().project?.id===id", arg=copied)
        page.locator("#open-budget").click()
        page.wait_for_function(
            "(id)=>furnitureBudget.snapshot().pid===id&&!furnitureBudget.snapshot().busy",
            arg=copied,
        )
        page.locator("#budget-item").select_option(saved["sofa_inventory_id"])
        page.locator("#budget-action").select_option("retain")
        revision = page.evaluate("furnitureBudget.snapshot().data.project_revision")
        page.locator("#budget-action-save").click()
        page.wait_for_function(
            "(r)=>furnitureBudget.snapshot().data.project_revision>r&&!furnitureBudget.snapshot().busy",
            arg=revision,
        )
        assert (
            page.evaluate(
                "furnitureBudget.snapshot().data.comparison[furnitureBudget.snapshot().lid].calculation.totals.expected_net.cents"
            )
            is None
        )
        page.locator("details:has(> #budget-missing) > summary").click()
        assert "retain_conflicts" in page.locator("#budget-missing").inner_text()
        assert not errors
        result = {
            "passed": 2,
            "checks": [
                "Escape/reopen preserves unsaved form draft without changing saved financial records or totals",
                "Retain with an existing explicit replacement is visibly flagged as incomplete without deleting that purchase",
            ],
            "page_errors": errors,
        }
        (OUT / "draft-browser-tests.json").write_text(json.dumps(result, indent=2))
        b.close()
        print(result)


if __name__ == "__main__":
    main()
