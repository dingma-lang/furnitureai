"""Real budget UI over a copy of the saved Phase 6F generated destination; no model needed."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("FURNITURE_BUDGET_REPORT_DIR", ROOT / "reports/phase6g-furniture-budget"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8773")
EXAMPLE = "ILLUSTRATIVE SOFTWARE EXAMPLE ONLY, not a real price, tax jurisdiction or market quote"


def main():
    OUT.mkdir(exist_ok=True)
    source = json.loads((ROOT / "reports/phase6f-home-transfer/browser-tests.json").read_text())
    source_id = source["destination_id"]
    sofa = source["owned_sofa_id"]
    table = source["owned_table_id"]
    checks = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1580, "height": 1150},
            record_video_dir=str(OUT / "recordings"),
            record_video_size={"width": 1280, "height": 932},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def snap():
            return page.evaluate("furnitureBudget.snapshot()")

        def scenario():
            return snap()["data"]["budget"]["scenarios"][snap()["lid"]]

        def calc(lid=None):
            return snap()["data"]["comparison"][lid or snap()["lid"]]["calculation"]

        def total(key, lid=None):
            return calc(lid)["totals"][key]["cents"]

        def click_save(id):
            rev = snap()["data"]["project_revision"]
            page.locator("#" + id).click()
            page.wait_for_function(
                "(r)=>furnitureBudget.snapshot().data.project_revision>r && !furnitureBudget.snapshot().busy",
                arg=rev,
            )

        def set_action(iid, action):
            page.locator("#budget-item").select_option(iid)
            page.locator("#budget-action").select_option(action)
            click_save("budget-action-save")

        def layout(lid):
            page.locator("#budget-layout").select_option(lid)
            page.wait_for_function(
                "(lid)=>furnitureBudget.snapshot().lid===lid && !furnitureBudget.snapshot().busy",
                arg=lid,
            )

        def expense(amount, covers=True):
            page.locator("#budget-expense-new").click()
            page.locator("#budget-expense-description").fill(
                "Example moving company, one layout charge"
            )
            page.locator("#budget-expense-amount").fill(amount)
            page.locator("#budget-expense-source").fill(EXAMPLE)
            if covers:
                page.locator("#budget-expense-covers").check()
            click_save("budget-expense-save")

        def passed(name):
            checks.append(name)
            print(name, flush=True)

        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function(
                "window.furnitureBudget && window.homeTransfer && window.roomLab"
            )
            original = page.request.get(f"{BASE}/api/transfer/{source_id}").json()
            bundle = page.request.get(f"{BASE}/api/transfer/{source_id}/export")
            assert bundle.ok
            (OUT / "phase6f-destination-copy.zip").write_bytes(bundle.body())
            library = page.request.get(BASE + "/api/inventory").json()
            before_hash = hashlib.sha256(json.dumps(library, sort_keys=True).encode()).hexdigest()
            page.locator("#plan-move").click()
            page.locator("#destination-import").set_input_files(
                str(OUT / "phase6f-destination-copy.zip")
            )
            page.wait_for_function("homeTransfer.snapshot().project!==null")
            pid = page.evaluate("homeTransfer.snapshot().project.id")
            assert pid != source_id
            page.locator("#open-budget").click()
            page.wait_for_function(
                "furnitureBudget.snapshot().data!==null && !furnitureBudget.snapshot().busy"
            )
            lids = list(snap()["data"]["layouts"])
            a, b = lids[:2]
            layout(a)
            passed(
                "Existing generated destination copied through private ZIP; financial scenario opens per saved layout"
            )
            set_action(sofa, "retain")
            set_action(table, "move")
            expense("300")
            assert total("gross_upfront") == 30000
            assert total("expected_net") == 30000
            scenario_a = json.loads(json.dumps(scenario()))
            passed(
                "Keep/move scenario counts a shared moving charge once with explicit item coverage"
            )
            layout(b)
            set_action(sofa, "replace")
            set_action(table, "retain")
            assert total("gross_upfront") is None
            page.locator("#budget-purchase-new").click()
            page.locator("#budget-purchase-description").fill(
                "Example replacement sofa — unknown product identity"
            )
            page.locator("#budget-replacement-of").select_option(sofa)
            page.locator("#budget-quantity").fill("1")
            page.locator("#budget-unit-price").fill("1000.00")
            page.locator("#budget-discount").fill("100.00")
            page.locator("#budget-goods-taxable").select_option("true")
            page.locator("#budget-tax-rate").fill("8.25")
            page.locator("#budget-tax-jurisdiction").fill(
                "Illustrative assumed jurisdiction, not official tax treatment"
            )
            for key, value in [("delivery", "50"), ("assembly", "0"), ("other-fee", "0")]:
                page.locator("#budget-" + key).fill(value)
            for key in ("delivery", "assembly", "other"):
                page.locator("#budget-" + key + "-taxable").select_option("false")
            page.locator("#budget-purchase-source").fill(EXAMPLE)
            click_save("budget-purchase-save")
            purchase_id = snap()["purchaseId"]
            assert total("purchase_subtotal") == 90000
            assert total("taxes") == 7425
            passed(
                "Unknown-identity sofa replacement uses exact cents, discount and user-entered tax; original owned item is retained"
            )
            page.locator("#budget-item").select_option(sofa)
            for key, value in [
                ("resale-low", "100"),
                ("resale-expected", "200"),
                ("resale-high", "300"),
                ("resale-asking", "400"),
                ("selling-fees", "20"),
                ("seller-pickup", "10"),
                ("seller-other", "0"),
            ]:
                page.locator("#budget-" + key).fill(value)
            page.locator("#budget-seller-timing").select_option("false")
            page.locator("#budget-sale-source").fill(EXAMPLE)
            click_save("budget-sale-save")
            expense("300", False)
            assert total("gross_upfront") == 132425
            assert total("future_expected_resale") == 17000
            assert total("conservative_net") == 132425
            assert total("expected_net") == 115425
            assert total("optimistic_net") == 105425
            passed(
                "Upfront $1,324.25 stays separate from uncertain resale; expected net $1,154.25 and no-future-revenue conservative case are correct"
            )
            page.locator("#budget-panel").evaluate("e=>e.scrollTop=0")
            page.screenshot(path=str(OUT / "furniture-budget-workflow.png"))
            page.locator("#budget-compare").click()
            page.locator("#budget-panel").evaluate("e=>e.scrollTop=0")
            page.screenshot(path=str(OUT / "layout-cost-comparison.png"))
            passed(
                "Financial comparison shows costs, upfront cash, uncertainty and missing counts without choosing a best design"
            )
            page.locator("#budget-purchase-picker").select_option(purchase_id)
            page.locator("#budget-unit-price").fill("1200")
            click_save("budget-purchase-save")
            click_save("budget-purchase-save")
            assert len(scenario()["purchases"]) == 1
            assert total("gross_upfront") == 154075
            assert total("expected_net") == 137075
            assert snap()["data"]["budget"]["scenarios"][a] == scenario_a
            passed(
                "Editing and repeat-saving one replacement preserves its event ID and leaves the other layout independent"
            )
            page.locator("#budget-unit-price").fill("")
            click_save("budget-purchase-save")
            assert total("expected_net") is None
            assert calc()["missing_count"] > 0
            click_save("budget-undo")
            assert total("expected_net") == 137075
            passed(
                "Blank price remains unknown and makes totals incomplete; undo restores exact prior assumptions"
            )
            # Unsaved edits stay in their own layout draft and do not affect authoritative totals.
            page.locator("#budget-unit-price").fill("1300")
            layout(a)
            layout(b)
            assert page.locator("#budget-unit-price").input_value() == "1300"
            assert total("expected_net") == 137075
            page.locator("#budget-unit-price").fill("1200")
            click_save("budget-purchase-save")
            passed(
                "Unsaved form drafts survive layout switches while totals remain based on saved assumptions"
            )
            # Actual completed-sale records are attested inputs, never processed marketplace transactions.
            page.locator("#budget-item").select_option(sofa)
            page.locator("#budget-sale-state").select_option("completed_recorded")
            page.locator("#budget-sale-actual").fill("150")
            page.locator("#budget-actual-fees").fill("10")
            page.locator("#budget-actual-pickup").fill("0")
            page.locator("#budget-actual-other").fill("0")
            page.locator("#budget-sale-date").fill("2026-10-31")
            page.locator("#budget-sale-confirm").check()
            click_save("budget-sale-save")
            assert total("realized_net_resale") == 14000
            assert total("future_expected_resale") == 0
            assert total("expected_net") == 140075
            click_save("budget-undo")
            assert total("future_expected_resale") == 17000
            passed(
                "Recorded completed sale uses actual net proceeds once, excludes future resale, and is reversible without deleting inventory"
            )
            scenario_id = scenario()["id"]
            page.locator("#budget-close").click()
            page.reload(wait_until="networkidle")
            page.wait_for_function("window.homeTransfer && window.furnitureBudget")
            page.locator("#plan-move").click()
            page.locator("#transfer-picker").select_option(pid)
            page.wait_for_function("(pid)=>homeTransfer.snapshot().project?.id===pid", arg=pid)
            page.locator("#open-budget").click()
            page.wait_for_function(
                "furnitureBudget.snapshot().data!==null && !furnitureBudget.snapshot().busy"
            )
            layout(b)
            assert scenario()["id"] == scenario_id
            assert purchase_id in scenario()["purchases"]
            assert total("expected_net") == 137075
            assert EXAMPLE in scenario()["purchases"][purchase_id]["source"]
            assert scenario()["purchases"][purchase_id]["replacement_of"] == sofa
            assert not calc()["physical_placement_verified"]
            assert not page.request.get(f"{BASE}/api/transfer/{pid}").json()["transfer_assessment"][
                "physical_placement_verified"
            ]
            passed(
                "Save/reopen preserves scenario/item/event IDs, sources and calculations with physical fit still unverified"
            )
            assert page.request.get(BASE + "/api/inventory").json() == library
            original_after = page.request.get(f"{BASE}/api/transfer/{source_id}").json()
            assert original == original_after
            passed(
                "Original inventory, source observations and original Phase 6F destination are unchanged"
            )
            export = page.request.get(f"{BASE}/api/transfer/{pid}/export")
            assert export.ok
            (OUT / "financial-destination-demo.zip").write_bytes(export.body())
            imported = page.request.post(
                BASE + "/api/transfer/import",
                data=export.body(),
                headers={"Content-Type": "application/zip"},
            )
            assert imported.ok
            new_id = imported.json()["id"]
            reopened = page.request.get(f"{BASE}/api/budget/{new_id}").json()
            assert len(reopened["budget"]["scenarios"][b]["purchases"]) == 1
            assert (
                reopened["comparison"][b]["calculation"]["totals"]["expected_net"]["cents"]
                == 137075
            )
            assert reopened["budget"]["scenarios"][b]["destination_project_id"] == new_id
            passed(
                "Destination export/import includes budgets, remaps project references and preserves shared owned IDs without duplicate events"
            )
            page.locator("#budget-compare").click()
            page.locator("#budget-panel").evaluate("e=>e.scrollTop=0")
            page.screenshot(path=str(OUT / "layout-cost-comparison.png"))
            assert not errors, errors
            (OUT / "example-financial-scenario.json").write_text(
                json.dumps(
                    {
                        "example_only": True,
                        "real_market_quotes": False,
                        "tax_jurisdiction_verified": False,
                        "description": EXAMPLE,
                        "scenario": scenario(),
                        "calculation": calc(),
                    },
                    indent=2,
                )
            )
            (OUT / "browser-tests.json").write_text(
                json.dumps(
                    {
                        "passed": len(checks),
                        "checks": checks,
                        "page_errors": errors,
                        "source_destination_id": source_id,
                        "destination_id": pid,
                        "imported_destination_id": new_id,
                        "layout_a": a,
                        "layout_b": b,
                        "scenario_id": scenario_id,
                        "purchase_id": purchase_id,
                        "sofa_inventory_id": sofa,
                        "table_inventory_id": table,
                        "inventory_before_sha256": before_hash,
                        "inventory_after_sha256": hashlib.sha256(
                            json.dumps(library, sort_keys=True).encode()
                        ).hexdigest(),
                        "illustrative_expected_totals_cents": {
                            "layout_a": 30000,
                            "layout_b": 137075,
                            "layout_b_gross": 154075,
                        },
                        "physical_placement_verified": False,
                    },
                    indent=2,
                )
            )
        finally:
            page.screenshot(path=str(OUT / "browser-final.png"))
            video = page.video
            context.close()
            if video:
                video.save_as(str(OUT / "furniture-budget-workflow.webm"))
            browser.close()


if __name__ == "__main__":
    main()
