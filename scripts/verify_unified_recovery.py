"""Focused actual UI duplicate-name, detail draft and plan-reference recovery checks."""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = ROOT / "reports/phase6i-unified-experience"
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8775")


def main():
    saved = json.loads((OUT / "end-to-end-browser-tests.json").read_text())
    checks = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))

        def wait(expr, arg=None):
            page.wait_for_function(expr, arg=arg)

        def passed(s):
            checks.append(s)
            print(s, flush=True)

        try:
            page.goto(BASE, wait_until="networkidle")
            wait("window.unifiedExperience&&unifiedExperience.snapshot().overview!==null")
            page.locator("#home-furniture").click()
            wait("document.getElementById('furniture-cards-panel').open")
            name = page.locator(
                '[data-owned-card="' + saved["owned_inventory_id"] + '"] h3'
            ).inner_text()
            page.locator("#cards-manual").click()
            wait("window.ownedInventory&&ownedInventory.snapshot().selected!==null")
            page.wait_for_function(
                "document.getElementById('inventory-name').value==='My furniture'"
            )
            new_id = page.evaluate("ownedInventory.snapshot().selected")
            page.locator("#inventory-name").fill(name)
            page.locator("#inventory-category").select_option("sofa")
            page.locator("#inventory-notes").fill(
                "SOFTWARE FIXTURE: same-name identity-disambiguation test; not another verified household sofa"
            )
            page.locator("#inventory-save").click()
            wait(
                "(name)=>ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].display_name===name",
                name,
            )
            page.locator("#close-inventory").click()
            page.locator("#nav-furniture").click()
            wait("document.getElementById('furniture-cards-panel').open")
            page.locator("#cards-search").fill(name)
            assert page.locator("[data-owned-card]").count() >= 2
            assert new_id != saved["owned_inventory_id"]
            assert new_id[-6:] in page.locator('[data-owned-card="' + new_id + '"]').inner_text()
            passed(
                "Two identical user-facing furniture names remain distinct cards and independent owned IDs"
            )
            page.locator('[data-edit-owned="' + new_id + '"]').click()
            wait("(id)=>ownedInventory.snapshot().selected===id", new_id)
            page.locator("#inventory-notes").fill("Unsaved notes draft")
            page.locator("#close-inventory").click()
            wait("document.getElementById('unsaved-navigation').open")
            page.locator("#unsaved-stay").click()
            assert page.locator("#inventory-notes").input_value() == "Unsaved notes draft"
            page.locator("#close-inventory").click()
            page.locator("#unsaved-discard").click()
            wait("!document.getElementById('inventory-panel').open")
            passed(
                "Existing detailed inventory edits receive an explicit stay/discard guard before modal closure"
            )
            page.locator("#nav-move").click()
            wait("document.getElementById('move-handoff').open")
            page.locator("#handoff-name").fill("Unavailable-plan preview test fixture")
            page.locator("#handoff-open").click()
            wait("homeTransfer.snapshot().project!==null")
            pid = page.evaluate("homeTransfer.snapshot().project.id")
            pattern = re.compile(r"/api/transfer/.*/assets/[^/]+\.png$")
            page.route(pattern, lambda r: r.fulfill(status=404, body="Missing preview fixture"))
            page.locator("#plan-rights").check()
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#plan-upload").set_input_files(
                str(ROOT / "reports/phase6f-home-transfer/generated-test-floorplan.png")
            )
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)
            wait(
                "document.getElementById('workflow-notice').textContent.includes('Floor-plan preview unavailable')"
            )
            assert page.evaluate("homeTransfer.snapshot().project.home_transfer.floorplan!==null")
            page.unroute(pattern)
            passed(
                "Unavailable floor-plan image reports recovery guidance while retaining the saved source reference and project"
            )
            page.locator("#nav-home").click()
            wait('unifiedExperience.snapshot().route==="home"')
            page.locator("#home-search").fill("Unified moving plan")
            page.locator('[data-continue-project="' + saved["destination_id"] + '"]').click()
            wait("(id)=>homeTransfer.snapshot().project?.id===id", saved["destination_id"])
            page.locator("#open-replacement").click()
            wait("roomReplacements.snapshot().project!==null&&!roomReplacements.snapshot().busy")
            page.locator("#replacement-name").fill("Saved after protected replacement draft test")
            page.locator("#replacement-close").click()
            wait("document.getElementById('unsaved-navigation').open")
            page.locator("#unsaved-stay").click()
            assert (
                page.locator("#replacement-name").input_value()
                == "Saved after protected replacement draft test"
            )
            revision = page.evaluate("roomReplacements.snapshot().project.revision")
            page.locator("#replacement-save").click()
            wait(
                "(r)=>roomReplacements.snapshot().project.revision>r&&!roomReplacements.snapshot().busy",
                revision,
            )
            page.locator("#replacement-close").click()
            wait("!document.getElementById('replacement-panel').open")
            assert not page.evaluate(
                "unifiedExperience.snapshot().dirty.includes('replacement-draft')"
            )
            passed(
                "Unsaved replacement changes are protected; saving after Stay clears the navigation guard without losing the proposal"
            )
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
            assert not errors, errors
            passed(
                "Unified dashboard/library/planning layout fits a 1280px desktop viewport without horizontal page overflow"
            )
            (OUT / "recovery-browser-tests.json").write_text(
                json.dumps(
                    {
                        "passed": len(checks),
                        "checks": checks,
                        "page_errors": errors,
                        "same_name_fixture_id": new_id,
                        "missing_plan_fixture_project_id": pid,
                        "asset_response_simulated": True,
                        "private_originals_deleted": False,
                    },
                    indent=2,
                )
            )
        finally:
            page.screenshot(path=str(OUT / "recovery-browser-final.png"))
            browser.close()


if __name__ == "__main__":
    main()
