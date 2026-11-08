"""Focused real UI checks for relink idempotency and duplication of coupled replacements."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("REPLACEMENT_REPORT_DIR", ROOT / "reports/phase6h-replacement-integration"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8774")


def main():
    saved = json.loads((OUT / "browser-tests.json").read_text())
    checks = []
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = b.new_page(viewport={"width": 1550, "height": 1150})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_function("window.homeTransfer&&window.roomReplacements")
        original = page.request.get(BASE + "/api/transfer/" + saved["destination_id"]).json()
        bundle = page.request.get(BASE + "/api/transfer/" + saved["destination_id"] + "/export")
        (OUT / "integrity-source.zip").write_bytes(bundle.body())
        page.locator("#plan-move").click()
        page.locator("#destination-import").set_input_files(str(OUT / "integrity-source.zip"))
        page.wait_for_function("homeTransfer.snapshot().project!==null")
        pid = page.evaluate("homeTransfer.snapshot().project.id")
        page.locator("#transfer-placement").select_option(saved["scene_id"])
        page.locator("#open-replacement").click()
        page.wait_for_function(
            "roomReplacements.snapshot().project!==null&&!roomReplacements.snapshot().busy"
        )

        def click(id):
            rev = page.evaluate("roomReplacements.snapshot().project.revision")
            page.locator("#" + id).click()
            page.wait_for_function(
                "(r)=>roomReplacements.snapshot().project.revision>r&&!roomReplacements.snapshot().busy",
                arg=rev,
            )

        def current():
            return page.evaluate(
                "roomReplacements.snapshot().project.home_transfer.layouts[roomReplacements.snapshot().lid].replacement_planning.proposals[roomReplacements.snapshot().selected]"
            )

        click("replacement-unlink")
        click("replacement-link")
        assert current()["purchase_event_id"] == saved["purchase_event_id"]
        click("replacement-undo")
        assert current()["purchase_event_id"] is None
        click("replacement-link")
        assert current()["purchase_event_id"] == saved["purchase_event_id"]
        budget = page.request.get(BASE + "/api/budget/" + pid).json()
        assert len(budget["budget"]["scenarios"][saved["layout_b"]]["purchases"]) == 1
        assert (
            budget["comparison"][saved["layout_b"]]["calculation"]["totals"]["expected_net"][
                "cents"
            ]
            == 72500
        )
        checks.append(
            "Unlink/relink and undo/relink reuse one existing priced event without duplicate cost or lost financial evidence"
        )
        page.locator("#replacement-close").click()
        page.locator("#layout-name").fill("Integrity copy — coupled replacement")
        rev = page.evaluate("homeTransfer.snapshot().project.revision")
        page.locator("#layout-duplicate").click()
        page.wait_for_function("(r)=>homeTransfer.snapshot().project.revision>r", arg=rev)
        home = page.evaluate("homeTransfer.snapshot().project.home_transfer")
        lid = home["active_layout"]
        layout = home["layouts"][lid]
        proposal = next(iter(layout["replacement_planning"]["proposals"].values()))
        assert proposal["id"] != saved["proposal_id"]
        assert proposal["scene_placement_id"] != saved["scene_id"]
        assert proposal["purchase_event_id"] == saved["purchase_event_id"]
        assert proposal["layout_id"] == lid
        page.locator("#transfer-placement").select_option(proposal["scene_placement_id"])
        page.locator("#open-replacement").click()
        page.wait_for_function(
            "(id)=>roomReplacements.snapshot().selected===id&&!roomReplacements.snapshot().busy",
            arg=proposal["id"],
        )
        click("replacement-revert")
        assert any(
            x["inventory_item_id"] == saved["owned_sofa_id"]
            for x in page.evaluate(
                "homeTransfer.snapshot().project.home_transfer.layouts[homeTransfer.snapshot().project.home_transfer.active_layout].placements"
            )
        )
        r = page.request.get(BASE + "/api/budget/" + pid).json()
        assert (
            r["budget"]["scenarios"][lid]["id"] != r["budget"]["scenarios"][saved["layout_b"]]["id"]
        )
        assert r["comparison"][lid]["calculation"]["totals"]["expected_net"]["cents"] == 72500
        assert (
            page.request.get(BASE + "/api/transfer/" + saved["destination_id"]).json() == original
        )
        checks.append(
            "Duplicating an applied replacement remaps scene/proposal IDs, copies an independent budget and safely reverts its own original snapshot"
        )
        assert not errors
        result = {
            "passed": len(checks),
            "checks": checks,
            "page_errors": errors,
            "integrity_copy_destination_id": pid,
        }
        (OUT / "integrity-browser-tests.json").write_text(json.dumps(result, indent=2))
        b.close()
        print(result)


if __name__ == "__main__":
    main()
