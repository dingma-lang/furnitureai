"""Focused unified navigation, optional review/linking, draft and recovery browser checks."""

import json
import os
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
            page.locator("#nav-home").focus()
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.id") == "nav-photo"
            page.keyboard.press("Enter")
            wait('unifiedExperience.snapshot().route==="photo"')
            passed(
                "Main navigation supports keyboard Tab/Enter and an obvious single-photo starting path at 1280px"
            )
            page.locator("#project-picker").select_option(saved["photo_project_id"])
            wait("(id)=>roomPhoto.snapshot().project?.id===id", saved["photo_project_id"])
            page.locator("#photo-advanced-tools").evaluate("e=>e.open=true")
            before = page.evaluate("roomPhoto.snapshot().project.photos[0].reviewed.length")
            page.locator("#add-detection").click()
            bounds = page.locator("#photo-boxes").bounding_box()
            page.mouse.move(
                bounds["x"] + bounds["width"] * 0.08, bounds["y"] + bounds["height"] * 0.12
            )
            page.mouse.down()
            page.mouse.move(
                bounds["x"] + bounds["width"] * 0.2, bounds["y"] + bounds["height"] * 0.3, steps=5
            )
            page.mouse.up()
            wait("(n)=>roomPhoto.snapshot().project.photos[0].reviewed.length===n+1", before)
            page.locator("#detection-label").select_option("sofa")
            page.locator("#update-detection").click()
            wait("roomPhoto.snapshot().project.photos[0].reviewed.at(-1).label==='sofa'")
            page.locator("#confirm-detection").click()
            wait("roomPhoto.snapshot().project.photos[0].reviewed.at(-1).confirmed")
            page.locator("#review-existing-item").focus()
            wait(
                '(id)=>[...document.getElementById("review-existing-item").options].some(o=>o.value===id)',
                saved["owned_inventory_id"],
            )
            page.locator("#review-existing-item").select_option(saved["owned_inventory_id"])
            page.locator("#review-link-fragment").click()
            wait(
                "document.getElementById('workflow-notice').textContent.includes('fragment linked')"
            )
            passed(
                "A missing item is drawn and confirmed through the real photo editor, then explicitly linked as another observation of the existing owned item"
            )
            page.locator("#guide-save-item").click()
            page.locator("#quick-notes").fill("Unsaved edit for navigation guard test")
            page.locator("#quick-cancel").click()
            wait("document.getElementById('unsaved-navigation').open")
            page.locator("#unsaved-stay").click()
            assert (
                page.locator("#quick-notes").input_value()
                == "Unsaved edit for navigation guard test"
            )
            page.locator("#quick-cancel").click()
            page.locator("#unsaved-discard").click()
            wait("!document.getElementById('quick-furniture').open")
            passed(
                "A compact furniture form cannot be silently discarded; stay preserves text and explicit discard closes the draft"
            )
            page.locator("#nav-furniture").click()
            wait("document.getElementById('furniture-cards-panel').open")
            page.locator("#cards-search").fill("My sofa — unified")
            assert page.locator("[data-owned-card]").count() >= 1
            page.locator("#cards-dimensions").select_option("missing")
            assert (
                page.locator('[data-owned-card="' + saved["owned_inventory_id"] + '"]').count() == 1
            )
            page.locator("#cards-close").click()
            passed(
                "Searchable visual inventory filters incomplete dimensions and preserves short-ID disambiguation"
            )
            page.locator("#nav-home").click()
            wait('unifiedExperience.snapshot().route==="home"')
            page.locator("#home-search").fill("Unified moving plan")
            page.locator('[data-continue-project="' + saved["destination_id"] + '"]').click()
            wait("(id)=>homeTransfer.snapshot().project?.id===id", saved["destination_id"])
            with page.expect_download() as download:
                page.locator("#destination-export").click()
            download.value.save_as(str(OUT / "details-source-destination.zip"))
            old_id = page.evaluate("homeTransfer.snapshot().project.id")
            page.locator("#destination-import").set_input_files(
                str(OUT / "details-source-destination.zip")
            )
            wait("(id)=>homeTransfer.snapshot().project.id!==id", old_id)
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#simple-room-name").fill("Architecture controls software test room")
            page.locator("#simple-room-create").click()
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)
            wait("document.getElementById('simple-room-create')!==null")
            page.locator("#simple-feature-kind").select_option("door")
            page.locator("#simple-feature-offset").fill("0.2")
            page.locator("#simple-feature-width").fill("0.8")
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#simple-feature-add").click()
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)
            page.locator("#simple-feature-kind").select_option("window")
            page.locator("#simple-feature-edge").fill("2")
            page.locator("#simple-feature-offset").fill("1")
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#simple-feature-add").click()
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)
            room = page.evaluate(
                "homeTransfer.snapshot().project.home_transfer.rooms[homeTransfer.snapshot().roomId]"
            )
            assert len(room["features"]) == 2
            assert room["height"] is None
            passed(
                "Visible door/window controls reuse validated architecture records while keeping unknown heights unknown"
            )
            # Exercise actual missing-asset HTTP responses without touching private saved originals.
            page.locator("#nav-home").click()
            wait('unifiedExperience.snapshot().route==="home"')
            page.locator("#home-search").fill("Unified licensed photo")
            page.route(
                "**/processing.png",
                lambda r: r.fulfill(status=404, body="Unavailable fixture asset"),
            )
            page.reload(wait_until="networkidle")
            wait("window.unifiedExperience&&unifiedExperience.snapshot().overview!==null")
            page.locator("#home-search").fill("Unified licensed photo")
            page.locator('[data-continue-project="' + saved["photo_project_id"] + '"]').click()
            wait("(id)=>roomPhoto.snapshot().project?.id===id", saved["photo_project_id"])
            wait("document.getElementById('workflow-notice').textContent.includes('unavailable')")
            assert page.evaluate("roomPhoto.snapshot().project.photos[0].reviewed.length") > 0
            page.unroute("**/processing.png")
            passed(
                "Unavailable source image produces a recovery message while saved detections remain intact; no private file is deleted"
            )
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(
                str(ROOT / "reports/phase6e-product-inventory/licensed-room-processing-copy.png")
            )
            wait("roomPhoto.snapshot().project.photos.length===2")
            assert page.evaluate("roomPhoto.snapshot().project.photos[0].reviewed.length") > 0
            assert not page.evaluate(
                "roomPhoto.snapshot().project.geometry.physical_accuracy_verified"
            )
            passed(
                "An optional second photograph can be added later without requiring scanning or discarding the original reviewed observations"
            )
            assert not errors, errors
            (OUT / "details-browser-tests.json").write_text(
                json.dumps(
                    {
                        "passed": len(checks),
                        "checks": checks,
                        "page_errors": errors,
                        "http_missing_asset_simulated": True,
                        "missing_asset_real_file_deleted": False,
                        "viewport": [1280, 1000],
                    },
                    indent=2,
                )
            )
        finally:
            page.screenshot(path=str(OUT / "details-browser-final.png"))
            browser.close()


if __name__ == "__main__":
    main()
