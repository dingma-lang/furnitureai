"""Continuous actual browser photo→owned item→destination→replacement→budget workflow."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = ROOT / "reports/phase6i-unified-experience"
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8775")
EXAMPLE = "ILLUSTRATIVE SOFTWARE EXAMPLE ONLY; not manufacturer measurements, real market quotes or verified building dimensions"


def main():
    OUT.mkdir(exist_ok=True)
    checks = []
    errors = []
    requests = []
    times = {}
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1150},
            record_video_dir=str(OUT / "recordings"),
            record_video_size={"width": 1280, "height": 920},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append({"method": r.method, "url": r.url}))

        def wait(expr, arg=None, timeout=30000):
            page.wait_for_function(expr, arg=arg, timeout=timeout)

        def photo():
            s = page.evaluate("roomPhoto.snapshot()")
            return next(p for p in s["project"]["photos"] if p["id"] == s["photoId"])

        def home():
            return page.evaluate("homeTransfer.snapshot().project.home_transfer")

        def layout():
            return home()["layouts"][home()["active_layout"]]

        def prop():
            return page.evaluate(
                "roomReplacements.snapshot().project.home_transfer.layouts[roomReplacements.snapshot().lid].replacement_planning.proposals[roomReplacements.snapshot().selected]"
            )

        def click_home(id):
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#" + id).click()
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)

        def click_rep(id):
            rev = page.evaluate("roomReplacements.snapshot().project.revision")
            page.locator("#" + id).click()
            wait(
                "(r)=>roomReplacements.snapshot().project.revision>r&&!roomReplacements.snapshot().busy",
                rev,
            )

        def click_budget(id):
            rev = page.evaluate("furnitureBudget.snapshot().data.project_revision")
            page.locator("#" + id).click()
            wait(
                "(r)=>furnitureBudget.snapshot().data.project_revision>r&&!furnitureBudget.snapshot().busy",
                rev,
            )

        def total(k):
            return page.evaluate(
                "(k)=>furnitureBudget.snapshot().data.comparison[furnitureBudget.snapshot().lid].calculation.totals[k].cents",
                k,
            )

        def passed(s):
            checks.append(s)
            print(s, flush=True)

        try:
            t = perf_counter()
            page.goto(BASE, wait_until="networkidle")
            wait("window.unifiedExperience&&unifiedExperience.snapshot().overview!==null")
            times["dashboard_load_seconds"] = perf_counter() - t
            assert page.locator("#home-photo").is_visible()
            assert not any(
                "/process" in r["url"] or "/match-products" in r["url"] for r in requests
            )
            page.screenshot(path=str(OUT / "unified-dashboard.png"))
            passed(
                "Home shows recent project types and honest progress without starting model analysis"
            )
            page.locator("#home-photo").click()
            page.locator("#project-name").fill("Unified licensed photo " + stamp)
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project!==null")
            photo_project_id = page.evaluate("roomPhoto.snapshot().project.id")
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(
                str(ROOT / "reports/phase6e-product-inventory/licensed-room-processing-copy.png")
            )
            wait("roomPhoto.snapshot().project.photos.length===1")
            t = perf_counter()
            page.locator("#guide-analyze").click()
            wait(
                "roomPhoto.snapshot().project.photos[0].processing_state==='complete'",
                timeout=600000,
            )
            times["analysis_request_to_complete_seconds"] = perf_counter() - t
            actual = photo()
            assert actual["predictions"]["model"]
            assert actual["reviewed"]
            (OUT / "actual-analysis.json").write_text(
                json.dumps(
                    {
                        "stages": actual["stages"],
                        "model": actual["predictions"]["model"],
                        "photo_project_id": photo_project_id,
                        "photo_id": actual["id"],
                        "licensed_source": "Phillip Goldsberry photo, data/detection-pilot/ATTRIBUTION.md",
                        "mocked": False,
                    },
                    indent=2,
                )
            )
            passed(
                "One licensed room photograph uses the genuine local analysis pipeline or its hash-matched saved caches"
            )
            sofa = next(d for d in actual["reviewed"] if d["label"] == "sofa")
            page.locator('[data-detection-id="' + sofa["detection_id"] + '"]').click()
            page.locator("#review-display-name").fill("My sofa — unified example " + stamp)
            page.locator("#review-name-save").click()
            wait(
                "(name)=>roomPhoto.snapshot().project.photos[0].reviewed.some(d=>d.display_name===name)",
                "My sofa — unified example " + stamp,
            )
            page.locator("#confirm-detection").click()
            wait(
                "(id)=>roomPhoto.snapshot().project.photos[0].reviewed.find(d=>d.detection_id===id).confirmed",
                sofa["detection_id"],
            )
            page.locator("#guide-save-item").click()
            page.locator("#quick-name").fill("My sofa — unified example " + stamp)
            page.locator("#quick-notes").fill(EXAMPLE)
            page.locator("#quick-save").click()
            wait("document.getElementById('furniture-cards-panel').open")
            cards = page.locator("[data-owned-card]")
            assert cards.count() > 0
            iid = (
                page.locator("[data-owned-card]")
                .filter(has_text="My sofa — unified example " + stamp)
                .get_attribute("data-owned-card")
            )
            assert iid
            page.screenshot(path=str(OUT / "photo-review-to-inventory.png"))
            passed(
                "User names and accepts a real sofa detection and saves it to My Furniture with unknown dimensions and retained source observation"
            )
            page.locator('[data-edit-owned="' + iid + '"]').click()
            wait("(id)=>ownedInventory.snapshot().selected===id", iid)
            assert page.evaluate(
                "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].links[0].observation_id"
            ).startswith(actual["id"])
            page.locator("#close-inventory").click()
            page.locator("#nav-furniture").click()
            wait("document.getElementById('furniture-cards-panel').open")
            page.locator('[data-move-owned="' + iid + '"]').click()
            page.locator("#handoff-name").fill("Unified moving plan " + stamp)
            page.locator("#handoff-open").click()
            wait("homeTransfer.snapshot().project!==null")
            pid = page.evaluate("homeTransfer.snapshot().project.id")
            page.locator("#simple-room-name").fill("Illustrative living room")
            page.locator("#simple-room-width").fill("6")
            page.locator("#simple-room-depth").fill("4")
            page.locator("#simple-room-source").fill(EXAMPLE)
            click_home("simple-room-create")
            assert len(home()["rooms"]) == 1
            assert next(iter(home()["rooms"].values()))["evidence_status"] == "unknown"
            assert home()["floorplan"] is None
            page.locator("#placement-ack").check()
            page.locator("#placement-x").fill("2")
            page.locator("#placement-y").fill("2")
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator('[data-add-owned="' + iid + '"]').click()
            wait("(r)=>homeTransfer.snapshot().project.revision>r", rev)
            owned = layout()["placements"][0]
            assert owned["inventory_item_id"] == iid
            assert owned["dimension_status"] == "unknown_approximate_placeholder"
            page.screenshot(path=str(OUT / "inventory-to-move.png"), full_page=True)
            passed(
                "Library handoff creates an explicitly illustrative manual room and places the same owned ID without compulsory measurements or a floor plan"
            )
            a = home()["active_layout"]
            a_original = json.loads(json.dumps(layout()))
            page.locator("#layout-name").fill("Layout B — illustrative replacement")
            click_home("layout-duplicate")
            b = home()["active_layout"]
            original_scene = layout()["placements"][0]["id"]
            page.locator("#transfer-placement").select_option(original_scene)
            page.locator("#open-replacement").click()
            wait("roomReplacements.snapshot().project!==null&&!roomReplacements.snapshot().busy")
            click_rep("replacement-create")
            key = prop()["id"]
            scene_id = prop()["scene_placement_id"]
            page.locator("#replacement-name").fill("Illustrative compact sofa")
            page.locator("#replacement-width").fill("1.6")
            page.locator("#replacement-depth").fill("0.7")
            page.locator("#replacement-height").fill("0.8")
            page.locator("#replacement-source").select_option("estimated")
            page.locator("#replacement-evidence").fill(EXAMPLE)
            click_rep("replacement-save")
            page.locator("#replacement-preview").click()
            wait("roomReplacements.snapshot().preview!==null&&!roomReplacements.snapshot().busy")
            page.screenshot(path=str(OUT / "integrated-replacement-workflow.png"))
            page.locator("#replacement-ack").check()
            click_rep("replacement-apply")
            assert home()["layouts"][a] == a_original
            page.locator("#replacement-close").click()
            page.locator("#transfer-placement").select_option(scene_id)
            page.locator("#transfer-three").click()
            wait("(id)=>roomLab.snapshot().scene?.furniture?.some(f=>f.id===id)", scene_id)
            assert not page.evaluate('roomLab.snapshot().status!=="disabled"')
            page.locator("#transfer-plan").click()
            passed(
                "Independent Layout B previews/applies an unknown-identity replacement and renders the same non-owned entity in 2D and Three.js with fit disabled"
            )
            page.locator("#open-replacement").click()
            wait(
                "(id)=>roomReplacements.snapshot().selected===id&&!roomReplacements.snapshot().busy",
                key,
            )
            click_rep("replacement-link")
            event = prop()["purchase_event_id"]
            page.locator("#replacement-budget").click()
            wait(
                "(id)=>furnitureBudget.snapshot().purchaseId===id&&!furnitureBudget.snapshot().busy",
                event,
            )
            page.locator("#budget-item").select_option(iid)
            page.locator("#budget-action").select_option("replace")
            click_budget("budget-action-save")
            page.locator("#budget-unit-price").fill("800")
            page.locator("#budget-discount").fill("0")
            page.locator("#budget-purchase-source").fill(EXAMPLE)
            page.locator("#budget-purchase-advanced > summary").click()
            page.locator("#budget-goods-taxable").select_option("true")
            page.locator("#budget-tax-rate").fill("5")
            page.locator("#budget-tax-jurisdiction").fill("Illustrative assumed rate only")
            for k, v in [("delivery", "25"), ("assembly", "0"), ("other-fee", "0")]:
                page.locator("#budget-" + k).fill(v)
            for k in ("delivery", "assembly", "other"):
                page.locator("#budget-" + k + "-taxable").select_option("false")
            click_budget("budget-purchase-save")
            page.locator("details:has(> #budget-sale-form) > summary").click()
            page.locator("#budget-item").select_option(iid)
            for k, v in [
                ("resale-low", "100"),
                ("resale-expected", "150"),
                ("resale-high", "200"),
                ("resale-asking", "200"),
                ("selling-fees", "10"),
                ("seller-pickup", "0"),
                ("seller-other", "0"),
            ]:
                page.locator("#budget-" + k).fill(v)
            page.locator("#budget-seller-timing").select_option("false")
            page.locator("#budget-sale-source").fill(EXAMPLE)
            click_budget("budget-sale-save")
            assert total("gross_upfront") == 86500
            assert total("expected_net") == 72500
            page.locator("#budget-layout").select_option(a)
            wait("(id)=>furnitureBudget.snapshot().lid===id&&!furnitureBudget.snapshot().busy", a)
            page.locator("#budget-item").select_option(iid)
            page.locator("#budget-action").select_option("retain")
            click_budget("budget-action-save")
            assert total("expected_net") == 0
            page.locator("#budget-layout").select_option(b)
            wait("(id)=>furnitureBudget.snapshot().lid===id&&!furnitureBudget.snapshot().busy", b)
            page.locator("#budget-compare").click()
            page.locator("#budget-simple-toggle").click()
            page.locator("#budget-panel").evaluate("e=>e.scrollTop=0")
            page.screenshot(path=str(OUT / "simplified-budget.png"))
            assert page.locator("#budget-panel").get_attribute("class") == "budget-simple"
            assert total("expected_net") == 72500
            assert "$725.00" in page.locator("#budget-summary").inner_text()
            page.locator("#budget-simple-toggle").click()
            assert total("expected_net") == 72500
            passed(
                "Linked budget uses illustrative exact-cent purchase/tax/delivery/resale assumptions; simple and advanced presentations share identical $865 cash and $725 expected net"
            )
            page.locator("#budget-close").click()
            page.locator("#transfer-save").click()
            wait(
                "document.getElementById('transfer-message').textContent.includes('Saved and reloaded')"
            )
            page.reload(wait_until="networkidle")
            wait("window.unifiedExperience&&unifiedExperience.snapshot().overview!==null")
            page.locator("#home-search").fill("Unified moving plan " + stamp)
            page.locator('[data-continue-project="' + pid + '"]').click()
            wait("(id)=>homeTransfer.snapshot().project?.id===id", pid)
            assert len(home()["layouts"]) == 2
            assert (
                next(x for x in layout()["placements"] if x["id"] == scene_id)["proposal_id"] == key
            )
            page.locator("#transfer-placement").select_option(scene_id)
            page.locator("#open-replacement").click()
            wait(
                "(id)=>roomReplacements.snapshot().selected===id&&!roomReplacements.snapshot().busy",
                key,
            )
            assert prop()["purchase_event_id"] == event
            click_rep("replacement-revert")
            assert layout()["placements"][0]["inventory_item_id"] == iid
            assert home()["layouts"][a]["placements"] == a_original["placements"]
            passed(
                "Dashboard discovery reopens saved destination with stable owned/proposal/scene/purchase IDs; revert changes only Layout B"
            )
            page.locator("#replacement-close").click()
            page.locator("#open-simple-budget").click()
            wait("document.getElementById('budget-panel').open")
            assert total("expected_net") == 72500
            assert not page.evaluate(
                "homeTransfer.snapshot().project.transfer_assessment?.physical_placement_verified===true"
            )
            page.locator("#budget-close").click()
            page.locator("#nav-home").click()
            wait('unifiedExperience.snapshot().route==="home"')
            page.locator("#home-search").fill("Unified licensed photo " + stamp)
            page.locator('[data-continue-project="' + photo_project_id + '"]').click()
            wait("(id)=>roomPhoto.snapshot().project?.id===id", photo_project_id)
            assert photo()["reviewed"]
            assert len(page.evaluate("roomPhoto.snapshot().project.photos")) == 1
            passed(
                "Return-to-home navigation retains the single-photo source project and financial records without ownership changes"
            )
            assert not errors, errors
            result = {
                "passed": len(checks),
                "checks": checks,
                "page_errors": errors,
                "photo_project_id": photo_project_id,
                "photo_id": actual["id"],
                "owned_inventory_id": iid,
                "destination_id": pid,
                "layout_a": a,
                "layout_b": b,
                "proposal_id": key,
                "replacement_scene_id": scene_id,
                "purchase_event_id": event,
                "timings_seconds": times,
                "dashboard_fetch_ms": page.evaluate("unifiedExperience.timings.dashboard_ms"),
                "model_stage_records": actual["stages"],
                "illustrative_dimensions_and_prices": True,
                "physical_placement_verified": False,
                "all_workflow_mutations_via_browser_ui": True,
                "background_requests_started_by_dashboard": False,
            }
            (OUT / "end-to-end-browser-tests.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(times))
        finally:
            page.screenshot(path=str(OUT / "browser-final.png"))
            video = page.video
            context.close()
            if video:
                video.save_as(str(OUT / "end-to-end-workflow.webm"))
            browser.close()


if __name__ == "__main__":
    main()
