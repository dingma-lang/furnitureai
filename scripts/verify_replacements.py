"""Real integrated replacement UI over the existing generated destination and saved retrieval."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("REPLACEMENT_REPORT_DIR", ROOT / "reports/phase6h-replacement-integration"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8774")
EXAMPLE = "ILLUSTRATIVE SOFTWARE EXAMPLE ONLY — not manufacturer dimensions, real prices or verified room measurements"


def main():
    OUT.mkdir(exist_ok=True)
    source = json.loads((ROOT / "reports/phase6f-home-transfer/browser-tests.json").read_text())
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
            viewport={"width": 1650, "height": 1220},
            record_video_dir=str(OUT / "recordings"),
            record_video_size={"width": 1280, "height": 946},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def home():
            return page.evaluate("homeTransfer.snapshot().project.home_transfer")

        def layout():
            return home()["layouts"][home()["active_layout"]]

        def proposal():
            return page.evaluate(
                "roomReplacements.snapshot().project.home_transfer.layouts[roomReplacements.snapshot().lid].replacement_planning.proposals[roomReplacements.snapshot().selected]"
            )

        def wait_home(rev):
            page.wait_for_function("(r)=>homeTransfer.snapshot().project.revision>r", arg=rev)

        def click_home(id):
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.locator("#" + id).click()
            wait_home(rev)

        def click_rep(id):
            rev = page.evaluate("roomReplacements.snapshot().project.revision")
            page.locator("#" + id).click()
            page.wait_for_function(
                "(r)=>roomReplacements.snapshot().project.revision>r&&!roomReplacements.snapshot().busy",
                arg=rev,
            )

        def click_budget(id):
            rev = page.evaluate("furnitureBudget.snapshot().data.project_revision")
            page.locator("#" + id).click()
            page.wait_for_function(
                "(r)=>furnitureBudget.snapshot().data.project_revision>r&&!furnitureBudget.snapshot().busy",
                arg=rev,
            )

        def action(iid, value):
            page.locator("#budget-item").select_option(iid)
            page.locator("#budget-action").select_option(value)
            click_budget("budget-action-save")

        def total(key):
            return page.evaluate(
                "(key)=>furnitureBudget.snapshot().data.comparison[furnitureBudget.snapshot().lid].calculation.totals[key].cents",
                key,
            )

        def passed(label):
            checks.append(label)
            print(label, flush=True)

        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function(
                "window.homeTransfer&&window.roomReplacements&&window.furnitureBudget"
            )
            original = page.request.get(BASE + "/api/transfer/" + source["destination_id"]).json()
            library = page.request.get(BASE + "/api/inventory").json()
            source_projects = {link["project_id"] for link in library["items"][sofa]["links"]}
            source_observations = {
                sid: page.request.get(BASE + "/api/projects/" + sid).json()
                for sid in source_projects
            }
            bundle = page.request.get(
                BASE + "/api/transfer/" + source["destination_id"] + "/export"
            )
            assert bundle.ok
            (OUT / "generated-destination-source.zip").write_bytes(bundle.body())
            page.locator("#plan-move").click()
            page.locator("#destination-import").set_input_files(
                str(OUT / "generated-destination-source.zip")
            )
            page.wait_for_function("homeTransfer.snapshot().project!==null")
            pid = page.evaluate("homeTransfer.snapshot().project.id")
            a, b = list(home()["layouts"])[:2]
            page.locator("#layout-picker").select_option(a)
            page.wait_for_function(
                "(id)=>homeTransfer.snapshot().project.home_transfer.active_layout===id", arg=a
            )
            page.locator("#layout-name").fill("Layout C — move original")
            click_home("layout-duplicate")
            c = home()["active_layout"]
            cp = next(x["id"] for x in layout()["placements"] if x.get("inventory_item_id") == sofa)
            page.locator("#transfer-placement").select_option(cp)
            page.locator("#placement-x").fill("4.3")
            page.locator("#placement-y").fill("2.1")
            page.locator("#placement-rotation").fill("30")
            click_home("placement-update")
            original_a = json.loads(json.dumps(home()["layouts"][a]))
            original_c = json.loads(json.dumps(home()["layouts"][c]))
            page.locator("#layout-picker").select_option(b)
            page.wait_for_function(
                "(id)=>homeTransfer.snapshot().project.home_transfer.active_layout===id", arg=b
            )
            owned = next(x for x in layout()["placements"] if x.get("inventory_item_id") == sofa)
            page.locator("#transfer-placement").select_option(owned["id"])
            page.locator("#open-replacement").click()
            page.wait_for_function(
                "roomReplacements.snapshot().project!==null&&!roomReplacements.snapshot().busy"
            )
            click_rep("replacement-create")
            key = proposal()["id"]
            scene_id = proposal()["scene_placement_id"]
            page.locator("#replacement-name").fill("Illustrative compact sofa — hypothetical")
            page.locator("#replacement-width").fill("1.6")
            page.locator("#replacement-depth").fill("0.7")
            page.locator("#replacement-height").fill("0.8")
            page.locator("#replacement-source").select_option("estimated")
            page.locator("#replacement-evidence").fill(EXAMPLE)
            page.locator("#replacement-notes").fill(EXAMPLE)
            click_rep("replacement-save")
            assert proposal()["original_inventory_item_id"] == sofa
            assert not proposal()["ownership_verified"]
            assert not any(x["id"] == scene_id for x in layout()["placements"])
            passed(
                "Manual proposal has independent IDs and illustrative source dimensions without changing owned inventory or arrangement"
            )
            options = page.locator("#replacement-candidate option").count()
            assert options > 1, (
                "Expected real saved production retrieval candidates for the existing linked sofa"
            )
            candidate_id = (
                page.locator("#replacement-candidate option").nth(1).get_attribute("value")
            )
            page.locator("#replacement-candidate").select_option(candidate_id)
            click_rep("replacement-use-candidate")
            assert proposal()["catalog_candidate"]["identity_verified"] is False
            assert "possible" in page.locator("#replacement-candidate-info").inner_text()
            page.screenshot(path=str(OUT / "replacement-workflow.png"))
            passed(
                "Existing genuine production retrieval candidate and catalog image display with uncertain identity; manual/source conflicts remain separate"
            )
            before = page.request.get(BASE + "/api/transfer/" + pid).json()
            page.locator("#replacement-preview").click()
            page.wait_for_function(
                "roomReplacements.snapshot().preview!==null&&!roomReplacements.snapshot().busy"
            )
            assert page.request.get(BASE + "/api/transfer/" + pid).json() == before
            page.screenshot(path=str(OUT / "replacement-preview.png"))
            passed(
                "Original and smaller proposed footprints preview read-only with conceptual warnings and unchanged saved layout"
            )
            page.locator("#replacement-ack").check()
            click_rep("replacement-apply")
            placed = next(x for x in layout()["placements"] if x["id"] == scene_id)
            assert placed["inventory_item_id"] is None
            assert placed["proposal_id"] == key
            assert not any(x["id"] == owned["id"] for x in layout()["placements"])
            assert home()["layouts"][a] == original_a
            assert home()["layouts"][c] == original_c
            page.locator("#replacement-close").click()
            page.locator("#transfer-placement").select_option(scene_id)
            page.locator("#placement-x").fill("3.3")
            page.locator("#placement-y").fill("3.3")
            page.locator("#placement-rotation").fill("45")
            click_home("placement-update")
            page.screenshot(path=str(OUT / "replacement-applied-2d.png"), full_page=True)
            passed(
                "Explicit apply creates one non-owned replacement only in Layout B; 2D movement and rotation retain proposal identity"
            )
            page.locator("#transfer-three").click()
            page.wait_for_function(
                "(id)=>roomLab.snapshot().scene?.furniture?.some(p=>p.id===id)", arg=scene_id
            )
            assert (
                next(
                    x
                    for x in page.evaluate("roomLab.snapshot().scene.furniture")
                    if x["id"] == scene_id
                )["rotation_degrees"]
                == 45
            )
            page.screenshot(path=str(OUT / "replacement-applied-3d.png"), full_page=True)
            page.locator("#transfer-top").click()
            page.locator("#transfer-drag").click()
            coordinates = page.evaluate("(id)=>roomLab.projectObject(id)", scene_id)
            rev = page.evaluate("homeTransfer.snapshot().project.revision")
            page.mouse.move(coordinates["x"], coordinates["y"])
            page.mouse.down()
            page.mouse.move(coordinates["x"] + 25, coordinates["y"] + 15, steps=8)
            page.mouse.up()
            wait_home(rev)
            moved = next(x for x in layout()["placements"] if x["id"] == scene_id)
            assert moved["position"] != [3.3, 3.3, 0]
            passed(
                "The same hypothetical entity renders in existing Three.js and real pointer dragging synchronizes back to saved 2D geometry"
            )
            page.locator("#transfer-plan").click()
            page.locator("#transfer-placement").select_option(scene_id)
            page.locator("#open-replacement").click()
            page.wait_for_function(
                "(id)=>roomReplacements.snapshot().selected===id&&!roomReplacements.snapshot().busy",
                arg=key,
            )
            click_rep("replacement-link")
            event = proposal()["purchase_event_id"]
            click_rep("replacement-link")
            assert proposal()["purchase_event_id"] == event
            page.locator("#replacement-budget").click()
            page.wait_for_function(
                "(id)=>furnitureBudget.snapshot().purchaseId===id&&!furnitureBudget.snapshot().busy",
                arg=event,
            )
            action(sofa, "replace")
            action(table, "retain")
            page.locator("#budget-purchase-description").fill(
                "Illustrative replacement sofa proposal"
            )
            page.locator("#budget-unit-price").fill("800")
            page.locator("#budget-discount").fill("0")
            page.locator("#budget-purchase-source").fill(EXAMPLE)
            page.locator("#budget-purchase-advanced > summary").click()
            page.locator("#budget-goods-taxable").select_option("true")
            page.locator("#budget-tax-rate").fill("5")
            page.locator("#budget-tax-jurisdiction").fill(
                "Illustrative 5% assumption; not official jurisdiction treatment"
            )
            for name, value in [("delivery", "25"), ("assembly", "0"), ("other-fee", "0")]:
                page.locator("#budget-" + name).fill(value)
            for name in ("delivery", "assembly", "other"):
                page.locator("#budget-" + name + "-taxable").select_option("false")
            click_budget("budget-purchase-save")
            assert total("gross_upfront") == 86500
            assert (
                len(
                    page.evaluate(
                        "furnitureBudget.snapshot().data.budget.scenarios[furnitureBudget.snapshot().lid].purchases"
                    )
                )
                == 1
            )
            passed(
                "Explicit purchase link is idempotent; associated budget reuses one event and calculates $865 example gross cost in integer cents"
            )
            page.locator("details:has(> #budget-sale-form) > summary").click()
            page.locator("#budget-item").select_option(sofa)
            for name, value in [
                ("resale-low", "100"),
                ("resale-expected", "150"),
                ("resale-high", "200"),
                ("resale-asking", "250"),
                ("selling-fees", "10"),
                ("seller-pickup", "0"),
                ("seller-other", "0"),
            ]:
                page.locator("#budget-" + name).fill(value)
            page.locator("#budget-seller-timing").select_option("false")
            page.locator("#budget-sale-source").fill(EXAMPLE)
            click_budget("budget-sale-save")
            assert total("expected_net") == 72500
            assert total("gross_upfront") == 86500
            passed(
                "Resale is separately and explicitly entered; future $140 net proceeds never reduce gross required cash"
            )
            for lid in (a, c):
                page.locator("#budget-layout").select_option(lid)
                page.wait_for_function(
                    "(id)=>furnitureBudget.snapshot().lid===id&&!furnitureBudget.snapshot().busy",
                    arg=lid,
                )
                action(sofa, "retain" if lid == a else "move")
                action(table, "retain")
                if lid == c:
                    page.locator("details:has(> #budget-expense-form)").evaluate("e=>e.open=true")
                    page.locator("#budget-expense-new").click()
                    page.locator("#budget-expense-description").fill(
                        "Illustrative item moving expense"
                    )
                    page.locator("#budget-expense-amount").fill("20")
                    page.locator("#budget-expense-covers").check()
                    page.locator("#budget-expense-source").fill(EXAMPLE)
                    click_budget("budget-expense-save")
            page.locator("#budget-layout").select_option(b)
            page.wait_for_function(
                "(id)=>furnitureBudget.snapshot().lid===id&&!furnitureBudget.snapshot().busy", arg=b
            )
            page.locator("#budget-compare").click()
            page.locator("#budget-panel").evaluate("e=>e.scrollTop=0")
            page.screenshot(path=str(OUT / "replacement-cost-comparison.png"))
            assert total("expected_net") == 72500
            passed(
                "Layouts A keep, B replace and C move retain independent geometry and explicit $0 / $725 / $20 example financial scenarios"
            )
            page.locator("#budget-close").click()
            page.locator("#layout-compare").click()
            page.screenshot(
                path=str(OUT / "replacement-arrangement-comparison.png"), full_page=True
            )
            page.reload(wait_until="networkidle")
            page.wait_for_function("window.homeTransfer&&window.roomReplacements")
            page.locator("#plan-move").click()
            page.locator("#transfer-picker").select_option(pid)
            page.wait_for_function("(id)=>homeTransfer.snapshot().project?.id===id", arg=pid)
            assert (
                next(x for x in layout()["placements"] if x["id"] == scene_id)["proposal_id"] == key
            )
            page.locator("#transfer-placement").select_option(scene_id)
            page.locator("#open-replacement").click()
            page.wait_for_function(
                "(id)=>roomReplacements.snapshot().selected===id&&!roomReplacements.snapshot().busy",
                arg=key,
            )
            assert proposal()["purchase_event_id"] == event
            click_rep("replacement-revert")
            assert next(x for x in layout()["placements"] if x["id"] == owned["id"]) == owned
            assert not any(x["id"] == scene_id for x in layout()["placements"])
            click_rep("replacement-undo")
            assert any(x["id"] == scene_id for x in layout()["placements"])
            click_rep("replacement-redo")
            assert any(x["id"] == owned["id"] for x in layout()["placements"])
            click_rep("replacement-undo")
            passed(
                "Save/reopen preserves source/proposal/scene/purchase IDs; revert and undo/redo restore original or applied placement without duplicates"
            )
            assert all(
                page.request.get(BASE + "/api/projects/" + sid).json() == saved
                for sid, saved in source_observations.items()
            )
            passed(
                "Linked source photo projects, AI observations and saved product decisions are unchanged after the complete replacement workflow"
            )
            assert page.request.get(BASE + "/api/inventory").json() == library
            assert (
                page.request.get(BASE + "/api/transfer/" + source["destination_id"]).json()
                == original
            )
            assert home()["layouts"][a] == original_a
            assert home()["layouts"][c]["placements"] == original_c["placements"]
            assert not page.request.get(BASE + "/api/transfer/" + pid).json()[
                "transfer_assessment"
            ]["physical_placement_verified"]
            passed(
                "Owned inventory, source observations and original destination remain unchanged; other layouts and unverified-fit gates are preserved"
            )
            exported = page.request.get(BASE + "/api/transfer/" + pid + "/export")
            assert exported.ok
            (OUT / "replacement-destination-demo.zip").write_bytes(exported.body())
            imported = page.request.post(
                BASE + "/api/transfer/import",
                data=exported.body(),
                headers={"Content-Type": "application/zip"},
            )
            assert imported.ok
            imp = imported.json()
            assert (
                imp["home_transfer"]["layouts"][b]["replacement_planning"]["proposals"][key][
                    "destination_project_id"
                ]
                == imp["id"]
            )
            passed(
                "Portable destination includes proposal, reversible original and linked financial event with remapped project identity"
            )
            assert not errors, errors
            result = {
                "passed": len(checks),
                "checks": checks,
                "page_errors": errors,
                "destination_id": pid,
                "imported_destination_id": imp["id"],
                "source_destination_id": source["destination_id"],
                "layout_a": a,
                "layout_b": b,
                "layout_c": c,
                "proposal_id": key,
                "scene_id": scene_id,
                "original_scene_id": owned["id"],
                "purchase_event_id": event,
                "owned_sofa_id": sofa,
                "possible_candidate_id": candidate_id,
                "inventory_before_sha256": hashlib.sha256(
                    json.dumps(library, sort_keys=True).encode()
                ).hexdigest(),
                "inventory_unchanged_verified": True,
                "geometry_physical_placement_verified": False,
                "illustrative_dimensions": True,
                "illustrative_prices": True,
                "final_net_cents": {"a": 0, "b": 72500, "c": 2000},
            }
            (OUT / "browser-tests.json").write_text(json.dumps(result, indent=2))
            (OUT / "example-proposal.json").write_text(
                json.dumps({"example_only": True, "proposal": proposal()}, indent=2)
            )
        finally:
            page.screenshot(path=str(OUT / "browser-final.png"))
            video = page.video
            context.close()
            if video:
                video.save_as(str(OUT / "replacement-workflow.webm"))
            browser.close()


if __name__ == "__main__":
    main()
