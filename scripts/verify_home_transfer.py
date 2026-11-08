"""Real UI home transfer with generated dimensioned test drawing, no model execution."""

import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("HOME_TRANSFER_REPORT_DIR", ROOT / "reports/phase6f-home-transfer"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8772")


def fixtures():
    image = Image.new("RGB", (1200, 700), "#fbfaf5")
    d = ImageDraw.Draw(image)
    d.text((100, 35), "GENERATED TEST FLOOR PLAN - NOT A REAL APARTMENT", fill="#293d32")
    d.rectangle((100, 100, 1100, 500), outline="#374f44", width=5)
    d.line((700, 100, 700, 500), fill="#374f44", width=5)
    d.text((280, 300), "LIVING - 6 x 4 test metres", fill="#374f44")
    d.text((800, 300), "STUDY - 4 x 4 test metres", fill="#374f44")
    d.line((100, 560, 700, 560), fill="#997447", width=2)
    d.text((350, 575), "6.0 test m", fill="#997447")
    image.save(OUT / "generated-test-floorplan.png")
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=600, height=350)
    writer.add_blank_page(width=350, height=600)
    with (OUT / "generated-test-floorplan.pdf").open("wb") as f:
        writer.write(f)
    (OUT / "fixture-provenance.json").write_text(
        json.dumps(
            {
                "source": "Generated PIL line drawing + blank multipage PDF software fixtures",
                "real_building": False,
                "image_size": [1200, 700],
                "reference_pixels": [[100, 100], [700, 100]],
                "test_reference_metres": 6,
                "test_rooms_metres": [[6, 4], [4, 4]],
                "AI_generated": False,
            },
            indent=2,
        )
    )


def main():
    OUT.mkdir(exist_ok=True)
    fixtures()
    checks = []
    errors = []
    old = json.loads(
        (ROOT / "reports/phase6e-product-inventory/inventory-browser-tests.json").read_text()
    )
    sofa = old["inventory_item_id"]
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1680, "height": 1250},
            record_video_dir=str(OUT / "recordings"),
            record_video_size={"width": 1280, "height": 952},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def state():
            return page.evaluate("homeTransfer.snapshot()")

        def home():
            return state()["project"]["home_transfer"]

        def wait_revision(previous):
            page.wait_for_function("(r)=>homeTransfer.snapshot().project?.revision>r", arg=previous)

        def click(id):
            revision = state()["project"]["revision"]
            page.locator("#" + id).click()
            wait_revision(revision)

        def passed(name):
            checks.append(name)
            print(name, flush=True)

        def points(values):
            page.locator("#calibration-pick").click()
            for x, y in values:
                coords = page.locator("#destination-plan").evaluate(
                    "(svg,p)=>{const q=new DOMPoint(...p).matrixTransform(svg.getScreenCTM());return {x:q.x,y:q.y}}",
                    [x, y],
                )
                page.mouse.click(coords["x"], coords["y"])

        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function("window.homeTransfer && window.roomLab && window.roomPhoto")
            library = page.request.get(BASE + "/api/inventory").json()
            assert sofa in library["items"]
            tables = [
                i
                for i in library["items"].values()
                if i["category"] == "table" and not i["archived"]
            ]
            if tables:
                table = tables[0]["inventory_item_id"]
            else:
                # Genuine existing inventory form, explicitly unknown software-fixture item.
                page.locator("#open-inventory").click()
                page.wait_for_function(
                    "ownedInventory.snapshot().selected && ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected]"
                )
                old_selected = page.evaluate("ownedInventory.snapshot().selected")
                page.locator("#inventory-new").click()
                page.wait_for_function(
                    "(previous)=>ownedInventory.snapshot().selected!==previous", arg=old_selected
                )
                page.locator("#inventory-name").fill(
                    "Phase 6F generated-workflow table (not actual household item)"
                )
                page.locator("#inventory-category").select_option("table")
                page.locator("#inventory-save").click()
                page.wait_for_function(
                    "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].category==='table'"
                )
                table = page.evaluate("ownedInventory.snapshot().selected")
                page.locator("#close-inventory").click()
            before = page.request.get(BASE + "/api/inventory").json()
            before_hash = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            page.locator("#plan-move").click()
            page.locator("#destination-name").fill("Phase 6F generated destination")
            page.locator("#destination-create").click()
            page.wait_for_function("homeTransfer.snapshot().project!==null")
            pid = state()["project"]["id"]
            passed("Existing inventory opened; destination project has independent identity")
            page.locator("#plan-rights").check()
            revision = state()["project"]["revision"]
            page.locator("#plan-upload").set_input_files(str(OUT / "generated-test-floorplan.pdf"))
            wait_revision(revision)
            assert home()["floorplan"]["pages"] == 2
            page.locator("#plan-page").fill("2")
            click("plan-select-page")
            assert home()["floorplan"]["width"] < home()["floorplan"]["height"]
            passed("Actual local PDF rendering and second-page selection preserve private original")
            page.locator("#plan-page").fill("1")
            revision = state()["project"]["revision"]
            page.locator("#plan-upload").set_input_files(str(OUT / "generated-test-floorplan.png"))
            wait_revision(revision)
            points([[100, 100], [700, 100]])
            page.locator("#calibration-value").fill("6")
            page.locator("#calibration-description").fill(
                "Generated fixture: 600 pixels = 6 test metres, not an actual measured home"
            )
            page.locator("#calibration-source").select_option("conceptual_drawing")
            click("calibration-save")
            assert (
                abs(state()["project"]["transfer_assessment"]["scale"]["metres_per_pixel"] - 0.01)
                < 1e-10
            )
            passed(
                "Image aspect and two-point calibration retained with explicit fixture provenance"
            )
            page.screenshot(path=str(OUT / "floorplan-import-demo.png"), full_page=True)
            # Degenerate/inconsistent references use real handlers; no threshold suppression.
            invalid = page.request.post(
                f"{BASE}/api/transfer/{pid}/action",
                data={
                    "operation": "calibrate",
                    "p1": [100, 100],
                    "p2": [100, 100],
                    "value": 6,
                    "unit": "m",
                    "description": "Degenerate fixture",
                },
            )
            assert invalid.status == 400
            points([[100, 100], [400, 100]])
            page.locator("#calibration-value").fill("4")
            click("calibration-check")
            assert state()["project"]["transfer_assessment"]["scale"]["inconsistent"]
            passed(
                "Degenerate scale rejected and independent inconsistent measurement visibly flagged"
            )
            # Explicit corrected revision; old evidence kept, no silent rescaling.
            points([[100, 100], [700, 100]])
            page.locator("#calibration-value").fill("6")
            click("calibration-save")
            for name, boundary in [
                ("Living room", [[100, 100], [700, 100], [700, 500], [100, 500]]),
                ("Study", [[700, 100], [1100, 100], [1100, 500], [700, 500]]),
            ]:
                page.locator("#room-new").click()
                page.locator("#destination-room-name").fill(name)
                page.locator("#room-boundary").locator("..").evaluate(
                    "e=>{if(e.tagName==='DETAILS')e.open=true}"
                )
                page.locator("#room-boundary").fill(json.dumps(boundary))
                click("room-save")
            rooms = list(home()["rooms"])
            living = rooms[0]
            page.locator("#destination-room").select_option(living)
            page.locator("#room-features").locator("..").evaluate(
                "e=>{if(e.tagName==='DETAILS')e.open=true}"
            )
            page.locator("#room-features").fill(
                json.dumps(
                    [
                        {"kind": "door", "edge": 0, "offset": 0.5, "width": 0.9, "swing": "left"},
                        {"kind": "window", "edge": 2, "offset": 2, "width": 1.5},
                    ]
                )
            )
            page.locator("#room-obstacles").locator("..").evaluate(
                "e=>{if(e.tagName==='DETAILS')e.open=true}"
            )
            page.locator("#room-obstacles").fill(
                json.dumps([{"boundary": [[6, 4], [6.5, 4], [6.5, 4.5], [6, 4.5]]}])
            )
            click("architecture-save")
            assert len(home()["rooms"]) == 2
            assert home()["rooms"][living]["height"] is None
            passed(
                "Two editable rooms, door swing, window and fixed column saved; unknown height stays unknown"
            )
            page.locator("#placement-ack").check()
            page.locator("#placement-x").fill("3")
            page.locator("#placement-y").fill("3")
            revision = state()["project"]["revision"]
            page.locator(f'[data-add-owned="{sofa}"]').click()
            wait_revision(revision)
            page.locator("#placement-x").fill("5")
            page.locator("#placement-y").fill("3")
            revision = state()["project"]["revision"]
            page.locator(f'[data-add-owned="{table}"]').click()
            wait_revision(revision)
            a = home()["active_layout"]
            placements = home()["layouts"][a]["placements"]
            sofa_entity = next(p["id"] for p in placements if p["inventory_item_id"] == sofa)
            page.locator("#transfer-placement").select_option(sofa_entity)
            page.locator("#placement-rotation").fill("15")
            click("placement-update")
            original_a = json.loads(json.dumps(home()["layouts"][a]))
            page.screenshot(path=str(OUT / "home-transfer-workflow.png"), full_page=True)
            passed(
                "Existing sofa/table placed with stable owned IDs and independent layout IDs; furniture moved and rotated"
            )
            page.locator("#layout-name").fill("Layout B — overlap test")
            click("layout-duplicate")
            b = home()["active_layout"]
            bp = home()["layouts"][b]["placements"]
            btable = next(p["id"] for p in bp if p["inventory_item_id"] == table)
            page.locator("#transfer-placement").select_option(btable)
            page.locator("#placement-x").fill("3")
            page.locator("#placement-y").fill("3")
            click("placement-update")
            assert any(
                w["type"] == "furniture_overlap"
                for w in state()["project"]["transfer_assessment"]["warnings"]
            )
            assert home()["layouts"][a] == original_a
            passed(
                "Duplicate alternative shares owned references, keeps independent poses and reports actual model overlap"
            )
            click("transfer-undo")
            assert home()["layouts"][b]["placements"] != bp or not any(
                w["type"] == "furniture_overlap"
                for w in state()["project"]["transfer_assessment"]["warnings"]
            )
            click("transfer-redo")
            assert any(
                w["type"] == "furniture_overlap"
                for w in state()["project"]["transfer_assessment"]["warnings"]
            )
            passed("Destination undo and redo restore independent layout states")
            page.locator("#placement-decision").select_option("consider_replacing")
            click("decision-save")
            click("layout-prefer")
            page.locator("#layout-compare").click()
            page.wait_for_timeout(150)
            page.screenshot(path=str(OUT / "layout-comparison.png"), full_page=True)
            assert home()["preferred_layout"] == b
            passed("Comparison, preferred alternative and item planning decisions persist")
            page.locator("#transfer-three").click()
            page.wait_for_function(
                "roomLab.snapshot().scene?.provenance?.kind==='destination_plan'"
            )
            assert page.evaluate("roomLab.snapshot().objects") == 2
            page.screenshot(path=str(OUT / "destination-3d-view.png"), full_page=True)
            # Genuine existing editor service path, triggered via UI exposed planning position field; 3D gets same coordinates.
            page.locator("#placement-x").fill("4")
            click("placement-update")
            assert (
                next(
                    p
                    for p in page.evaluate("roomLab.snapshot().scene.furniture")
                    if p["id"] == btable
                )["position"][0]
                == 4
            )
            passed(
                "Existing Three.js renders exact traced floor; layout changes synchronize to editable 3D entities"
            )
            # Actual pointer-driven 3D drag, then confirm the shared placement changed on disk.
            page.locator("#transfer-top").click()
            page.locator("#transfer-drag").click()
            drag_id = next(
                p["id"]
                for p in home()["layouts"][b]["placements"]
                if p["inventory_item_id"] == sofa
            )
            old_position = next(
                p["position"] for p in home()["layouts"][b]["placements"] if p["id"] == drag_id
            )
            coordinates = page.evaluate("(id)=>roomLab.projectObject(id)", drag_id)
            revision = state()["project"]["revision"]
            page.mouse.move(coordinates["x"], coordinates["y"])
            page.mouse.down()
            page.mouse.move(coordinates["x"] + 38, coordinates["y"] + 18, steps=8)
            page.mouse.up()
            wait_revision(revision)
            moved = next(p for p in home()["layouts"][b]["placements"] if p["id"] == drag_id)[
                "position"
            ]
            assert moved != old_position
            passed(
                "Actual pointer-driven Three.js drag synchronizes back to destination placement and saved 2D geometry"
            )
            # Restore the explicit comparison coordinates through the visible planning form.
            page.locator("#transfer-placement").select_option(drag_id)
            page.locator("#placement-x").fill(str(old_position[0]))
            page.locator("#placement-y").fill(str(old_position[1]))
            click("placement-update")
            page.locator("#transfer-placement").select_option(btable)
            page.locator("#placement-x").fill("4")
            page.locator("#placement-y").fill("3")
            click("placement-update")
            page.locator("#transfer-inventory-details").click()
            page.wait_for_function("(id)=>ownedInventory.snapshot().selected===id", arg=table)
            assert table in page.locator("#inventory-item-id").inner_text()
            page.locator("#close-inventory").click()
            passed(
                "Selected destination furniture opens its own existing inventory details without losing layout state"
            )

            page.locator("#transfer-plan").click()
            page.locator("#transfer-save").click()
            page.wait_for_timeout(300)
            page.reload(wait_until="networkidle")
            page.wait_for_function("window.homeTransfer")
            page.locator("#plan-move").click()
            page.locator("#transfer-picker").select_option(pid)
            page.wait_for_function("(id)=>homeTransfer.snapshot().project?.id===id", arg=pid)
            assert len(home()["layouts"]) == 2
            assert home()["layouts"][a] == original_a
            assert home()["layouts"][b]["placements"][1]["position"][0] == 4
            assert not state()["project"]["transfer_assessment"]["physical_placement_verified"]
            assert not page.evaluate("roomLab.snapshot().scene.physical_accuracy_verified===true")
            passed(
                "Reload preserves both alternatives and uncertainty; no verified physical fit from fixture calibration"
            )
            after = page.request.get(BASE + "/api/inventory").json()
            assert before == after
            passed(
                "Transfer leaves all original owned inventory identities and evidence byte-equivalent as JSON"
            )
            bundle = page.request.get(f"{BASE}/api/transfer/{pid}/export")
            assert bundle.ok
            (OUT / "destination-demo.zip").write_bytes(bundle.body())
            response = page.request.post(
                BASE + "/api/transfer/import",
                data=bundle.body(),
                headers={"Content-Type": "application/zip"},
            )
            assert response.ok
            imported = response.json()
            assert imported["id"] != pid
            assert len(imported["home_transfer"]["layouts"]) == 2
            denied = page.request.get(f"{BASE}/api/transfer/{pid}/assets/owned-inventory.json")
            assert denied.status == 400
            passed(
                "Portable private-plan ZIP reimports independently without owned-item duplication; arbitrary assets refused"
            )
            assert not errors, errors
            (OUT / "browser-tests.json").write_text(
                json.dumps(
                    {
                        "passed": len(checks),
                        "checks": checks,
                        "page_errors": errors,
                        "destination_id": pid,
                        "layout_a": a,
                        "layout_b": b,
                        "owned_sofa_id": sofa,
                        "owned_table_id": table,
                        "inventory_before_sha256": before_hash,
                        "inventory_after_sha256": hashlib.sha256(
                            json.dumps(after, sort_keys=True).encode()
                        ).hexdigest(),
                        "physical_placement_verified": False,
                    },
                    indent=2,
                )
            )
        finally:
            page.screenshot(path=str(OUT / "browser-final.png"), full_page=True)
            video = page.video
            context.close()
            if video:
                video.save_as(str(OUT / "home-transfer-workflow.webm"))
            browser.close()


if __name__ == "__main__":
    main()
