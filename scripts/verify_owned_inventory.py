"""Real licensed-photo/model/browser inventory workflow, including cross-project reuse."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("INVENTORY_REPORT_DIR", ROOT / "reports/phase6e-product-inventory"))
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8771")


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "data/detection-pilot/photos/goldsberry-fZuleEfeA1Q.jpg"
    photograph = OUT / "licensed-room-processing-copy.png"
    if not photograph.exists():
        with Image.open(source) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            image.thumbnail((1440, 1440))
            image.save(photograph)
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1100},
            record_video_dir=str(OUT / "recordings"),
            record_video_size={"width": 1600, "height": 1100},
        )
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))

        def snap():
            return page.evaluate("roomPhoto.snapshot()")

        def inventory():
            return page.evaluate("ownedInventory.snapshot()")

        def wait(expression, timeout=30000):
            page.wait_for_function(expression, timeout=timeout)

        def passed(label):
            checks.append(label)
            print(label, flush=True)

        def shot(name):
            page.screenshot(path=str(OUT / name))

        try:
            page.goto(BASE, wait_until="networkidle")
            wait("window.roomPhoto && window.roomLab && window.ownedInventory")
            page.locator("#project-name").fill(
                "Phase6E real photo " + datetime.now(timezone.utc).strftime("%H%M%S")
            )
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project !== null")
            project_id = snap()["project"]["id"]
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(str(photograph))
            wait("roomPhoto.snapshot().project.photos.length === 1")
            page.locator("#analyze-photo").click()
            wait("roomPhoto.snapshot().project.photos[0].processing_state === 'complete'", 240000)
            photo = snap()["project"]["photos"][0]
            (OUT / "actual-model-results.json").write_text(json.dumps(photo, indent=2) + "\n")
            sofa = next(d for d in photo["reviewed"] if d["label"] == "sofa")
            page.locator(f'[data-detection-id="{sofa["detection_id"]}"]').click()
            page.locator("#confirm-detection").click()
            wait("roomPhoto.snapshot().project.observations.length >= 1")
            observation = photo["id"] + ":" + sofa["detection_id"]
            predictions = json.dumps(snap()["project"]["photos"][0]["predictions"], sort_keys=True)
            passed(
                "Licensed real room photograph analyzed by actual OWLv2/UPerNet/depth workers; sofa confirmed without metric inference"
            )
            page.locator("#find-products").click()
            wait(
                "Object.values(roomPhoto.snapshot().project.photos[0].product_matches||{}).some(r=>r.state==='complete')",
                180000,
            )
            record = snap()["project"]["photos"][0]["product_matches"][sofa["detection_id"]]
            assert len(record["candidates"]) == 5
            candidate = record["candidates"][0]
            page.locator(f'[data-confirm-product="{candidate["product_id"]}"]').click()
            wait("Object.keys(roomPhoto.snapshot().project.inventory||{}).length===1")
            rejected = record["candidates"][1]["product_id"]
            page.locator(f'[data-reject-product="{rejected}"]').click()
            wait(
                "Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].decisions['"
                + rejected
                + "']?.decision==='reject'"
            )
            passed(
                "Genuine SigLIP/FAISS candidates support separate possible-match confirmation and rejection"
            )
            page.locator("#room-setup").evaluate("el=>el.open=true")
            page.locator("#design-room-ack").check()
            page.locator("#setup-design-room").click()
            wait(
                "roomPhoto.snapshot().project.room.provenance?.kind==='approximate_design_workspace'"
            )
            page.locator("#placeholder-ack").check()
            page.locator("#create-placeholder").click()
            wait("roomPhoto.snapshot().project.room.furniture.length===1")
            entity_id = snap()["project"]["room"]["furniture"][0]["id"]
            page.locator("#open-inventory").click()
            page.locator("#inventory-from-candidate").click()
            wait(
                "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected]?.identification.status==='user_confirmed_catalog_listing'"
            )
            item_id = inventory()["selected"]
            assert (
                item_id != entity_id
                and item_id != observation
                and item_id != candidate["product_id"]
            )
            assert not inventory()["library"]["items"][item_id]["identification"]["catalog"][
                "exact_identity_verified"
            ]
            shot("photo-to-inventory-demo.png")
            passed(
                "Reusable owned ID links the confirmed observation, saved listing and existing 3D object without exact-identity claims"
            )
            page.locator("#inventory-name").fill("My sofa · inventory demo")
            page.locator("#inventory-condition").select_option("fair")
            page.locator("#inventory-stains").fill(
                "Automated UI demonstration note; no actual damage diagnosis"
            )
            page.locator("#inventory-notes").fill(
                "Demonstration dimensions below are supplied examples, not measurements of this photograph."
            )
            page.locator("#inventory-save").click()
            wait(
                "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].condition.grade==='fair'"
            )
            for field, value in [("width", "210"), ("depth", "90"), ("height", "82")]:
                page.locator("#inventory-field").select_option(field)
                page.locator("#inventory-value").fill(value)
                page.locator("#inventory-unit").select_option("cm")
                page.locator("#inventory-source").select_option("user_entered")
                page.locator("#inventory-evidence-note").fill(
                    "AUTOMATED DEMONSTRATION supplied size; not physically measured household furniture"
                )
                page.locator("#inventory-add-evidence").click()
                wait(
                    "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].specifications."
                    + field
                    + ".evidence.some(e=>e.source==='user_entered')"
                )
            page.locator("#inventory-position-ack").check()
            page.locator("#inventory-scene-object").select_option(entity_id)
            page.locator("#inventory-apply").click()
            wait(
                "roomPhoto.snapshot().project.room.furniture[0].inventory_item_id==='"
                + item_id
                + "'"
            )
            assert snap()["project"]["room"]["furniture"][0]["size"][0] == 2.1
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            assert (
                json.dumps(snap()["project"]["photos"][0]["predictions"], sort_keys=True)
                == predictions
            )
            passed(
                "Dimension/condition edits preserve evidence and AI outputs; explicit 3D application keeps physical fit disabled"
            )
            page.locator("#inventory-panel").evaluate("el=>el.scrollTop=0")
            shot("inventory-workflow.png")
            page.locator("#close-inventory").click()
            page.locator("#three-view").click()
            shot("inventory-to-3d-demo.png")
            page.locator("#save-project").click()
            wait("document.querySelector('#save-note').textContent.includes('saved')")
            page.reload(wait_until="networkidle")
            page.locator("#project-picker").select_option(project_id)
            wait("roomPhoto.snapshot().project?.id==='" + project_id + "'")
            page.locator("#open-inventory").click()
            page.locator("#inventory-list").select_option(item_id)
            assert page.locator("#inventory-condition").input_value() == "fair"
            assert any(
                link_record["observation_id"] == observation
                for link_record in inventory()["library"]["items"][item_id]["links"]
            )
            assert snap()["project"]["room"]["furniture"][0]["inventory_item_id"] == item_id
            passed(
                "Save/reopen retains the same owned item, condition, specifications, observation and scene identities"
            )
            with page.expect_download() as pending:
                page.locator("#inventory-export").click()
            pending.value.save_as(str(OUT / "browser-inventory-export.json"))
            page.locator("#inventory-import").set_input_files(
                str(OUT / "browser-inventory-export.json")
            )
            wait("document.querySelector('#inventory-status').textContent.includes('preserved')")
            assert (
                sum(
                    i["inventory_item_id"] == item_id
                    for i in inventory()["library"]["items"].values()
                )
                == 1
            )
            passed(
                "Portable export/import preserves stable IDs without overwriting or duplicating existing items"
            )
            page.locator("#inventory-owned-photos").evaluate("el=>el.open=true")
            page.locator("#inventory-photo-rights").check()
            page.locator("#inventory-photo-upload").set_input_files(str(photograph))
            wait(
                "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].owned_photos.length===1"
            )
            passed(
                "Private original owned photograph is associated only after explicit local-processing permission"
            )
            page.locator("#inventory-clear-catalog").click()
            wait(
                "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].identification.status==='unknown'"
            )
            assert (
                snap()["project"]["inventory"][observation]["candidate"]["product_id"]
                == candidate["product_id"]
            )
            passed(
                "Removing reusable product association keeps owned identity, evidence and legacy confirmed possible match"
            )
            page.locator("#close-inventory").click()
            page.locator("#project-name").fill("Phase6E future-room design")
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project.id!=='" + project_id + "'")
            second = snap()["project"]["id"]
            page.locator("#room-setup").evaluate("el=>el.open=true")
            page.locator("#design-room-ack").check()
            page.locator("#setup-design-room").click()
            wait("roomPhoto.snapshot().project.room?.schema_version===1")
            page.locator("#open-inventory").click()
            page.locator("#inventory-list").select_option(item_id)
            page.locator("#inventory-position-ack").check()
            page.locator("#inventory-place").click()
            wait("roomPhoto.snapshot().project.room.furniture.length===1")
            other_entity = snap()["project"]["room"]["furniture"][0]
            assert other_entity["inventory_item_id"] == item_id and other_entity["id"] != entity_id
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            passed(
                "Second project reuses one owned item with a different scene ID and unresolved physical location"
            )
            final = inventory()["library"]["items"][item_id]
            (OUT / "demonstrated-item.json").write_text(json.dumps(final, indent=2) + "\n")
            (OUT / "inventory-browser-tests.json").write_text(
                json.dumps(
                    {
                        "passed": len(checks),
                        "checks": checks,
                        "page_errors": errors,
                        "project_id": project_id,
                        "second_project_id": second,
                        "inventory_item_id": item_id,
                        "observation_id": observation,
                        "scene_object_id": entity_id,
                        "second_scene_object_id": other_entity["id"],
                        "identity_verified": False,
                        "physical_fit_verified": False,
                        "demonstration_dimensions_not_measurements": True,
                        "genuine_model_photo": str(photograph),
                    },
                    indent=2,
                )
                + "\n"
            )
            assert not errors
        finally:
            if errors:
                (OUT / "browser-errors.json").write_text(json.dumps(errors, indent=2) + "\n")
            page.screenshot(path=str(OUT / "browser-final.png"))
            await_video = page.video
            context.close()
            if await_video:
                await_video.save_as(str(OUT / "inventory-workflow.webm"))
            browser.close()


if __name__ == "__main__":
    main()
