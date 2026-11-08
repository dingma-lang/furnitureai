"""Real browser/model workflow and WebM demo for confirmed photo placeholders."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("ROOM_3D_REPORT_DIR", ROOT / "reports/phase6c-photo-to-3d"))
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8768")
PHOTO = ROOT / "data/detection-pilot/photos/goldsberry-fZuleEfeA1Q.jpg"


def main():
    checks, errors = [], []
    start = perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, record_video_dir=str(OUT / "recordings"), record_video_size={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        def snap():
            return page.evaluate("roomPhoto.snapshot()")
        def wait(expression, timeout=30000):
            page.wait_for_function("window.roomPhoto && (" + expression + ")", timeout=timeout)
        def passed(name):
            checks.append(name)
            print(name, flush=True)
        def shot(name):
            for selector in (".library", ".inspector"):
                page.locator(selector).evaluate("element => element.scrollTop = 0")
            page.screenshot(path=str(OUT / name), full_page=True)
        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function("window.roomPhoto && window.roomLab && roomLab.snapshot().objects === 5")
            page.locator("#project-name").fill("Phase 6C automated real photo " + datetime.now(timezone.utc).strftime("%H%M%S"))
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project !== null")
            project_id = snap()["project"]["id"]
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(str(PHOTO))
            wait("roomPhoto.snapshot().project.photos.length === 1")
            photo_id = snap()["photoId"]
            page.locator("#reviewer-name").fill("Phase 6C automated verification")
            page.locator("#analyze-photo").click()
            wait("['complete','partial','failed'].includes(roomPhoto.snapshot().project.photos[0].processing_state)", 240000)
            photo = snap()["project"]["photos"][0]
            assert photo["processing_state"] == "complete", photo["stages"]
            sofa = next(d for d in photo["reviewed"] if d["label"] == "sofa")
            original_predictions = json.dumps(photo["predictions"], sort_keys=True)
            (OUT / "actual-model-results.json").write_text(json.dumps(photo, indent=2) + "\n")
            passed("Real licensed room photo uploaded and existing CPU model workers return actual furniture, semantics and depth")
            page.locator(f'[data-detection-id="{sofa["detection_id"]}"]').click()
            page.locator("#confirm-detection").click()
            wait("roomPhoto.snapshot().project.observations.length === 1")
            observation = photo_id + ":" + sofa["detection_id"]
            assert snap()["project"]["observations"][0]["estimated_pose"] is None
            shot("01-confirmed-photo.png")
            passed("Actual sofa detection confirmed without inferring physical coordinates")
            page.locator("#room-setup").evaluate("element => element.open = true")
            page.locator("#design-room-ack").check()
            page.locator("#setup-design-room").click()
            wait("roomPhoto.snapshot().project.room.provenance?.kind === 'approximate_design_workspace'")
            assert snap()["project"]["room"]["provenance"]["dimension_evidence"]["value"] is None
            assert snap()["project"]["room_history"]
            passed("Explicit approximate rectangular workspace preserves unknown measurements and earlier photo room")
            page.locator("#placeholder-ack").check()
            page.locator("#create-placeholder").click()
            wait("roomPhoto.snapshot().project.room.furniture.length === 1")
            item = snap()["project"]["room"]["furniture"][0]
            entity_id = item["id"]
            assert snap()["project"]["associations"][observation]["entity_id"] == entity_id
            assert item["design_evidence"]["dimensions"]["value"] is None
            page.locator("#three-view").click()
            page.wait_for_function("roomLab.snapshot().objects === 1 && roomLab.snapshot().selected === '" + entity_id + "'")
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            assert page.locator("#apply").is_enabled()
            assert "APPROXIMATE" in page.locator("#view-label").inner_text()
            shot("02-linked-placeholder-before.png")
            passed("Confirmed observation creates and selects a category-appropriate editable placeholder with fit disabled")
            page.locator("#size-x").fill("2.6")
            page.locator("#position-x").fill("3")
            page.locator("#position-y").fill("2.8")
            page.locator("#rotation").fill("30")
            page.locator("#apply").click()
            wait("roomPhoto.snapshot().project.photo_objects?.['" + entity_id + "']?.entity.rotation_degrees === 30")
            assert snap()["project"]["room"]["furniture"][0]["size"][0] == 2.6
            assert json.dumps(snap()["project"]["photos"][0]["predictions"], sort_keys=True) == original_predictions
            assert snap()["project"]["associations"][observation]["status"] == "editable_placeholder"
            passed("Dimensions, position and rotation edit independently of immutable AI predictions and retain object identity")
            page.locator("#replace").click()
            wait("roomPhoto.snapshot().project.photo_objects?.['" + entity_id + "']?.entity.size[0] === 2.8")
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            assert "disabled" in page.locator("#feedback").inner_text()
            shot("03-replacement-after.png")
            (OUT / "object-after-replacement.json").write_text(json.dumps(snap()["project"]["photo_objects"][entity_id], indent=2) + "\n")
            passed("Recognized sofa can be replaced with another placeholder without a physical-fit assertion")
            page.locator("#top").click()
            page.wait_for_timeout(300)
            shot("04-top-down-design.png")
            page.locator("#remove").click()
            wait("roomPhoto.snapshot().project.photo_objects?.['" + entity_id + "']?.state === 'removed'")
            assert snap()["project"]["associations"][observation]["status"] == "placeholder_removed"
            page.reload(wait_until="networkidle")
            page.locator("#project-picker").select_option(project_id)
            wait("roomPhoto.snapshot().project?.id === '" + project_id + "'")
            assert snap()["project"]["photo_objects"][entity_id]["state"] == "removed"
            page.locator(f'[data-restore-id="{entity_id}"]').click()
            wait("roomPhoto.snapshot().project.photo_objects?.['" + entity_id + "']?.state === 'active'")
            assert snap()["project"]["room"]["furniture"][0]["id"] == entity_id
            passed("Removal, reload and persistent restore retain the same photo-linked object ID and replacement properties")
            page.locator("#save-project").click()
            page.wait_for_timeout(250)
            page.reload(wait_until="networkidle")
            page.locator("#project-picker").select_option(project_id)
            wait("roomPhoto.snapshot().project?.id === '" + project_id + "'")
            assert snap()["project"]["room"]["furniture"][0]["rotation_degrees"] == 30
            assert snap()["project"]["room"]["furniture"][0]["size"] == [2.8, 1.05, .9]
            assert snap()["project"]["associations"][observation]["entity_id"] == entity_id
            assert len(snap()["project"]["editing_history"]) >= 5
            passed("Save/reopen preserves linked identity, edited transforms, replacement dimensions and editing history")
            page.locator(f'[data-detection-id="{sofa["detection_id"]}"]').click()
            page.locator("#three-view").click()
            page.wait_for_function("roomLab.snapshot().selected === '" + entity_id + "'")
            page.locator('[data-object-id="' + entity_id + '"]').click()
            page.locator("#photo-view").click()
            assert snap()["selected"] == sofa["detection_id"]
            passed("Photo and 3D selection remain connected after reload")
            with page.expect_download() as download:
                page.locator("#export-project").click()
            download.value.save_as(str(OUT / "editable-photo-project.zip"))
            page.locator("#import-project").set_input_files(str(OUT / "editable-photo-project.zip"))
            wait("roomPhoto.snapshot().project?.imported_from_id === '" + project_id + "'")
            assert snap()["project"]["associations"][observation]["entity_id"] == entity_id
            assert snap()["project"]["photo_objects"][entity_id]["state"] == "active"
            page.locator("#three-view").click()
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            shot("05-reopened-project.png")
            passed("Portable ZIP roundtrip preserves linked objects and still cannot claim calibrated placement")
            assert not errors, errors
            report = {"result": "passed", "passed": len(checks), "checks": checks, "page_errors": errors, "seconds": perf_counter() - start, "browser": browser.version, "project_id": project_id, "photo_id": photo_id, "entity_id": entity_id, "observation_id": observation, "input": "Phillip Goldsberry / Unsplash fZuleEfeA1Q; existing licensed local photo", "real_model_stages": photo["stages"], "model_names": {"detection": photo["predictions"]["model"], "segmentation": photo["surfaces"]["model"], "depth": photo["depth"]["model"]}, "physical_fit_verified": False, "recording": "browser-demo.webm", "test_actions_not_ground_truth": True}
            (OUT / "browser-tests.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        except Exception as exc:
            shot("browser-failure.png")
            (OUT / "browser-failure.json").write_text(json.dumps({"checks": checks, "error": str(exc), "page_errors": errors, "state": snap()}, indent=2) + "\n")
            raise
        finally:
            video = page.video
            context.close()
            video.save_as(str(OUT / "browser-demo.webm"))
            browser.close()


if __name__ == "__main__":
    main()
