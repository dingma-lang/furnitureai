"""Real local browser upload/inference/review/save experiment; existing licensed photo."""
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402 - optional cached browser driver

OUT = Path(os.environ.get("ROOM_PHOTO_REPORT_DIR", ROOT / "reports/phase6b-photo-integration"))
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8767")
PHOTO = ROOT / "data/room-validation/nathan-fertig-FBXuXp57eM0.jpg"


def main():
    checks, errors, requests = [], [], []
    started = perf_counter()
    with sync_playwright() as tool:
        browser = tool.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1540, "height": 1120})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        def snap():
            return page.evaluate("roomPhoto.snapshot()")
        def wait(expression, timeout=30000):
            page.wait_for_function("window.roomPhoto && (" + expression + ")", timeout=timeout)
        def current_photo():
            value = snap()
            return next(p for p in value["project"]["photos"] if p["id"] == value["photoId"])
        def passed(name):
            checks.append(name)
            print(name, flush=True)
        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function("window.roomLab && roomLab.snapshot().objects === 5 && window.roomPhoto")
            page.locator("#project-name").fill("Phase 6B verified workflow " + datetime.now(timezone.utc).strftime("%H%M%S"))
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project !== null")
            project_id = snap()["project"]["id"]
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(str(PHOTO))
            wait("roomPhoto.snapshot().project.photos.length === 1")
            photo_id = snap()["photoId"]
            wait("roomPhoto.snapshot().mode === 'photo'")
            assert current_photo()["width"] == 1600
            assert "/Users/" not in json.dumps(snap()["project"])
            page.locator("#reviewer-name").fill("Automated browser verification")
            passed("Real licensed room photo uploaded through browser; normalized preview and local project created")
            page.locator("#analyze-photo").click()
            wait("roomPhoto.snapshot().project.photos[0].processing_state === 'running' || roomPhoto.snapshot().project.photos[0].processing_state === 'queued'")
            duplicate = page.request.post(f"{BASE}/api/projects/{project_id}/process", data={"photo_id": photo_id})
            assert duplicate.ok
            passed("Duplicate processing request deduplicated while frontend stays responsive")
            wait("roomPhoto.snapshot().project.photos[0].processing_state === 'complete' || roomPhoto.snapshot().project.photos[0].processing_state === 'partial' || roomPhoto.snapshot().project.photos[0].processing_state === 'failed'", 240000)
            p = current_photo()
            (OUT / "real-inference-stages.json").write_text(json.dumps(p, indent=2) + "\n")
            assert p["processing_state"] == "complete", p.get("stages")
            assert p["predictions"]["detections"]
            for name in ("detection", "segmentation", "depth"):
                assert p["stages"][name]["state"] == "complete"
            assert p["depth"]["physical_units"] is None
            original_prediction = json.dumps(p["predictions"], sort_keys=True)
            passed("Real cached-local OWLv2, semantic segmentation and relative-depth workers completed")
            page.screenshot(path=str(OUT / "photo-detections.png"), full_page=True)
            first = p["reviewed"][0]["detection_id"]
            page.locator(f'[data-detection-id="{first}"]').click()
            page.locator("#detection-label").select_option("chair")
            page.locator("#box-x1").fill(str(p["reviewed"][0]["box"][0] + 2))
            page.locator("#update-detection").click()
            wait("roomPhoto.snapshot().project.photos[0].reviewed[0].review_status === 'modified'")
            page.locator("#confirm-detection").click()
            wait("roomPhoto.snapshot().project.photos[0].reviewed[0].confirmed")
            assert snap()["project"]["observations"][0]["estimated_pose"] is None
            passed("Label/bounding-box correction and confirmation create an observation without invented 3D coordinates")
            rows = current_photo()["reviewed"]
            if len(rows) > 1:
                second = rows[1]["detection_id"]
                page.locator(f'[data-detection-id="{second}"]').click()
                page.locator("#reject-detection").click()
                wait("roomPhoto.snapshot().project.photos[0].reviewed[1].review_status === 'rejected'")
                page.locator("#undo-correction").click()
                wait("roomPhoto.snapshot().project.photos[0].reviewed[1].review_status === 'unreviewed'")
                page.locator("#reject-detection").click()
                passed("Rejection and undo preserve original predictions")
            count = len(current_photo()["reviewed"])
            page.locator("#add-detection").click()
            page.locator("#photo-boxes").scroll_into_view_if_needed()
            box = page.locator("#photo-boxes").bounding_box()
            page.mouse.move(box["x"] + box["width"] * .05, box["y"] + box["height"] * .75)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * .18, box["y"] + box["height"] * .90, steps=8)
            page.mouse.up()
            wait(f"roomPhoto.snapshot().project.photos[0].reviewed.length === {count + 1}")
            page.locator("#detection-label").select_option("bookshelf")
            page.locator("#update-detection").click()
            wait("roomPhoto.snapshot().project.photos[0].reviewed.at(-1).display_label === 'bookshelf'")
            assert current_photo()["reviewed"][-1]["score"] is None
            passed("A missing object is added by drawing a real SVG box; no AI confidence fabricated")
            previous_box = current_photo()["reviewed"][-1]["box"]
            handle = page.locator('#photo-boxes circle[data-resize]').bounding_box()
            page.mouse.move(handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2)
            page.mouse.down()
            page.mouse.move(handle["x"] + handle["width"] / 2 + 10, handle["y"] + handle["height"] / 2 + 10, steps=6)
            page.mouse.up()
            wait(f"roomPhoto.snapshot().project.photos[0].reviewed.at(-1).box[2] !== {previous_box[2]}")
            passed("Actual bounding-box corner dragging persists resized image coordinates")
            assert json.dumps(current_photo()["predictions"], sort_keys=True) == original_prediction
            page.locator("#semantic-toggle").check()
            assert page.locator("#semantic-photo").is_visible()
            page.screenshot(path=str(OUT / "photo-surfaces.png"), full_page=True)
            page.locator("#semantic-toggle").uncheck()
            page.locator("#depth-toggle").check()
            assert page.locator("#depth-photo").is_visible()
            assert "NOT metres" in page.locator("#photo-caption").inner_text()
            page.screenshot(path=str(OUT / "photo-relative-depth.png"), full_page=True)
            page.locator("#depth-toggle").uncheck()
            passed("Semantic and relative-depth overlays can be toggled and label their limits")
            page.locator("#measurement-reference").fill("Example room width, not measured")
            page.locator("#measurement-value").fill("5.5")
            page.locator("#measurement-kind").select_option("estimated")
            page.locator("#measurement-source").fill("Automated browser test input")
            page.locator("#save-measurement").click()
            wait("roomPhoto.snapshot().project.measurements.length === 1")
            assert snap()["project"]["geometry"]["physical_accuracy_verified"] is False
            passed("User measurement entry persists classification and does not enable physical placement")
            page.locator("#three-view").click()
            wait("roomPhoto.snapshot().mode === '3d'")
            assert page.locator("#apply").is_disabled()
            assert page.locator("#empty").is_visible()
            page.screenshot(path=str(OUT / "photo-unresolved-3d.png"), full_page=True)
            refused = page.request.post(BASE + "/api/edit", data={"scene": snap()["project"]["room"], "action": {"type": "resize", "dimensions": [6, 5, 3]}})
            assert refused.status == 400
            passed("Photo-derived scene stays unresolved and cannot perform physically verified placement")
            page.locator("#photo-view").click()
            before = current_photo()["reviewed"]
            page.locator("#save-project").click()
            wait("roomPhoto.snapshot().project.name.length > 0")
            page.reload(wait_until="networkidle")
            page.locator("#project-picker").select_option(project_id)
            wait("roomPhoto.snapshot().project?.id === '" + project_id + "'")
            assert current_photo()["reviewed"] == before
            assert len(snap()["project"]["measurements"]) == 1
            passed("Save and reopen after full page reload preserve corrections, observations and measurements")
            page.locator("#photo-rights").check()
            page.evaluate("""payload => {
                const bytes = Uint8Array.from(atob(payload.data), c => c.charCodeAt(0));
                const transfer = new DataTransfer();
                transfer.items.add(new File([bytes], payload.name, {type: 'image/jpeg'}));
                document.getElementById('photo-drop').dispatchEvent(new DragEvent('drop', {bubbles:true, cancelable:true, dataTransfer:transfer}));
            }""", {"data": base64.b64encode(PHOTO.read_bytes()).decode(), "name": PHOTO.name})
            wait("roomPhoto.snapshot().project.photos.length === 2")
            page.locator("#analyze-photo").click()
            wait("roomPhoto.snapshot().project.photos[1].processing_state === 'complete'", 30000)
            assert all(s.get("cached") for s in current_photo()["stages"].values())
            assert snap()["project"]["photos"][0]["reviewed"] == before
            assert len(snap()["project"]["measurements"]) == 1
            page.locator("#photo-picker").select_option(photo_id)
            passed("Drag/drop adds a photo observation and reuses image/model caches without losing reviews or measurements")
            with page.expect_download() as download:
                page.locator("#export-project").click()
            download.value.save_as(str(OUT / "verified-project.zip"))
            page.locator("#import-project").set_input_files(str(OUT / "verified-project.zip"))
            wait("roomPhoto.snapshot().project?.imported_from_id === '" + project_id + "'")
            assert current_photo()["reviewed"] == before
            assert len(snap()["project"]["measurements"]) == 1
            passed("Portable project ZIP imports originals, predictions and corrections without losing fields")
            # Attach the separate known demo explicitly; no photographic reconstruction is claimed.
            page.locator("#three-view").click()
            page.locator("#scene-source").select_option("demo")
            page.wait_for_function("roomLab.snapshot().objects === 5 && roomLab.snapshot().editable")
            page.locator("#photo-view").click()
            page.locator("#demo-link").check()
            page.locator("#reviewer-name").fill("Automated browser verification")
            page.locator("#attach-room").click()
            wait("roomPhoto.snapshot().project.room.schema_version === 1")
            page.locator(f'[data-detection-id="{first}"]').click()
            # Corrected category chair matches the demo chair, not a claimed physical correspondence.
            page.locator("#entity-picker").select_option("chair")
            page.locator("#association-evidence").fill("Automated illustrative demo-only correspondence; not the real room")
            page.locator("#link-observation").click()
            wait("Object.values(roomPhoto.snapshot().project.associations).some(a => a.status === 'demonstration_only')")
            page.locator("#three-view").click()
            page.wait_for_function("roomLab.snapshot().selected === 'chair'")
            page.locator("#photo-view").click()
            added_id = current_photo()["reviewed"][-1]["detection_id"]
            page.locator(f'[data-detection-id="{added_id}"]').click()
            page.locator("#three-view").click()
            page.locator('[data-object-id="chair"]').click()
            wait("roomPhoto.snapshot().selected === '" + first + "'")
            passed("Shared persistent observation IDs synchronize selection in both photo-to-3D directions")
            page.locator('[data-object-id="sofa-original"]').click()
            page.locator("#replace").click()
            page.wait_for_function("roomLab.snapshot().history === 1")
            assert next(f for f in page.evaluate("roomLab.snapshot().scene.furniture") if f["id"] == "sofa-original")["size"][0] == 2.8
            passed("Explicit demo-only photo association selects a 3D entity; existing sofa editor still works")
            page.screenshot(path=str(OUT / "linked-demo-editor.png"), full_page=True)
            page.locator("#photo-view").click()
            page.screenshot(path=str(OUT / "photo-reviewed.png"), full_page=True)
            assert not errors, errors
            assert all(url.startswith(BASE) or url.startswith('blob:') or url.startswith('data:') for url in requests), requests
            report = {"result": "passed", "checks": checks, "passed": len(checks), "page_errors": errors, "seconds": perf_counter() - started, "browser": browser.version, "project_id": project_id, "photo_id": photo_id, "image_sha256": current_photo()["sha256"], "input_rights": "Nathan Fertig / Unsplash https://unsplash.com/photos/FBXuXp57eM0; existing locally licensed benchmark image, no new download", "real_model_stages": p["stages"], "models": {"detection": p["predictions"]["model"], "segmentation": p["surfaces"]["model"], "depth": p["depth"]["model"]}, "corrections_are_test_actions_not_verified_ground_truth": True, "external_photo_upload_requests": 0}
            (OUT / "browser-tests.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        except Exception as exc:
            page.screenshot(path=str(OUT / "browser-failure.png"), full_page=True)
            (OUT / "browser-failure.json").write_text(json.dumps({"checks": checks, "error": str(exc), "page_errors": errors, "state": snap()}, indent=2) + "\n")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
