"""Browser query of actual detected furniture against the real licensed ABO index."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(os.environ.get("PRODUCT_REPORT_DIR", ROOT / "reports/phase6d-product-identification"))
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8769")
PHOTO = ROOT / "data/detection-pilot/photos/goldsberry-fZuleEfeA1Q.jpg"


def main():
    checks, errors = [], []
    started = perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))
        def snap():
            return page.evaluate("roomPhoto.snapshot()")
        def wait(expression, timeout=30000):
            page.wait_for_function("window.roomPhoto && (" + expression + ")", timeout=timeout)
        def matches():
            s = snap()
            return s["project"]["photos"][0]["product_matches"][s["selected"]]
        def passed(name):
            checks.append(name)
            print(name, flush=True)
        def shot(name):
            page.screenshot(path=str(OUT / name), full_page=True)
        try:
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_function("window.roomPhoto && window.roomLab")
            page.locator("#project-name").fill("Phase 6D real catalog test " + datetime.now(timezone.utc).strftime("%H%M%S"))
            page.locator("#new-project").click()
            wait("roomPhoto.snapshot().project !== null")
            project_id = snap()["project"]["id"]
            page.locator("#photo-rights").check()
            page.locator("#photo-upload").set_input_files(str(PHOTO))
            wait("roomPhoto.snapshot().project.photos.length === 1")
            page.locator("#analyze-photo").click()
            wait("roomPhoto.snapshot().project.photos[0].processing_state === 'complete'", 120000)
            photo = snap()["project"]["photos"][0]
            detection = next(d for d in photo["reviewed"] if d["label"] == "sofa")
            page.locator(f'[data-detection-id="{detection["detection_id"]}"]').click()
            page.locator("#confirm-detection").click()
            wait("roomPhoto.snapshot().project.observations.length === 1")
            passed("Real room photo and saved genuine OWLv2 sofa observation available and confirmed")
            page.locator("#find-products").click()
            wait("roomPhoto.snapshot().project.photos[0].product_matches?.[roomPhoto.snapshot().selected]?.state === 'complete'", 120000)
            initial = matches()
            assert len(initial["candidates"]) == 5
            assert len({r["product_id"] for r in initial["candidates"]}) == 5
            assert all(not c["identity_verified"] for c in initial["candidates"])
            assert initial["exact_identification"] == "No verified exact match."
            assert len(page.locator(".product-card").all()) == 5
            page.locator("#product-candidates").scroll_into_view_if_needed()
            shot("product-candidates.png")
            (OUT / "actual-room-query.json").write_text(json.dumps(initial, indent=2) + "\n")
            passed("Genuine SigLIP 2 crop embedding retrieves five unique ABO identities via FAISS with no exact-match claim")
            first, second = initial["candidates"][:2]
            page.locator(f'[data-confirm-product="{first["product_id"]}"]').click()
            wait("Object.keys(roomPhoto.snapshot().project.inventory || {}).length === 1")
            assert not next(iter(snap()["project"]["inventory"].values()))["identity_verified"]
            page.locator(f'[data-reject-product="{second["product_id"]}"]').click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].decisions['" + second["product_id"] + "']?.decision === 'reject'")
            passed("User confirmation saves only a possible match; rejection is separate and persisted")
            page.locator("#preferred-retailer").select_option("IKEA")
            page.locator("#find-products").click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].state === 'complete' && Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].settings.preferred_retailers[0] === 'IKEA'", 120000)
            assert len(matches()["candidates"]) == 5
            assert matches()["query_embedding_cached"] is True
            passed("Soft retailer preference retains real alternatives and reuses the crop embedding")
            page.locator("#retailer-only").check()
            page.locator("#find-products").click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].state === 'complete' && Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].settings.retailer_only", 120000)
            assert not matches()["candidates"]
            assert "No products" in page.locator("#product-candidates").inner_text()
            shot("retailer-filter-no-match.png")
            passed("Strict absent-retailer filter produces an honest empty result and no verified exact match")
            page.locator("#retailer-only").uncheck()
            page.locator("#preferred-retailer").select_option("")
            page.locator("#find-products").click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].state === 'complete' && Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].settings.preferred_retailers.length === 0", 120000)
            assert matches()["cached"] is True
            passed("Repeated search reuses complete matching results rather than rerunning SigLIP")
            page.locator("#room-setup").evaluate("e => e.open = true")
            page.locator("#design-room-ack").check()
            page.locator("#setup-design-room").click()
            wait("roomPhoto.snapshot().project.room.provenance?.kind === 'approximate_design_workspace'")
            page.locator("#placeholder-ack").check()
            page.locator("#create-placeholder").click()
            wait("roomPhoto.snapshot().project.room.furniture.length === 1")
            item = snap()["project"]["room"]["furniture"][0]
            assert item["product_details"]["candidate"]["product_id"] == first["product_id"]
            assert item["size"] == [2.1, .85, .85]
            assert not item["product_details"]["geometry_applied"]
            page.locator("#three-view").click()
            assert page.evaluate("roomLab.snapshot().status") == "disabled"
            passed("Possible product details attach to the photo-linked 3D placeholder without applying catalog dimensions or enabling physical fit")
            page.locator("#save-project").click()
            page.reload(wait_until="networkidle")
            page.locator("#project-picker").select_option(project_id)
            wait("roomPhoto.snapshot().project?.id === '" + project_id + "'")
            page.locator(f'[data-detection-id="{detection["detection_id"]}"]').click()
            assert len(snap()["project"]["inventory"]) == 1
            assert snap()["project"]["room"]["furniture"][0]["product_details"]["candidate"]["product_id"] == first["product_id"]
            assert matches()["decisions"][second["product_id"]]["decision"] == "reject"
            passed("Save/reopen preserves inventory, rejection history and linked-object metadata")
            with page.expect_download() as download:
                page.locator("#export-project").click()
            download.value.save_as(str(OUT / "matched-product-project.zip"))
            old_x = detection["box"][0]
            page.locator("#box-x1").fill(str(old_x+1))
            page.locator("#update-detection").click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].state === 'stale'")
            refused = page.request.post(f"{BASE}/api/projects/{project_id}/action", data={"type":"product_choice","photo_id":photo["id"],"detection_id":detection["detection_id"],"product_id":first["product_id"],"decision":"confirm"})
            assert refused.status == 400
            assert "review required" in page.locator("#product-inventory").inner_text()
            page.locator("#undo-correction").click()
            wait("Object.values(roomPhoto.snapshot().project.photos[0].product_matches)[0].state === 'complete'")
            passed("Changed observations invalidate stale candidates and the server rejects confirmation until review")
            assert not errors, errors
            report = {"result": "passed", "passed": len(checks), "checks": checks, "page_errors": errors, "seconds": perf_counter()-started, "browser": browser.version, "project_id": project_id, "photo_id": photo["id"], "detection_id": detection["detection_id"], "query_embedding_seconds": initial["embedding_seconds"], "query_search_seconds": initial["search_seconds"], "query_total_seconds": initial["total_seconds"], "query_peak_rss_bytes": initial["peak_process_rss_bytes"], "query_embedding_cached_initially": initial["query_embedding_cached"], "exact_product_identity_verified": False, "user_actions_are_test_preferences_not_ground_truth": True}
            (OUT / "browser-tests.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        except Exception as exc:
            shot("browser-failure.png")
            (OUT / "browser-failure.json").write_text(json.dumps({"checks": checks, "error": str(exc), "page_errors": errors, "state": snap()}, indent=2) + "\n")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
