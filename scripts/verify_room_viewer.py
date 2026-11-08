"""Real Chrome interaction checks; no model inference or browser download."""
import json
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8766")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ROOM_VIEWER_REPORT_DIR", ROOT / "reports/phase6-interactive-prototype"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    checks, errors = [], []
    start = perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        def state():
            return page.evaluate("window.roomLab.snapshot()")
        def wait(expression):
            page.wait_for_function("window.roomLab && (" + expression + ")")
        def passed(name):
            checks.append(name)
        try:
            page.goto(BASE_URL, wait_until="networkidle")
            wait("roomLab.snapshot().objects === 5")
            assert state()["status"] == "valid"
            initial = state()["scene"]
            assert page.locator("#position-y").evaluate("input => input.checkValidity()")
            (OUT / "browser-before.json").write_text(json.dumps(initial, indent=2) + "\n")
            page.screenshot(path=str(OUT / "browser-before.png"), full_page=True)
            passed("Known-dimension demo renders five independent objects with valid feedback")
            page.locator("#top").click()
            wait("roomLab.snapshot().top")
            page.wait_for_timeout(300)
            position = page.evaluate("roomLab.projectObject('chair')")
            page.mouse.click(position["x"], position["y"])
            wait("roomLab.snapshot().selected === 'chair'")
            passed("Actual canvas ray picking selects chair in orthographic top-down view")
            page.locator("#perspective").click()
            page.wait_for_timeout(300)
            before_camera = page.evaluate("roomLab.camera()")
            rect = page.locator("canvas").bounding_box()
            x, y = rect["x"] + 60, rect["y"] + 100
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 100, y + 40, steps=10)
            page.mouse.up()
            page.wait_for_timeout(400)
            assert page.evaluate("roomLab.camera()") != before_camera
            passed("Pointer orbit changes perspective camera")
            old = page.evaluate("roomLab.camera()")
            page.mouse.wheel(0, -250)
            page.wait_for_timeout(400)
            assert old != page.evaluate("roomLab.camera()")
            passed("Wheel zoom changes camera")
            old = page.evaluate("roomLab.camera().target")
            page.mouse.move(x, y)
            page.mouse.down(button="right")
            page.mouse.move(x + 70, y + 20, steps=8)
            page.mouse.up(button="right")
            page.wait_for_timeout(400)
            assert old != page.evaluate("roomLab.camera().target")
            passed("Right-drag pan changes camera target")
            page.locator("#frame").click()
            page.locator('[data-object-id="sofa-original"]').click()
            page.locator("#replace").click()
            wait("roomLab.snapshot().history === 1")
            assert state()["status"] == "restricted"
            replacement = next(f for f in state()["scene"]["furniture"] if f["id"] == "sofa-original")
            assert replacement["size"] == [2.8, 1.05, .9]
            assert " m" in page.locator("#clearances").inner_text()
            for field in ("walls", "floor", "openings", "dimensions"):
                assert state()["scene"][field] == initial[field]
            passed("Sofa replacement preserves room geometry and shows restricted clearance distances")
            page.screenshot(path=str(OUT / "browser-after.png"), full_page=True)
            with page.expect_download() as download:
                page.locator("#export").click()
            download.value.save_as(str(OUT / "browser-after.json"))
            passed("Editable room JSON exports through browser download")
            page.locator("#before").click()
            assert page.locator("#size-x").input_value() == "2.1"
            assert page.locator("#apply").is_disabled()
            page.locator("#after").click()
            assert page.locator("#size-x").input_value() == "2.8"
            passed("Before/current comparison preserves a read-only original snapshot")
            page.locator("#size-x").fill("9")
            page.locator("#apply").click()
            wait("roomLab.snapshot().status === 'invalid'")
            page.screenshot(path=str(OUT / "browser-invalid.png"), full_page=True)
            page.locator("#undo").click()
            wait("roomLab.snapshot().status === 'restricted'")
            passed("Oversized object produces invalid draft; undo restores previous geometry")
            page.locator("#position-y").fill("2.45")
            page.locator("#rotation").fill("30")
            page.locator("#apply").click()
            wait("roomLab.snapshot().status === 'invalid'")
            assert next(f for f in state()["scene"]["furniture"] if f["id"] == "sofa-original")["rotation_degrees"] == 30
            page.locator("#undo").click()
            passed("Numeric movement/rotation reuse collision checks")
            page.locator("#room-width").fill("7")
            page.locator("#resize").click()
            wait("roomLab.snapshot().scene.dimensions[0] === 7")
            page.locator("#undo").click()
            assert state()["scene"]["dimensions"][0] == 6
            passed("Room resizing updates geometry and can be undone")
            page.locator('[data-object-id="chair"]').click()
            page.locator("#remove").click()
            wait("roomLab.snapshot().objects === 4")
            page.locator("#undo").click()
            wait("roomLab.snapshot().objects === 5")
            passed("Independent furniture removal and undo")
            page.locator("#reset").click()
            wait("roomLab.snapshot().history === 0 && roomLab.snapshot().status === 'valid'")
            page.locator("#top").click()
            page.locator("#drag").click()
            page.wait_for_timeout(300)
            pos = page.evaluate("roomLab.projectObject('sofa-original')")
            page.mouse.move(pos["x"], pos["y"])
            page.mouse.down()
            page.mouse.move(pos["x"] + 35, pos["y"] + 20, steps=10)
            page.mouse.up()
            wait("roomLab.snapshot().history === 1")
            assert next(f for f in state()["scene"]["furniture"] if f["id"] == "sofa-original")["position"] != [3, 3.72, 0]
            passed("Real canvas dragging moves furniture and checks placement on release")
            page.locator("#reset").click()
            assert state()["scene"] == initial
            passed("Reset restores original complete room document")
            for object_id, category in (("chair", "chair"), ("coffee-table", "table")):
                page.locator(f'[data-object-id="{object_id}"]').click()
                page.locator("#replace").click()
                wait("roomLab.snapshot().history === 1")
                assert next(f for f in state()["scene"]["furniture"] if f["id"] == object_id)["prototype_variant"] == "replacement"
                passed(f"Selectable {category} can be replaced independently")
                page.locator("#reset").click()
            page.locator("#scene-source").select_option("phase52")
            wait("roomLab.snapshot().observedPoints === 4951")
            assert state()["editable"] is False and state()["status"] == "disabled"
            assert page.locator("#apply").is_disabled() and page.locator("#resize").is_disabled()
            assert "checks disabled" in page.locator("#clearances").inner_text()
            refused = page.request.post(BASE_URL + "/api/edit", data={"scene": state()["scene"], "action": {"type": "resize", "dimensions": [6, 5, 3]}})
            assert refused.status == 400
            passed("Server refuses partial-scene placement even when a direct API request bypasses UI")
            page.screenshot(path=str(OUT / "browser-partial.png"), full_page=True)
            passed("Phase 5.2 observed cloud loads with metric editing and placement disabled")
            page.locator("#scene-source").select_option("phase51")
            wait("roomLab.snapshot().observedPoints === 20659")
            page.wait_for_timeout(300)
            page.screenshot(path=str(OUT / "browser-phase51.png"), full_page=True)
            passed("Phase 5.1 point cloud and existing observed surface mesh render")
            page.locator("#scene-source").select_option("single-photo")
            wait("roomLab.snapshot().observedPoints === 0 && roomLab.snapshot().scene.schema_version === 2")
            assert page.locator("#empty").is_visible()
            passed("Single-photo unresolved geometry stays empty rather than becoming an invented room")
            page.locator("#import").set_input_files({"name": "demo.json", "mimeType": "application/json", "buffer": json.dumps(initial).encode()})
            wait("roomLab.snapshot().scene.schema_version === 1")
            assert state()["scene"] == initial
            passed("Existing room JSON imports through real file input")
            page.locator("#import").set_input_files({"name": "bad.json", "mimeType": "application/json", "buffer": b'{broken'})
            assert page.locator("#error").is_visible()
            assert state()["scene"] == initial
            passed("Malformed import reports an error and preserves current scene")
            assert not errors, errors
            report = {"passed": len(checks), "checks": checks, "page_errors": errors, "seconds": perf_counter() - start, "browser": browser.version, "renderer": "Headless Chrome / SwiftShader WebGL; no real-GPU performance claim", "result": "passed"}
            (OUT / "browser-tests.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        except Exception as exc:
            page.screenshot(path=str(OUT / "browser-failure.png"), full_page=True)
            (OUT / "browser-failure.json").write_text(json.dumps({"completed_checks": checks, "error": str(exc), "page_errors": errors}, indent=2) + "\n")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
