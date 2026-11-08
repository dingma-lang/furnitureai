"""Focused inventory UI checks without rerunning any model inference."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    output = Path(os.environ.get("INVENTORY_REPORT_DIR", ROOT / "reports/phase6e-product-inventory"))
    saved = json.loads((output / "inventory-browser-tests.json").read_text())
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8771"), wait_until="networkidle")
        page.wait_for_function("window.ownedInventory && window.roomPhoto && window.roomLab")
        page.locator("#open-inventory").click()
        page.locator("#inventory-list").select_option(saved["inventory_item_id"])
        page.locator('[data-inventory-view="' + saved["scene_object_id"] + '"]').first.click()
        page.wait_for_function(
            "roomPhoto.snapshot().project?.id==='"
            + saved["project_id"]
            + "' && roomLab.snapshot().selected==='"
            + saved["scene_object_id"]
            + "'"
        )
        assert not page.locator("#inventory-panel").is_visible()
        page.screenshot(path=str(output / "linked-object-navigation.png"))
        checks.append(
            "Inventory link opens the original project and selects its existing 3D object"
        )
        page.locator("#open-inventory").click()
        before = page.evaluate("Object.keys(ownedInventory.snapshot().library.items).length")
        page.locator("#inventory-new").click()
        page.wait_for_function(
            "Object.keys(ownedInventory.snapshot().library.items).length===" + str(before + 1)
        )
        item_id = page.evaluate("ownedInventory.snapshot().selected")
        name = "Manual demo chair " + item_id[-8:]
        page.locator("#inventory-name").fill(name)
        page.locator("#inventory-category").select_option("chair")
        page.locator("#inventory-notes").fill(
            "AUTOMATED DEMONSTRATION only; no actual household identity or specification evidence"
        )
        page.locator("#inventory-save").click()
        page.wait_for_function(
            "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].category==='chair'"
        )
        assert "unknown" in page.locator("#inventory-identity").inner_text()
        checks.append(
            "Manual furniture creation works without product matching or mandatory dimensions"
        )
        page.locator("#inventory-search").fill(name)
        assert page.locator("#inventory-list option").count() == 1
        page.locator("#inventory-filter-dimensions").select_option("complete")
        assert page.locator("#inventory-list option").count() == 0
        page.locator("#inventory-filter-dimensions").select_option("missing")
        assert page.locator("#inventory-list option").count() == 1
        checks.append("Search and dimensions-present/missing filters reflect actual evidence")
        page.locator("#inventory-documents").evaluate("el=>el.open=true")
        page.locator("#inventory-document-title").fill(
            "Sample link, not a real manufacturer document"
        )
        page.locator("#inventory-document-url").fill("https://example.invalid/specification.pdf")
        page.locator("#inventory-document-permission").fill(
            "Unknown; test link only, never fetched"
        )
        page.locator("#inventory-document-form button").click()
        page.wait_for_function(
            "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].documents.length===1"
        )
        item = page.evaluate(
            "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected]"
        )
        assert item["documents"][0]["retrieval_status"] == "link_only_not_retrieved"
        assert item["identification"]["status"] == "unknown"
        checks.append(
            "Documentation links persist without web extraction or identity verification claims"
        )
        page.locator("#inventory-archive").click()
        page.wait_for_function(
            "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].archived===true"
        )
        assert page.locator("#inventory-list option").count() == 0
        page.locator("#inventory-show-archived").check()
        assert page.locator("#inventory-list option").count() == 1
        page.locator("#inventory-archive").click()
        page.wait_for_function(
            "ownedInventory.snapshot().library.items[ownedInventory.snapshot().selected].archived===false"
        )
        assert page.evaluate("ownedInventory.snapshot().selected") == item_id
        checks.append("Archive/restore preserves owned identity and documentation")
        response = page.request.post(
            os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8771") + "/api/inventory/import", data={"schema_version": 999, "items": {}}
        )
        assert response.status == 400
        response = page.request.get(
            os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8771") + "/api/inventory/" + item_id + "/photos/../../etc/passwd"
        )
        assert response.status in (400, 404)
        checks.append("Unsupported inventory versions and arbitrary private file paths are refused")
        assert not errors
        (output / "inventory-detail-browser-tests.json").write_text(
            json.dumps(
                {
                    "passed": len(checks),
                    "checks": checks,
                    "page_errors": errors,
                    "manual_sample_id": item_id,
                    "no_model_inference": True,
                },
                indent=2,
            )
            + "\n"
        )
        browser.close()
    print({"passed": len(checks)})


if __name__ == "__main__":
    main()
