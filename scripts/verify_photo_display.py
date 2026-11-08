"""Focused final display regression against saved real predictions; no inference."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

SOURCE = ROOT / "reports/phase6b-photo-integration"
OUT = Path(os.environ.get("PHOTO_DISPLAY_REPORT_DIR", SOURCE))
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8767")


def main():
    saved = json.loads((SOURCE / "browser-tests.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1540, "height": 1120})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle")
        page.locator("#project-picker").select_option(saved["project_id"])
        page.wait_for_function("roomPhoto.snapshot().project !== null")
        page.locator("#photo-picker").select_option(saved["photo_id"])
        first = page.evaluate("roomPhoto.snapshot().project.photos[0].reviewed[0].detection_id")
        page.locator(f'[data-detection-id="{first}"]').click()
        assert "Original AI" in page.locator("#detection-info").inner_text()
        assert "not rescored" in page.locator("#detection-info").inner_text()
        assert "reviewed" in page.locator("#photo-boxes").text_content()
        page.screenshot(path=str(OUT / "final-photo-review.png"), full_page=True)
        page.locator("#semantic-toggle").check()
        page.locator("#depth-toggle").check()
        assert not page.locator("#semantic-toggle").is_checked()
        assert "NOT metres" in page.locator("#photo-caption").inner_text()
        page.screenshot(path=str(OUT / "final-relative-depth.png"), full_page=True)
        assert not errors
        (OUT / "display-tests.json").write_text(json.dumps({"passed": 3, "checks": ["Corrected labels explicitly retain original AI score semantics", "Reviewed boxes do not imply rescored confidence", "Depth and surface views are mutually exclusive and depth is not metres"], "page_errors": errors}, indent=2) + "\n")
        browser.close()


if __name__ == "__main__":
    main()
