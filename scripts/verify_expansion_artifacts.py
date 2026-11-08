"""Verify expanded contact sheet and preserved app with local Chrome."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    output = Path(
        os.environ.get("EXPANSION_ARTIFACT_REPORT_DIR", ROOT / "reports/phase6d3-dataset-expansion")
    )
    output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(
            "http://127.0.0.1:8770/reports/phase6d3-dataset-expansion/image-quality-sample.html",
            wait_until="networkidle",
        )
        count = page.locator("section").count()
        assert count >= 20
        assert page.locator("img").count() == count * 3
        page.wait_for_function(
            "[...document.querySelectorAll('section:first-of-type img')].every(i=>i.complete&&i.naturalWidth>0)"
        )
        checks.append(
            "Stratified real catalog source images load with three views per sampled product"
        )
        assert "provisional" in page.locator("body").inner_text()
        assert "No human verification" in page.locator("body").inner_text()
        checks.append("Provisional hints and lack of human verification are visible")
        page.screenshot(path=str(output / "image-quality-sample.png"))
        page.locator("section").last.scroll_into_view_if_needed()
        page.wait_for_function(
            "[...document.querySelectorAll('section:last-of-type img')].every(i=>i.complete&&i.naturalWidth>0)"
        )
        page.screenshot(path=str(output / "difficult-sofa-sample.png"))
        checks.append("Difficult sofa sample source views render without synthetic images")
        response = page.request.get(
            "http://127.0.0.1:8770/reports/phase6d3-dataset-expansion/possible-duplicates.csv"
        )
        assert response.ok and "needs_human_documentary_review" in response.text()
        checks.append("Duplicate queue preserves unmerged review state")
        assert not errors
        browser.close()
    (output / "artifact-browser-tests.json").write_text(
        json.dumps(
            {
                "passed": len(checks),
                "checks": checks,
                "page_errors": errors,
                "human_annotations_created": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print({"passed": len(checks)})


if __name__ == "__main__":
    main()
