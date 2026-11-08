"""Exercise the review page without pretending an automated test is human review."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/browser-tools"))
from playwright.sync_api import sync_playwright  # noqa: E402

from furniture_bench.data import read_json  # noqa: E402
from furniture_bench.dataset_quality import apply_reviews  # noqa: E402


def main():
    source = ROOT / "reports/phase6d2-data-quality"
    out = Path(os.environ.get("QUALITY_REVIEW_REPORT_DIR", source))
    out.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(
            "http://127.0.0.1:8770/reports/phase6d2-data-quality/image-review.html",
            wait_until="networkidle",
        )
        assert page.evaluate("datasetReview.count") == 1140
        assert page.locator(".card").count() == 20
        page.wait_for_function("document.querySelector('.card img').naturalWidth > 0")
        checks.append("Real licensed images and provisional labels render with bounded pagination")
        page.screenshot(path=str(out / "image-review-before.png"), full_page=True)
        first = page.locator(".card").first.get_attribute("data-image-id")
        page.locator("#next").click()
        assert page.locator(".card").first.get_attribute("data-image-id") != first
        page.locator("#previous").click()
        assert page.locator(".card").first.get_attribute("data-image-id") == first
        checks.append("Pagination preserves source identity")
        page.locator("#search").fill(first)
        assert page.locator(".card").count() == 1
        checks.append("Image-ID search finds the correct record")
        page.locator(".card select").nth(0).select_option("whole_object")
        page.locator(".card select").nth(1).select_option("yes")
        assert "not human-verified" in page.locator("#message").inner_text()
        checks.append("Edits are explicitly unverified local drafts")
        page.reload(wait_until="networkidle")
        page.locator("#search").fill(first)
        assert page.locator(".card select").nth(0).input_value() == "whole_object"
        assert page.locator(".card select").nth(1).input_value() == "yes"
        checks.append("Drafts persist after reload without modifying source annotations")
        with page.expect_download() as pending:
            page.locator("#export").click()
        pending.value.save_as(str(out / "automated-unverified-review-draft.json"))
        patch = read_json(out / "automated-unverified-review-draft.json")
        assert patch["reviews"] and not patch["reviews"][0]["attested_human_review"]
        try:
            apply_reviews(read_json(source / "licensed-dataset-manifest.json"), patch)
        except ValueError:
            pass
        else:
            raise AssertionError("Automated draft must not be accepted as human review")
        checks.append("Exported automation draft is rejected by human-review importer")
        page.screenshot(path=str(out / "image-review-after-draft.png"), full_page=True)
        page.locator("#clear").click()
        assert page.evaluate("datasetReview.patch().reviews.length") == 0
        checks.append("Clearing drafts restores the original provisional annotation")
        page.locator("#search").fill("")
        page.locator("#filter").select_option("material_closeup")
        assert "98 images" in page.locator("#status").inner_text()
        checks.append("Quality filtering shows provisional source counts")
        assert not errors
        (out / "review-browser-tests.json").write_text(
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
        browser.close()
    print({"passed": len(checks), "human_annotations_created": 0})


if __name__ == "__main__":
    main()
