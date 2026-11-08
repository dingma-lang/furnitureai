"""Run all prior Room Lab browser suites into a fresh report tree, without overwriting reports."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ROOM_REGRESSION_REPORT_DIR", ROOT / "reports/phase6g-furniture-budget"))
TASKS = [
    ("editor", "verify_room_viewer.py", "ROOM_VIEWER_REPORT_DIR"),
    ("photo", "verify_photo_integration.py", "ROOM_PHOTO_REPORT_DIR"),
    ("photo-to-3d", "verify_photo_to_3d.py", "ROOM_3D_REPORT_DIR"),
    ("product", "verify_product_matching.py", "PRODUCT_REPORT_DIR"),
    ("display", "verify_photo_display.py", "PHOTO_DISPLAY_REPORT_DIR"),
    ("quality-review", "verify_quality_review.py", "QUALITY_REVIEW_REPORT_DIR"),
    ("expansion", "verify_expansion_artifacts.py", "EXPANSION_ARTIFACT_REPORT_DIR"),
    ("inventory", "verify_owned_inventory.py", "INVENTORY_REPORT_DIR"),
    ("inventory-details", "verify_inventory_details.py", "INVENTORY_REPORT_DIR"),
    ("home-transfer", "verify_home_transfer.py", "HOME_TRANSFER_REPORT_DIR"),
    ("budget", "verify_furniture_budget.py", "FURNITURE_BUDGET_REPORT_DIR"),
    ("budget-drafts", "verify_budget_draft.py", "FURNITURE_BUDGET_REPORT_DIR"),
    ("replacement", "verify_replacements.py", "REPLACEMENT_REPORT_DIR"),
    ("replacement-integrity", "verify_replacement_integrity.py", "REPLACEMENT_REPORT_DIR"),
]


def main():
    OUT.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        ROOM_VIEWER_URL=os.environ.get("ROOM_VIEWER_URL", "http://127.0.0.1:8773"),
        PYTHONPYCACHEPREFIX="/private/tmp/furniture-pycache",
        PLAYWRIGHT_BROWSERS_PATH=str(ROOT / ".cache/playwright-recording"),
    )
    start = os.environ.get("ROOM_REGRESSION_START")
    tasks = TASKS
    results = []
    if start:
        names = [t[0] for t in TASKS]
        if start not in names:
            raise ValueError("Unknown browser suite to resume")
        tasks = TASKS[names.index(start) :]
        previous = OUT / "browser-regression-status.json"
        if previous.exists():
            results = [
                r
                for r in json.loads(previous.read_text())
                if r["suite"] in names[: names.index(start)] and r["exit_code"] == 0
            ]
    for name, script, key in tasks:
        folder = OUT / (
            "inventory-browser"
            if name == "inventory-details"
            else "budget-browser"
            if name == "budget-drafts"
            else "replacement-browser"
            if name == "replacement-integrity"
            else name + "-browser"
        )
        folder.mkdir(exist_ok=True)
        with (OUT / (name + "-regression.log")).open("w") as log:
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / script)],
                cwd=ROOT,
                env=dict(env, **{key: str(folder)}),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results.append({"suite": name, "exit_code": result.returncode, "output": str(folder)})
        (OUT / "browser-regression-status.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
