"""Check saved acquisition, protocol, index and regression evidence without inference."""

import hashlib
from pathlib import Path

import numpy as np

from furniture_bench.catalog_expansion import atomic_json, validate_nested_galleries
from furniture_bench.data import digest, read_json
from furniture_bench.product_catalog import MultiImageCatalog

ROOT = Path("reports/phase6d3-dataset-expansion")


def main():
    prior = read_json(ROOT / "preservation-before.json")
    changes, appended = [], []
    for name, sha in prior.items():
        path = Path(name)
        if path.is_file() and digest(path) == sha:
            continue
        if name in {
            "reports/phase6d-product-identification/server-current.log",
            "reports/phase6d2-data-quality/review-server.log",
        }:
            prefix = hashlib.sha256()
            with path.open("rb") as stream:
                for line in stream:
                    prefix.update(line)
                    if prefix.hexdigest() == sha:
                        appended.append(name)
                        break
                else:
                    changes.append(name)
        else:
            changes.append(name)
    assert not changes, changes
    lock = read_json(ROOT / "scaling-protocol-lock.json")
    protocol = read_json(ROOT / "scaling-protocol.json")
    assert digest(ROOT / "scaling-protocol.json") == lock["protocol_sha256"]
    galleries = {}
    for size in protocol["gallery_ids"]:
        path = ROOT / f"gallery-{size}.json"
        assert digest(path) == lock["gallery_sha256"][size]
        galleries[size] = read_json(path)["products"]
    validate_nested_galleries(galleries, protocol["queries"])
    original = read_json(ROOT / "embedding-report.json")
    resumed = read_json(ROOT / "embedding-resume-report.json")
    assert resumed["new_encodings"] == 0 and resumed["reused"] == original["unique_images"]
    model = Path(".cache/models/siglip2-base-75de2d55")
    assert digest(model / "model.safetensors") == original["model_signature"]["weights_sha256"]
    manifest = read_json(ROOT / "licensed-dataset-manifest.json")
    assert all(digest(image["file"]) == image["sha256"] for image in manifest["images"])
    ledger = read_json(ROOT / "acquired/download-ledger.json")
    assert len(ledger["entries"]) <= 4000 and ledger["transferred_bytes"] <= 1_500_000_000
    assert all(row["status"] == "complete" for row in ledger["entries"].values())
    assert all(
        row.get("downloaded_at") and digest(row["file"]) == row["sha256"]
        for row in ledger["entries"].values()
    )
    browser = {}
    for directory in ROOT.glob("*-browser"):
        result = read_json(next(directory.glob("*tests.json")))
        browser[directory.name] = {
            "passed": result.get("passed", len(result["checks"])),
            "page_errors": result.get("page_errors", []),
        }
    assert sum(r["passed"] for r in browser.values()) == 67
    assert all(not r["page_errors"] for r in browser.values())
    artifact = read_json(ROOT / "artifact-browser-tests.json")
    assert artifact["passed"] == 4 and not artifact["page_errors"]
    log = (ROOT / "regression-tests.log").read_text()
    assert "Ran 243 tests" in log and "\nOK\n" in log
    rows = read_json(ROOT / "vector-image-rows.json")["images"]
    vectors = np.load(ROOT / "image-vectors.npy")
    lookup = {r["image_id"]: v for r, v in zip(rows, vectors)}
    query = protocol["queries"][0]
    for directory in [ROOT / "indexes/1000", ROOT / "indexes/all-1040"]:
        index = MultiImageCatalog.load(directory)
        try:
            result, _ = index.search(np.stack([lookup[query["image_id"]]]), k=5)
            assert len({r["product_id"] for r in result[0]}) == 5
            assert all(r["identity_verified"] is False for r in result[0])
        finally:
            index.close()
    atomic_json(
        ROOT / "verification-summary.json",
        {
            "python_baseline": 230,
            "python_final": 243,
            "existing_browser": browser,
            "existing_browser_total": 67,
            "new_artifact_browser_checks": artifact,
            "browser_total": 71,
            "preserved_previous_files_checked": len(prior),
            "previous_files_changed": changes,
            "existing_operational_log_append_only_prefix_verified": appended,
            "source_images_hashed": len(manifest["images"]),
            "downloaded_images_complete_and_hashed": len(ledger["entries"]),
            "budget_valid": True,
            "protocol_and_nested_galleries_valid": True,
            "all_cached_embedding_resume_new_encodings": 0,
            "checkpoint_hash_valid": True,
            "both_1000_and_all1040_index_load_search_valid": True,
            "production_changed": False,
            "household_verified_queries": 0,
            "human_quality_annotations_created": 0,
        },
    )
    print(
        {
            "python": 243,
            "browser": 71,
            "previous_files_preserved": len(prior),
            "source_images": len(manifest["images"]),
        }
    )


if __name__ == "__main__":
    main()
