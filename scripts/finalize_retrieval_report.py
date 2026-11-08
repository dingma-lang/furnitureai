"""Report-only summaries from frozen vectors; never infer or select parameters."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.retrieval_experiments import (
    align_image_rows,
    category_scores,
    query_ranks,
    score_products,
    summarize,
)


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    protocol = read_json(root / "protocol.json")
    lock = read_json(root / "protocol-lock.json")
    assert digest(root / "protocol.json") == lock["protocol_sha256"]
    assert digest(root / "gallery.json") == lock["gallery_sha256"]
    final = read_json(root / "final-evaluation-aligned.json")
    assert final["selection_lock_sha256"] == digest(root / "selection-lock.json")
    assert final["category_selection_lock_sha256"] == digest(root / "category-selection-lock.json")
    products = read_json(root / "gallery.json")["products"]
    pids = [p["product_id"] for p in products]
    members, offset = [], 0
    for product in products:
        n = len(product["images"])
        members.append(list(range(offset, offset + n)))
        offset += n
    original = np.load(root / "embeddings/original.npy")
    queries = [
        q
        for q in protocol["queries"]
        if q["image_id"] in protocol["predicted_category_query_image_ids"]
    ]
    indices = [protocol["queries"].index(q) for q in queries]
    base = score_products(original[offset:][indices], original[:offset], members, "main_only")
    base_ranks = query_ranks(base, pids, queries)
    condition = read_json(root / "conditions/report.json")
    detector = read_json(root / "detector-predictions/predictions.json")
    detections = {r["image_id"]: r["detections"] for r in detector["images"]}
    predictions = []
    for q in queries:
        best = max(detections[q["image_id"]], key=lambda d: d["score"], default={})
        predictions.append({"label": best.get("label"), "score": best.get("score", 0)})
    categories = [p["label"] for p in products]
    rows = []

    def record(name, scores, seconds, inference_seconds=0):
        ranks = query_ranks(scores, pids, queries)
        for split in ["development", "heldout"]:
            subset = [i for i, q in enumerate(queries) if q["split"] == split]
            sofa = [i for i in subset if queries[i]["label"] == "sofa"]
            baseline = summarize(base_ranks, queries, subset)
            m = summarize(ranks, queries, subset)
            rows.append(
                {
                    "split": split,
                    "configuration": name,
                    "query_count": m["queries"],
                    "catalog_products": len(products),
                    "baseline_recall1": baseline["recall_at_1"],
                    "recall1": m["recall_at_1"],
                    "baseline_recall5": baseline["recall_at_5"],
                    "recall5": m["recall_at_5"],
                    "sofa_query_count": len(sofa),
                    "sofa_recall5": summarize(ranks, queries, sofa)["recall_at_5"],
                    "cached_matrix_seconds_all_27": seconds,
                    "inference_seconds_all_27": inference_seconds,
                    "peak_condition_embedding_process_rss_bytes": condition[
                        "peak_process_rss_bytes"
                    ],
                    "selected_for_app": False,
                }
            )

    for name, prediction, mode in [
        ("global", predictions, "none"),
        (
            "verified-category-oracle",
            [{"label": q["label"], "score": 1} for q in queries],
            "oracle",
        ),
        ("actual-predicted-strict", predictions, "strict"),
        ("actual-predicted-soft-0.08", predictions, "soft"),
    ]:
        start = perf_counter()
        scores = base if mode == "none" else category_scores(base, categories, prediction, mode)[0]
        record(name, scores, perf_counter() - start)
    modes = ["detector_bbox", "detector_margin10"] + [
        "synthetic_" + kind
        for kind in [
            "lighting",
            "shadow",
            "occlusion",
            "color",
            "synthetic_stain",
            "synthetic_wear",
            "background_clutter",
        ]
    ]
    for name in modes:
        vectors = align_image_rows(
            np.load(root / "conditions" / (name + ".npy")),
            condition["query_image_ids"],
            [q["image_id"] for q in queries],
        )
        start = perf_counter()
        scores = score_products(vectors, original[:offset], members, "main_only")
        record(name, scores, perf_counter() - start, condition["inference_seconds"][name])
    with (root / "supplemental-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    variants = read_json(root / "embeddings/report.json")
    ablation = root / "ablation-study.md"
    marker = "\n## Supplemental paired comparisons\n"
    text = ablation.read_text().split(marker)[0] + marker
    text += "\nAll rows use the frozen first-view proposal as comparator on the same preselected subset. Sofa counts are only two per split. These diagnostics do not select a new held-out method.\n\n"
    text += "| Split | Configuration | N | Baseline R@1 / R@5 | New R@1 / R@5 | Sofa R@5 |\n|---|---|---:|---:|---:|---:|\n"
    for row in rows:
        text += (
            f"| {row['split']} | {row['configuration']} | {row['query_count']} | "
            f"{row['baseline_recall1']:.3f} / {row['baseline_recall5']:.3f} | "
            f"{row['recall1']:.3f} / {row['recall5']:.3f} | {row['sofa_recall5']:.3f} |\n"
        )
    text += "\nThe CSV separates model inference, cached-matrix scoring, and shared process peak RSS. Zero model inference for category reranking means reuse of existing embeddings; OWLv2 preprocessing cost is recorded separately in runtime-costs.json. Synthetic background clutter here is a colored surround, not a realistic composited room. Camera-angle changes have no measured ground truth.\n"
    ablation.write_text(text)
    comparison = root / "baseline-comparison.csv"
    with comparison.open() as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        comparison_rows = [r for r in reader if r["split"] != "historical-examined"]
    historical = dict.fromkeys(fieldnames, "")
    historical.update(
        {
            "split": "historical-examined",
            "configuration": "Phase6D-original-different-dataset",
            "query_count": 119,
            "catalog_products": 120,
            "baseline_recall1": 60 / 119,
            "recall1": 60 / 119,
            "baseline_recall5": 80 / 119,
            "recall5": 80 / 119,
            "sofa_recall5": 0.4,
        }
    )
    with comparison.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([historical] + comparison_rows)
    runtime = {
        "variant_job": variants,
        "condition_job": {
            k: condition[k]
            for k in ["total_seconds", "inference_seconds", "peak_process_rss_bytes"]
        },
        "detector_timing_seconds_sum": {},
        "detector_errors": [r["errors"] for r in detector["images"] if r["errors"]],
        "final_scoring_seconds_all_140": final["ranking_seconds"],
        "definitions": "Model encode timings exclude load. Job wall time includes load/preprocessing. Peak macOS ru_maxrss is bytes for whole process, shared across configurations. Matrix scoring excludes inference, browser, worker startup, and IPC. OWLv2 peak memory was not separately sampled.",
    }
    for image in detector["images"]:
        for key, value in image["timing_seconds"].items():
            if isinstance(value, (int, float)):
                runtime["detector_timing_seconds_sum"][key] = (
                    runtime["detector_timing_seconds_sum"].get(key, 0) + value
                )
    write_json(root / "runtime-costs.json", runtime)
    browser = {}
    for directory in sorted(root.glob("*-browser")):
        report = read_json(next(directory.glob("*tests.json")))
        browser[directory.name] = {
            "passed": report.get("passed", len(report["checks"])),
            "page_errors": report.get("page_errors", []),
        }
    assert sum(r["passed"] for r in browser.values()) == 59
    assert all(not r["page_errors"] for r in browser.values())
    log = (root / "regression-tests.log").read_text()
    assert "Ran 214 tests" in log and "\nOK\n" in log
    old_products = read_json("data/catalogs/abo-phase6d/selected-products.json")["products"]
    new_products = read_json("data/catalogs/abo-phase6d1/selected-products.json")["products"]
    old_hashes = {i["sha256"] for p in old_products for i in p["images"]}
    new_hashes = {i["sha256"] for p in new_products for i in p["images"]}
    assert not old_hashes & new_hashes
    assert all(
        digest(Path("data/catalogs/abo-phase6d1") / i["file"]) == i["sha256"]
        for p in new_products
        for i in p["images"]
    )
    assert all(
        digest(Path("data/catalogs/abo-phase6d") / i["file"]) == i["sha256"]
        for p in old_products
        for i in p["images"]
    )
    verification = {
        "python_tests": 214,
        "browser_checks": browser,
        "browser_total": 59,
        "frozen_locks_valid": True,
        "historical_and_fresh_source_image_hashes_valid": True,
        "historical_fresh_sha256_overlap": 0,
        "production_ranking_changed": False,
        "heldout_adoption": final["adopt_in_app"],
        "canonical_evaluation": "final-evaluation-aligned.json",
        "historical_report_current_sha256": digest(
            Path("reports/phase6d-product-identification/RESULTS.md")
        ),
        "preservation_scope": "Previous reports retained; source image hashes checked against saved manifests. No pre-phase code hash inventory existed, so no whole-repository bytewise equality assertion is made.",
    }
    write_json(root / "verification-summary.json", verification)
    print(
        {
            "python": 214,
            "browser": 59,
            "supplemental_rows": len(rows),
            "adopted": final["adopt_in_app"],
        }
    )


if __name__ == "__main__":
    main()
