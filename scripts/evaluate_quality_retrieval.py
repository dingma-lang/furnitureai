"""Development-only rejection selection, followed by one locked fresh evaluation."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.dataset_quality import category_condition
from furniture_bench.retrieval import CatalogIndex
from furniture_bench.retrieval_experiments import (
    paired_interval,
    query_ranks,
    score_products,
    summarize,
)

ROOT = Path("reports/phase6d2-data-quality")


def csv_file(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    protocol = read_json(ROOT / "fresh-protocol.json")
    lock = read_json(ROOT / "protocol-lock.json")
    assert digest(ROOT / "fresh-protocol.json") == lock["protocol_sha256"]
    assert digest(ROOT / "fresh-gallery.json") == lock["gallery_sha256"]
    gallery = read_json(ROOT / "fresh-gallery.json")["products"]
    rows = read_json(ROOT / "vector-image-rows.json")["images"]
    v = np.load(ROOT / "image-vectors.npy")
    assert len(v) == len(rows) and len({r["image_id"] for r in rows}) == len(rows)
    by_id = {r["image_id"]: (r["sha256"], vector) for r, vector in zip(rows, v)}

    def vector(row):
        sha, value = by_id[row["image_id"]]
        assert sha == row["sha256"]
        return value

    q = np.stack([vector(row) for row in protocol["queries"]])
    images = [row for p in gallery for row in p["images"]]
    g = np.stack([vector(row) for row in images])
    members, offset = [], 0
    for p in gallery:
        members.append(list(range(offset, offset + len(p["images"]))))
        offset += len(p["images"])
    return protocol, gallery, q, g, members


def rejection(scores, queries, indices, threshold, family):
    ordered = np.sort(scores, axis=1)
    signal = ordered[:, -1] if family == "similarity" else ordered[:, -1] - ordered[:, -2]
    present = [i for i in indices if queries[i]["true_product_in_catalog"]]
    absent = [i for i in indices if not queries[i]["true_product_in_catalog"]]
    known_accept = float(np.mean(signal[present] >= threshold))
    unknown_reject = float(np.mean(signal[absent] < threshold))
    return {
        "family": family,
        "threshold": threshold,
        "known_queries": len(present),
        "unknown_queries": len(absent),
        "known_accept_rate": known_accept,
        "known_false_reject_rate": 1 - known_accept,
        "unknown_reject_rate": unknown_reject,
        "unknown_false_accept_rate": 1 - unknown_reject,
        "balanced_accuracy": (known_accept + unknown_reject) / 2,
        "calibrated_probability": False,
    }


def develop():
    if (ROOT / "unknown-selection-lock.json").exists():
        raise ValueError("Selection already locked")
    protocol, products, q, g, members = load()
    indices = [i for i, row in enumerate(protocol["queries"]) if row["split"] == "development"]
    queries = [protocol["queries"][i] for i in indices]
    scores = score_products(q[indices], g, members, "max")
    grid = []
    selected = {}
    for family, thresholds in [
        ("similarity", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]),
        ("margin", [0, 0.005, 0.01, 0.02, 0.04, 0.08]),
    ]:
        results = [
            rejection(scores, queries, list(range(len(queries))), t, family) for t in thresholds
        ]
        grid.extend(results)
        selected[family] = max(
            results, key=lambda r: (r["balanced_accuracy"], r["known_accept_rate"])
        )
    write_json(ROOT / "unknown-development-grid.json", {"results": grid})
    write_json(
        ROOT / "unknown-selection-lock.json",
        {
            "selected": selected,
            "protocol_sha256": digest(ROOT / "fresh-protocol.json"),
            "heldout_scores_seen": False,
            "production_changed": False,
        },
    )
    print({name: r["threshold"] for name, r in selected.items()})


def final():
    if (ROOT / "fresh-evaluation.json").exists():
        raise ValueError("Final evaluation already saved; do not retune")
    protocol, products, q, g, members = load()
    selection = read_json(ROOT / "unknown-selection-lock.json")
    assert selection["protocol_sha256"] == digest(ROOT / "fresh-protocol.json")
    started = perf_counter()
    scores = score_products(q, g, members, "max")
    pids = [p["product_id"] for p in products]
    queries = protocol["queries"]
    ranks = query_ranks(scores, pids, queries)
    ranking_seconds = perf_counter() - started
    all_metrics, categories = {}, []
    unknown_rows = []
    for split in ["development", "heldout"]:
        ids = [i for i, r in enumerate(queries) if r["split"] == split]
        present = [i for i in ids if queries[i]["true_product_in_catalog"]]
        hits = np.array([ranks[i] <= 5 for i in present], dtype=float)
        rng = np.random.default_rng(6228)
        interval = np.quantile(
            hits[rng.integers(0, len(hits), (5000, len(hits)))].mean(axis=1), [0.025, 0.975]
        ).tolist()
        all_metrics[split] = {
            **summarize(ranks, queries, ids),
            "recall5_bootstrap95ci": interval,
            "sofa": summarize(ranks, queries, [i for i in ids if queries[i]["label"] == "sofa"]),
        }
        for category in sorted({p["label"] for p in products}):
            categories.append(
                {
                    "split": split,
                    "category": category,
                    **summarize(
                        ranks, queries, [i for i in ids if queries[i]["label"] == category]
                    ),
                }
            )
        for family, row in selection["selected"].items():
            unknown_rows.append(
                {"split": split, **rejection(scores, queries, ids, row["threshold"], family)}
            )
    csv_file(ROOT / "unknown-product-analysis.csv", unknown_rows)
    csv_file(ROOT / "fresh-category-metrics.csv", categories)
    detector = read_json(ROOT / "detector-predictions/predictions.json")
    detector_by_id = {r["image_id"]: r for r in detector["images"]}
    subset = [
        i
        for i, row in enumerate(queries)
        if row["image_id"] in protocol["detector_subset_image_ids"]
    ]
    subqueries = [queries[i] for i in subset]
    predictions = []
    for row in subqueries:
        found = max(
            detector_by_id[row["image_id"]]["detections"], key=lambda d: d["score"], default={}
        )
        predictions.append({"label": found.get("label"), "score": found.get("score", 0)})
    truth = [{"label": row["label"]} for row in subqueries]
    wrong = [
        {
            "label": sorted({p["label"] for p in products})[
                (sorted({p["label"] for p in products}).index(row["label"]) + 1)
                % len({p["label"] for p in products})
            ],
            "confirmed_by": "synthetic_wrong_category",
            "evidence_source": "user_confirmation",
        }
        for row in subqueries
    ]
    scenarios = [
        ("global", "global", None),
        ("automatic_strict_diagnostic", "predicted_strict_diagnostic", None),
        ("automatic_soft0.08", "predicted_soft", None),
        ("catalog_category_oracle_simulated_confirmation", "oracle", truth),
        ("incorrect_confirmed_category_synthetic", "confirmed", wrong),
        ("unconfirmed_category_global_fallback", "confirmed", truth),
    ]
    category_rows = []
    base_ranks = query_ranks(scores[subset], pids, subqueries)
    for name, mode, evidence in scenarios:
        start = perf_counter()
        sc = category_condition(
            scores[subset], [p["label"] for p in products], predictions, evidence, mode
        )
        rr = query_ranks(sc, pids, subqueries)
        seconds = perf_counter() - start
        for split in ["development", "heldout"]:
            ids = [i for i, row in enumerate(subqueries) if row["split"] == split]
            delta = paired_interval(base_ranks, rr, ids, seed=6228)
            category_rows.append(
                {
                    "split": split,
                    "scenario": name,
                    **summarize(rr, subqueries, ids),
                    "sofa_recall5": summarize(
                        rr, subqueries, [i for i in ids if subqueries[i]["label"] == "sofa"]
                    )["recall_at_5"],
                    "recall5_delta95ci_low": delta[0],
                    "recall5_delta95ci_high": delta[1],
                    "cached_reranking_seconds_all_subset": seconds,
                    "actual_human_confirmations": 0,
                    "identity_scope": "catalog_listing",
                    "not_full_household_pipeline_accuracy": True,
                }
            )
    csv_file(ROOT / "category-retrieval-analysis.csv", category_rows)
    # Independently check FAISS FlatIP against the production cosine definition.
    index = CatalogIndex.build(
        g,
        [
            {"product_id": f"image-{n}", "label": "reference", "file": "not-used"}
            for n in range(len(g))
        ],
        read_json(ROOT / "inference-report.json")["model"],
    )
    try:
        fs, fi = index.index.search(q, len(g))
    finally:
        index.close()
    dot = np.einsum("qd,gd->qg", q, g, optimize=False)
    write_json(
        ROOT / "fresh-evaluation.json",
        {
            "metrics": all_metrics,
            "query_ranks": ranks,
            "queries": queries,
            "product_ids": pids,
            "top5": [[pids[j] for j in np.argsort(-r, kind="stable")[:5]] for r in scores],
            "top1_similarity": scores.max(axis=1).tolist(),
            "top1_top2_margin": (
                np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
            ).tolist(),
            "ranking_seconds": ranking_seconds,
            "faiss_vs_cosine_max_error": float(
                np.abs(fs - np.take_along_axis(dot, fi, axis=1)).max()
            ),
            "category_diagnostics": category_rows,
            "actual_detector_predictions": predictions,
            "detector_query_image_ids": [q["image_id"] for q in subqueries],
            "unknown_rejection": unknown_rows,
            "protocol_sha256": digest(ROOT / "fresh-protocol.json"),
            "unknown_selection_sha256": digest(ROOT / "unknown-selection-lock.json"),
            "production_changed": False,
            "household_verified_queries": 0,
        },
    )
    print(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "final"])
    args = parser.parse_args()
    (develop if args.stage == "develop" else final)()
