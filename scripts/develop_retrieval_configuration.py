"""Development-only selection. Held-out vectors are not scored by this script."""

from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.retrieval_experiments import (
    category_scores,
    query_ranks,
    score_products,
    summarize,
)


def objective(row):
    return (
        (row["metrics"]["recall_at_5"] + row["sofa"]["recall_at_5"]) / 2,
        row["metrics"]["recall_at_1"],
    )


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    if (root / "selection-lock.json").exists():
        raise ValueError("Selection already frozen; do not retune")
    protocol = read_json(root / "protocol.json")
    assert (
        digest(root / "protocol.json") == read_json(root / "protocol-lock.json")["protocol_sha256"]
    )
    catalog = read_json(root / "gallery.json")["products"]
    queries = protocol["queries"]
    ids = [i for i, q in enumerate(queries) if q["split"] == "development"]
    qs = [queries[i] for i in ids]
    product_ids = [p["product_id"] for p in catalog]
    members = []
    offset = 0
    for p in catalog:
        members.append(list(range(offset, offset + len(p["images"]))))
        offset += len(p["images"])
    matrices = {
        mode: np.load(root / "embeddings" / f"{mode}.npy")
        for mode in ["original", "white_margin_trim", "margin10", "letterbox"]
    }
    rows = []
    saved = {}
    configs = [
        ("original", "max"),
        ("original", "normalized_centroid"),
        ("original", "top2_mean"),
        ("original", "main_only"),
        ("white_margin_trim", "max"),
        ("margin10", "max"),
        ("letterbox", "max"),
    ]

    def evaluate(mode, aggregation, name=None, pool=None):
        name = name or mode + "+" + aggregation
        before = perf_counter()
        a = matrices[mode]
        scores = score_products(a[offset:][ids], a[:offset], members, aggregation, pool)
        ranks = query_ranks(scores, product_ids, qs)
        result = {
            "name": name,
            "preprocessing": mode,
            "aggregation": aggregation,
            "metrics": summarize(ranks, qs, range(len(qs))),
            "sofa": summarize(ranks, qs, [i for i, q in enumerate(qs) if q["label"] == "sofa"]),
            "retrieval_seconds": perf_counter() - before,
            "ranks": ranks,
        }
        rows.append(result)
        saved[name] = scores
        return result

    for mode, aggregation in configs:
        evaluate(mode, aggregation)
    best_pre = max([r for r in rows if r["aggregation"] == "max"], key=objective)
    best_agg = max([r for r in rows if r["preprocessing"] == "original"], key=objective)
    evaluate(
        best_pre["preprocessing"], best_agg["aggregation"], name="development-selected-combination"
    )
    chosen = max(rows, key=objective)
    for pool in [10, 40]:
        evaluate(chosen["preprocessing"], chosen["aggregation"], f"pool-{pool}", pool)
    global_scores = saved[chosen["name"]]
    categories = [p["label"] for p in catalog]
    oracle, _ = category_scores(
        global_scores, categories, [{"label": q["label"], "score": 1} for q in qs], "oracle"
    )
    r = query_ranks(oracle, product_ids, qs)
    category_rows = [
        {
            "name": "verified-category-oracle-diagnostic",
            "metrics": summarize(r, qs, range(len(qs))),
            "sofa": summarize(r, qs, [i for i, q in enumerate(qs) if q["label"] == "sofa"]),
        }
    ]
    # Keep unknown score tuning separate from exact-identity verification.
    top = global_scores.max(axis=1)
    present = np.array([q["true_product_in_catalog"] for q in qs])
    thresholds = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]:
        accepted = top >= threshold
        known = float(accepted[present].mean())
        unknown = float((~accepted[~present]).mean())
        thresholds.append(
            {
                "threshold": threshold,
                "known_accept_rate": known,
                "unknown_reject_rate": unknown,
                "balanced_accuracy": (known + unknown) / 2,
            }
        )
    unknown = max(thresholds, key=lambda r: (r["balanced_accuracy"], r["known_accept_rate"]))
    selection = {
        "preprocessing": chosen["preprocessing"],
        "aggregation": chosen["aggregation"],
        "development_name": chosen["name"],
        "selection_objective": protocol["selection_objective"],
        "development_metrics": chosen["metrics"],
        "development_sofa": chosen["sofa"],
        "unknown_threshold_experimental": unknown["threshold"],
        "unknown_threshold_tradeoffs": thresholds,
        "category_policy": "pending actual development detector subset; no held-out tuning",
        "heldout_scores_inspected": False,
        "metadata_ranking": "unavailable query brands/specs not used",
        "production_change": "pending predeclared heldout adoption gate",
    }
    write_json(
        root / "development-results.json",
        {
            "individual_and_combination": rows,
            "oracle_diagnostic": category_rows,
            "queries": qs,
            "product_ids": product_ids,
        },
    )
    write_json(root / "selection-lock.json", selection)
    print(selection)


if __name__ == "__main__":
    main()
