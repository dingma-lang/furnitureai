"""Choose a small category policy on actual development detections only."""

from pathlib import Path

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.retrieval_experiments import (
    category_scores,
    query_ranks,
    score_products,
    summarize,
)


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    if (root / "category-selection-lock.json").exists():
        raise ValueError("Do not reselect category policy")
    protocol = read_json(root / "protocol.json")
    chosen = read_json(root / "selection-lock.json")
    products = read_json(root / "gallery.json")["products"]
    all_queries = protocol["queries"]
    subset = [
        (i, q)
        for i, q in enumerate(all_queries)
        if q["split"] == "development"
        and q["image_id"] in protocol["predicted_category_query_image_ids"]
    ]
    queries = [q for _, q in subset]
    detector = {
        r["image_id"]: r
        for r in read_json(root / "detector-predictions/predictions.json")["images"]
    }
    predictions = []
    for q in queries:
        rows = detector[q["image_id"]]["detections"]
        best = max(rows, key=lambda d: d["score"]) if rows else {}
        predictions.append({"label": best.get("label"), "score": best.get("score", 0)})
    members = []
    n = 0
    for p in products:
        members.append(list(range(n, n + len(p["images"]))))
        n += len(p["images"])
    vectors = np.load(root / "embeddings" / f"{chosen['preprocessing']}.npy")
    scores = score_products(
        vectors[n:][[i for i, _ in subset]], vectors[:n], members, chosen["aggregation"]
    )
    ids = [p["product_id"] for p in products]
    categories = [p["label"] for p in products]
    rows = []
    for mode, weight in [("none", 0), ("strict", 0), ("soft", 0.03), ("soft", 0.08)]:
        s, fallback = (
            (scores, 0)
            if mode == "none"
            else category_scores(scores, categories, predictions, mode, weight)
        )
        ranks = query_ranks(s, ids, queries)
        rows.append(
            {
                "mode": mode,
                "weight": weight,
                "fallbacks": fallback,
                "metrics": summarize(ranks, queries, range(len(queries))),
                "sofa": summarize(
                    ranks, queries, [i for i, q in enumerate(queries) if q["label"] == "sofa"]
                ),
            }
        )
    best = max(
        rows,
        key=lambda r: (
            (r["metrics"]["recall_at_5"] + r["sofa"]["recall_at_5"]) / 2,
            r["metrics"]["recall_at_1"],
        ),
    )
    write_json(
        root / "category-selection-lock.json",
        {
            "selected": best,
            "development_rows": rows,
            "development_queries": queries,
            "actual_predictions": predictions,
            "correct_category_predictions": sum(
                p["label"] == q["label"] for p, q in zip(predictions, queries)
            ),
            "query_count": len(queries),
            "minimum_confidence": 0.35,
            "confidence_is_calibrated_probability": False,
            "heldout_evaluated": False,
            "base_selection_sha256": digest(root / "selection-lock.json"),
        },
    )
    print(best)


if __name__ == "__main__":
    main()
