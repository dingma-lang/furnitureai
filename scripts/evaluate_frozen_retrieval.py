"""One final held-out evaluation after both development choices are locked."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.retrieval import CatalogIndex
from furniture_bench.retrieval_experiments import (
    align_image_rows,
    category_scores,
    paired_interval,
    query_ranks,
    score_products,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="final-evaluation.json")
    args = parser.parse_args()
    root = Path("reports/phase6d1-retrieval-optimization")
    if (root / args.output).exists():
        raise ValueError("Final held-out results already saved; no retuning")
    protocol = read_json(root / "protocol.json")
    chosen = read_json(root / "selection-lock.json")
    category_choice = read_json(root / "category-selection-lock.json")
    assert digest(root / "selection-lock.json") == category_choice["base_selection_sha256"]
    products = read_json(root / "gallery.json")["products"]
    queries = protocol["queries"]
    pids = [p["product_id"] for p in products]
    categories = [p["label"] for p in products]
    members = []
    offset = 0
    for p in products:
        members.append(list(range(offset, offset + len(p["images"]))))
        offset += len(p["images"])
    original = np.load(root / "embeddings/original.npy")
    variant = np.load(root / "embeddings" / f"{chosen['preprocessing']}.npy")
    before = perf_counter()
    baseline = score_products(original[offset:], original[:offset], members, "max")
    baseline_seconds = perf_counter() - before
    before = perf_counter()
    selected = score_products(variant[offset:], variant[:offset], members, chosen["aggregation"])
    selected_seconds = perf_counter() - before
    br, sr = query_ranks(baseline, pids, queries), query_ranks(selected, pids, queries)
    dev = [i for i, q in enumerate(queries) if q["split"] == "development"]
    held = [i for i, q in enumerate(queries) if q["split"] == "heldout"]
    present = [i for i in held if queries[i]["true_product_in_catalog"]]
    sofa = [i for i in present if queries[i]["label"] == "sofa"]
    interval = paired_interval(br, sr, present)
    final = {
        "baseline": summarize(br, queries, held),
        "selected": summarize(sr, queries, held),
        "baseline_sofa": summarize(br, queries, sofa),
        "selected_sofa": summarize(sr, queries, sofa),
        "paired_recall5_delta_95ci": interval,
        "paired_recall1_delta_95ci": paired_interval(br, sr, present, k=1),
        "paired_sofa_recall5_delta_95ci": paired_interval(br, sr, sofa),
        "ranking_seconds": {"baseline": baseline_seconds, "selected": selected_seconds},
        "selected_configuration": chosen,
    }
    final["adopt_in_app"] = bool(
        interval[0] > 0
        and final["selected_sofa"]["recall_at_5"] >= final["baseline_sofa"]["recall_at_5"]
        and final["selected"]["recall_at_1"] >= final["baseline"]["recall_at_1"]
    )
    final["adoption_reason"] = (
        "Passed predeclared held-out paired gate"
        if final["adopt_in_app"]
        else "Insufficient independently held-out evidence under predeclared adoption gate; keep production Phase6D aggregation"
    )
    rows = []
    category_rows = []
    development = read_json(root / "development-results.json")["individual_and_combination"]
    memory = read_json(root / "embeddings/report.json")["peak_process_rss_bytes"]
    for result in development:
        mode = result["preprocessing"]
        aggregation = result["aggregation"]
        a = np.load(root / "embeddings" / f"{mode}.npy")
        scores = score_products(
            a[offset:][dev],
            a[:offset],
            members,
            aggregation,
            int(result["name"].split("-")[1]) if result["name"].startswith("pool-") else None,
        )
        ranks = query_ranks(scores, pids, [queries[i] for i in dev])
        b = [br[i] for i in dev]
        paired = paired_interval(
            b, ranks, [j for j, i in enumerate(dev) if queries[i]["true_product_in_catalog"]]
        )
        rows.append(
            {
                "split": "development",
                "configuration": result["name"],
                "query_count": result["metrics"]["queries"],
                "catalog_products": len(products),
                "baseline_recall1": summarize(br, queries, dev)["recall_at_1"],
                "recall1": result["metrics"]["recall_at_1"],
                "baseline_recall5": summarize(br, queries, dev)["recall_at_5"],
                "recall5": result["metrics"]["recall_at_5"],
                "sofa_recall5": result["sofa"]["recall_at_5"],
                "retrieval_seconds": result["retrieval_seconds"],
                "peak_embedding_process_rss_bytes": memory,
                "paired_recall5_delta_ci_low": paired[0],
                "paired_recall5_delta_ci_high": paired[1],
            }
        )
        for category in sorted(set(categories)):
            m = summarize(
                ranks,
                [queries[i] for i in dev],
                [j for j, i in enumerate(dev) if queries[i]["label"] == category],
            )
            category_rows.append(
                {"split": "development", "configuration": result["name"], "category": category, **m}
            )
    for name, ranks, seconds in [
        ("baseline", br, baseline_seconds),
        ("selected", sr, selected_seconds),
    ]:
        m = summarize(ranks, queries, held)
        rows.append(
            {
                "split": "heldout",
                "configuration": name,
                "query_count": m["queries"],
                "catalog_products": len(products),
                "baseline_recall1": final["baseline"]["recall_at_1"],
                "recall1": m["recall_at_1"],
                "baseline_recall5": final["baseline"]["recall_at_5"],
                "recall5": m["recall_at_5"],
                "sofa_recall5": summarize(ranks, queries, sofa)["recall_at_5"],
                "retrieval_seconds": seconds,
                "peak_embedding_process_rss_bytes": memory,
                "paired_recall5_delta_ci_low": interval[0] if name == "selected" else 0,
                "paired_recall5_delta_ci_high": interval[1] if name == "selected" else 0,
            }
        )
        for category in sorted(set(categories)):
            category_rows.append(
                {
                    "split": "heldout",
                    "configuration": name,
                    "category": category,
                    **summarize(
                        ranks, queries, [i for i in held if queries[i]["label"] == category]
                    ),
                }
            )
    for file, records in [
        ("baseline-comparison.csv", rows),
        ("category-analysis.csv", category_rows),
    ]:
        with (root / file).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    # Verify exact nearest-neighbour machinery independently from aggregation.
    image_rows = [{"label": p["label"], "file": im["file"]} for p in products for im in p["images"]]
    for i, row in enumerate(image_rows):
        row["product_id"] = f"gallery-image-{i}"
    index = CatalogIndex.build(
        original[:offset], image_rows, read_json(root / "embeddings/report.json")["model"]
    )
    try:
        scores, indices = index.index.search(original[offset:], offset)
    finally:
        index.close()
    dot = np.einsum("qd,gd->qg", original[offset:], original[:offset], optimize=False)
    final["faiss_max_cosine_error"] = float(
        abs(scores - np.take_along_axis(dot, indices, axis=1)).max()
    )
    detector = {
        r["image_id"]: r
        for r in read_json(root / "detector-predictions/predictions.json")["images"]
    }
    subindices = [
        i
        for i, q in enumerate(queries)
        if q["image_id"] in protocol["predicted_category_query_image_ids"]
    ]
    subqueries = [queries[i] for i in subindices]
    pred = []
    for q in subqueries:
        rows = detector[q["image_id"]]["detections"]
        best = max(rows, key=lambda d: d["score"]) if rows else {}
        pred.append({"label": best.get("label"), "score": best.get("score", 0)})
    conditions = []
    chosen_category = category_choice["selected"]
    for split in ["development", "heldout"]:
        subset = [j for j, q in enumerate(subqueries) if q["split"] == split]
        s = selected[subindices]
        truth = [{"label": q["label"], "score": 1} for q in subqueries]
        diagnostic = [("global", s, 0)]
        for name, mode, prs, weight in [
            ("verified-category-oracle", "oracle", truth, 0),
            ("actual-predicted-strict", "strict", pred, 0),
            (
                "development-selected-category",
                chosen_category["mode"],
                pred,
                chosen_category["weight"],
            ),
        ]:
            cs, fb = (s, 0) if mode == "none" else category_scores(s, categories, prs, mode, weight)
            diagnostic.append((name, cs, fb))
        wrong = [
            {
                "label": sorted(set(categories))[
                    (sorted(set(categories)).index(q["label"]) + 1) % len(set(categories))
                ],
                "score": 0.9,
            }
            for q in subqueries
        ]
        for mode in ["strict", "soft"]:
            cs, fb = category_scores(s, categories, wrong, mode)
            diagnostic.append(("synthetic-wrong-category-" + mode, cs, fb))
        for name, sc, fb in diagnostic:
            conditions.append(
                {
                    "split": split,
                    "condition": name,
                    "metrics": summarize(query_ranks(sc, pids, subqueries), subqueries, subset),
                    "fallbacks_all_subset": fb,
                    "actual_detector_category_correct": sum(
                        pred[j]["label"] == subqueries[j]["label"] for j in subset
                    ),
                    "actual_detector_views": len(subset),
                }
            )
    condition_report = read_json(root / "conditions/report.json")
    crop_results = []
    robustness = []
    for mode in ["detector_bbox", "detector_margin10"]:
        v = np.load(root / "conditions" / f"{mode}.npy")
        v = align_image_rows(
            v, condition_report["query_image_ids"], [q["image_id"] for q in subqueries]
        )
        sc = score_products(v, variant[:offset], members, chosen["aggregation"])
        ranks = query_ranks(sc, pids, subqueries)
        for split in ["development", "heldout"]:
            subset = [i for i, q in enumerate(subqueries) if q["split"] == split]
            crop_results.append(
                {
                    "split": split,
                    "condition": mode,
                    "baseline": summarize([sr[i] for i in subindices], subqueries, subset),
                    "metrics": summarize(ranks, subqueries, subset),
                }
            )
    for kind in [
        "lighting",
        "shadow",
        "occlusion",
        "color",
        "synthetic_stain",
        "synthetic_wear",
        "background_clutter",
    ]:
        v = np.load(root / "conditions" / ("synthetic_" + kind + ".npy"))
        v = align_image_rows(
            v, condition_report["query_image_ids"], [q["image_id"] for q in subqueries]
        )
        sc = score_products(v, variant[:offset], members, chosen["aggregation"])
        ranks = query_ranks(sc, pids, subqueries)
        for split in ["development", "heldout"]:
            subset = [i for i, q in enumerate(subqueries) if q["split"] == split]
            base = [sr[i] for i in subindices]
            robustness.append(
                {
                    "split": split,
                    "synthetic_condition": kind,
                    "baseline": summarize(base, subqueries, subset),
                    "metrics": summarize(ranks, subqueries, subset),
                    "top1_changed_fraction": float(
                        np.mean(
                            np.argmax(sc, axis=1)[subset]
                            != np.argmax(selected[subindices], axis=1)[subset]
                        )
                    ),
                }
            )
    threshold = chosen["unknown_threshold_experimental"]
    unknown = []
    for split, inds in [("development", dev), ("heldout", held)]:
        for name, sc in [("baseline", baseline), ("selected", selected)]:
            present_ids = [i for i in inds if queries[i]["true_product_in_catalog"]]
            absent_ids = [i for i in inds if not queries[i]["true_product_in_catalog"]]
            top = sc.max(axis=1)
            unknown.append(
                {
                    "split": split,
                    "configuration": name,
                    "threshold": threshold,
                    "known_queries": len(present_ids),
                    "absent_queries": len(absent_ids),
                    "known_accept_rate": float(np.mean(top[present_ids] >= threshold)),
                    "unknown_reject_rate": float(np.mean(top[absent_ids] < threshold)),
                    "exact_identity_verified_claims": 0,
                    "calibrated_identity_probability": False,
                }
            )
    write_json(
        root / args.output,
        {
            **final,
            "category_experiments": conditions,
            "actual_detector_crop_experiments": crop_results,
            "synthetic_robustness": robustness,
            "unknown_product_tradeoff": unknown,
            "queries": queries,
            "product_ids": pids,
            "baseline_ranks": br,
            "selected_ranks": sr,
            "baseline_top5": [[pids[j] for j in np.argsort(-row)[:5]] for row in baseline],
            "selected_top5": [[pids[j] for j in np.argsort(-row)[:5]] for row in selected],
            "protocol_sha256": digest(root / "protocol.json"),
            "selection_lock_sha256": digest(root / "selection-lock.json"),
            "category_selection_lock_sha256": digest(root / "category-selection-lock.json"),
            "heldout_selection_feedback_used": False,
            "condition_inference": condition_report,
        },
    )
    print(
        {
            k: final[k]
            for k in [
                "baseline",
                "selected",
                "baseline_sofa",
                "selected_sofa",
                "paired_recall5_delta_95ci",
                "adopt_in_app",
                "adoption_reason",
            ]
        }
    )


if __name__ == "__main__":
    main()
