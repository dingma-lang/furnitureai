"""Generate review figures and compact summaries from locked evaluation artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from furniture_bench.data import read_json, write_json
from furniture_bench.retrieval_experiments import score_products


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    result = read_json(root / "final-evaluation-aligned.json")
    protocol = read_json(root / "protocol.json")
    development = read_json(root / "development-results.json")
    gallery = read_json(root / "gallery.json")
    products = {p["product_id"]: p for p in gallery["products"]}
    data = Path(gallery["data_root"])
    queries = result["queries"]
    before = result["baseline_ranks"]
    after = result["selected_ranks"]
    held = [
        i for i, q in enumerate(queries) if q["split"] == "heldout" and q["true_product_in_catalog"]
    ]
    gains = [i for i in held if before[i] > 5 and after[i] <= 5]
    losses = [i for i in held if before[i] <= 5 and after[i] > 5]
    same = [i for i in held if before[i] == after[i]]
    chosen = [
        gains[0] if gains else held[0],
        next(
            (i for i in losses if queries[i]["label"] == "sofa"), losses[0] if losses else held[1]
        ),
        same[0] if same else held[2],
    ]
    canvas = Image.new("RGB", (1320, 930), "#f5f4ed")
    draw = ImageDraw.Draw(canvas)
    for row, i in enumerate(chosen):
        q = queries[i]
        baseid = result["baseline_top5"][i][0]
        newid = result["selected_top5"][i][0]
        true = products[q["product_id"]]
        caption = f"Held-out {q['label']} | true ID {q['product_id']} | baseline rank {before[i]}, frozen proposal rank {after[i]} (NOT DEPLOYED)"
        y = row * 310
        draw.text((10, y + 8), caption, fill="black", font=ImageFont.load_default(size=15))
        images = [
            Path(q["file"]),
            data / true["images"][0]["file"],
            data / products[baseid]["images"][0]["file"],
            data / products[newid]["images"][0]["file"],
        ]
        labels = [
            "Query (separate photo)",
            "Correct product reference",
            "Baseline top1 " + baseid,
            "Frozen proposal top1 " + newid,
        ]
        for col, (path, label) in enumerate(zip(images, labels)):
            with Image.open(path) as source:
                im = ImageOps.contain(source.convert("RGB"), (310, 220))
            canvas.paste(
                im, (10 + col * 328 + (310 - im.width) // 2, y + 38 + (220 - im.height) // 2)
            )
            draw.text(
                (10 + col * 328, y + 265), label, fill="black", font=ImageFont.load_default(size=12)
            )
    canvas.save(root / "retrieval-comparison.png")
    individual = development["individual_and_combination"][:7]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(individual))
    axes[0].bar(
        x - 0.17, [r["metrics"]["recall_at_5"] for r in individual], 0.34, label="Overall R@5"
    )
    axes[0].bar(x + 0.17, [r["sofa"]["recall_at_5"] for r in individual], 0.34, label="Sofa R@5")
    axes[0].set_xticks(
        x, [r["name"].replace("original+", "") for r in individual], rotation=35, ha="right"
    )
    axes[0].set_title("Development only: 61 present queries / 11 sofas")
    axes[0].legend()
    axes[1].bar(
        np.arange(2) - 0.17,
        [result["baseline"]["recall_at_5"], result["selected"]["recall_at_5"]],
        0.34,
        label="Overall R@5",
    )
    axes[1].bar(
        np.arange(2) + 0.17,
        [result["baseline_sofa"]["recall_at_5"], result["selected_sofa"]["recall_at_5"]],
        0.34,
        label="Sofa R@5",
    )
    axes[1].set_xticks([0, 1], ["Baseline", "Frozen proposal"])
    axes[1].set_title("Held out: 56 present queries / 10 sofas")
    axes[1].legend()
    for ax in axes:
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(root / "ablation-overview.png", dpi=160)
    plt.close(fig)
    # Real-room rank stability is separate; no known true product identity here.
    oldcatalog = read_json("data/catalogs/abo-phase6d/catalog.json")["products"]
    members = []
    n = 0
    for p in oldcatalog:
        members.append(list(range(n, n + len(p["images"]))))
        n += len(p["images"])
    oldg = np.load(
        "reports/phase6d-product-identification/benchmark-validated/gallery-embeddings.npy"
    )
    room = np.load(root / "conditions/real_room_crops.npy")
    roomranks = score_products(room, oldg, members, "max")
    ids = [p["product_id"] for p in oldcatalog]
    write_json(
        root / "room-to-catalog-diagnostic.json",
        {
            "scope": "Actual licensed household photo with saved OWLv2 crop; no verified retail identity so NO Recall/accuracy",
            "crop_variants": ["actual_confirmed_box", "shrink5percent", "expand10percent"],
            "top5": [
                [
                    {"product_id": ids[j], "similarity": float(scores[j])}
                    for j in np.argsort(-scores)[:5]
                ]
                for scores in roomranks
            ],
            "rankings_changed": bool(
                not np.array_equal(np.argsort(-roomranks[0])[:5], np.argsort(-roomranks[1])[:5])
                or not np.array_equal(np.argsort(-roomranks[0])[:5], np.argsort(-roomranks[2])[:5])
            ),
            "identity_verified": False,
        },
    )
    detection = read_json(root / "conditions/report.json")
    predictions = {r["image_id"]: r for r in detection["detector_crops"]}
    pred_counts = {}
    for split in ["development", "heldout"]:
        rows = [q for q in queries if q["split"] == split and q["image_id"] in predictions]
        pred_counts[split] = {
            "queries": len(rows),
            "no_detection": sum(predictions[q["image_id"]]["fallback"] for q in rows),
            "below_category_confidence_threshold_or_missing": sum(
                predictions[q["image_id"]]["score"] < 0.35 for q in rows
            ),
            "correct_top_detection_category": sum(
                predictions[q["image_id"]]["predicted_label"] == q["label"] for q in rows
            ),
        }
    summary = {
        "historical_sofa_diagnostics": read_json(root / "failure-analysis/audit.json"),
        "fresh_protocol_counts": protocol["counts"],
        "development_choice": read_json(root / "selection-lock.json"),
        "heldout_baseline": result["baseline"],
        "heldout_frozen_proposal": result["selected"],
        "heldout_baseline_sofa": result["baseline_sofa"],
        "heldout_proposal_sofa": result["selected_sofa"],
        "heldout_R5_gains": len(gains),
        "heldout_R5_losses": len(losses),
        "paired_R5_delta_95ci": result["paired_recall5_delta_95ci"],
        "production_retrieval_changed": result["adopt_in_app"],
        "reason": result["adoption_reason"],
        "predicted_category_subset_counts": pred_counts,
        "unknown_tradeoffs": result["unknown_product_tradeoff"],
        "new_inference": {
            "variants": read_json(root / "embeddings/report.json"),
            "conditions": detection,
        },
        "multiple_comparison_scope": "Ablation selection development only; heldout proposal comparison fixed; secondary diagnostics cannot reselect model",
        "physical_accuracy_claimed": False,
    }
    write_json(root / "summary.json", summary)
    (root / "ablation-study.md").write_text(
        "# Controlled ablations\n\nSee baseline-comparison.csv and category-analysis.csv for numerical rows. Development and held-out rows must not be combined for selection.\n\n"
        + "\n".join(
            f"- {r['name']}: development R@1 {r['metrics']['recall_at_1']:.3f}, R@5 {r['metrics']['recall_at_5']:.3f}, sofa R@5 {r['sofa']['recall_at_5']:.3f}."
            for r in development["individual_and_combination"]
        )
        + "\n\nThe one allowed combination duplicates original + first retained view because original preprocessing won independently. Combining transformations was not assumed to improve results.\n\nThe frozen proposal did not improve held-out R@5 and regressed on sofa R@5. Its 95% paired interval spans zero, so production aggregation remains unchanged. Verified-category filtering is an oracle diagnostic. Actual detector filtering uses the predeclared 27-photo subset and confidence fallback; those small-subset numbers are not full-set accuracy.\n\nTight/10%-margin catalog crops are white-border heuristics, not verified furniture masks. Actual detector crops are measured separately. Background reduction with reliable instance masks was unavailable and was not faked using room-surface segmentation. Gallery image counts are exactly two for every identity, preventing unequal view-count advantages in this experiment. Main-only means the first retained view: 106/117 are official main_image_id photographs; 11 are the next eligible available photo.\n\nPool10/40 restrict candidate identity coverage but still use an exact small-gallery score matrix; no approximate-nearest-neighbour speedup is claimed. FlatIP matches direct cosine to the error recorded in final-evaluation-aligned.json. Metadata-based brand reranking was not used because query brands/specifications are unavailable without identity leakage.\n\nThe synthetic analyses are image-coordinate perturbations, not independently validated real wear, damage or camera-angle tests. Supplemental condition vectors were reconciled by explicit image ID after an ordering bug; main held-out scores and selection locks were unchanged.\n"
    )


if __name__ == "__main__":
    main()
