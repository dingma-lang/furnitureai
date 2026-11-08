"""Evidence-only historical sofa error sheets and numeric indexing audit."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from furniture_bench.data import read_json, write_json
from furniture_bench.retrieval import CatalogIndex


def main():
    old = Path("reports/phase6d-product-identification")
    data = Path("data/catalogs/abo-phase6d")
    out = Path("reports/phase6d1-retrieval-optimization/failure-analysis")
    out.mkdir(parents=True, exist_ok=True)
    review_path = out / "visual-review.json"
    visual_review = read_json(review_path).get("query_types", {}) if review_path.exists() else {}
    products = {p["product_id"]: p for p in read_json(data / "catalog.json")["products"]}
    rows = [
        q
        for q in read_json(old / "benchmark-validated/rankings.json")["queries"]
        if q["label"] == "sofa" and q["true_product_in_catalog"]
    ]
    findings = []
    for page, start in enumerate(range(0, len(rows), 5)):
        canvas = Image.new("RGB", (1200, 5 * 240), "#f5f4ed")
        draw = ImageDraw.Draw(canvas)
        for n, q in enumerate(rows[start : start + 5]):
            pid = q["product_id"]
            truth = products[pid]
            top = products[q["candidates"][0]["product_id"]]
            y = n * 240
            title = f"{pid} true rank {q['correct_rank']} | top: {top['product_id']} ({top['label']}) | query / true gallery 1 / true gallery 2 / wrong-or-correct top"
            draw.text((10, y + 5), title, fill="black", font=ImageFont.load_default(size=13))
            paths = (
                [data / q["file"]]
                + [data / i["file"] for i in truth["images"]]
                + [
                    data
                    / next(
                        i["file"]
                        for i in top["images"]
                        if i["image_id"] == q["candidates"][0]["best_image_id"]
                    )
                ]
            )
            for j, path in enumerate(paths):
                with Image.open(path) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (280, 190))
                canvas.paste(
                    thumb,
                    (10 + j * 295 + (280 - thumb.width) // 2, y + 30 + (190 - thumb.height) // 2),
                )
            findings.append(
                {
                    "product_id": pid,
                    "rank": q["correct_rank"],
                    "query_image_id": q["image_id"],
                    "true_gallery_image_ids": [i["image_id"] for i in truth["images"]],
                    "top_product_id": top["product_id"],
                    "top_category": top["label"],
                    "true_gallery_available": True,
                    "same_brand_as_wrong_top": truth.get("brand") == top.get("brand"),
                    "metadata_identity_error_verified": False,
                    "visual_cause": "requires review of saved image sheet; numeric category confusion known",
                    "visually_reviewed_query_type": visual_review.get(pid, "not_reviewed"),
                }
            )
        canvas.save(out / f"sofa-errors-{page + 1}.png")
    g = np.load(old / "benchmark-validated/gallery-embeddings.npy")
    q = np.load(old / "benchmark-validated/query-embeddings.npy")
    index = CatalogIndex.load(old / "benchmark/index")
    try:
        scores, indices = index.index.search(q, len(g))
    finally:
        index.close()
    dot = np.einsum("qd,gd->qg", q, g, optimize=False)
    expected = np.take_along_axis(dot, indices, axis=1)
    write_json(
        out / "audit.json",
        {
            "sofa_queries": len(rows),
            "sofa_top1_correct": sum(r["correct_rank"] == 1 for r in rows),
            "sofa_top5_correct": sum(r["correct_rank"] <= 5 for r in rows),
            "top1_category_confusions": sum(r["candidates"][0]["label"] != "sofa" for r in rows),
            "all_true_identities_present": True,
            "gallery_images_per_identity": sorted({len(p["images"]) for p in products.values()}),
            "embedding_norm_range": [
                float(min(np.linalg.norm(g, axis=1).min(), np.linalg.norm(q, axis=1).min())),
                float(max(np.linalg.norm(g, axis=1).max(), np.linalg.norm(q, axis=1).max())),
            ],
            "faiss_vs_direct_cosine_max_error": float(abs(scores - expected).max()),
            "preprocessing": "Official bilinear resize to224x224 (aspect-ratio warp), RGB normalization mean/std0.5; no object crop or instance mask",
            "findings": findings,
            "scope": "Examined historical set; diagnostic only, not held-out optimization evidence",
        },
    )


if __name__ == "__main__":
    main()
