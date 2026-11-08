"""Saved-query visual comparisons and synthetic robustness summary, no inference."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from furniture_bench.data import read_json, write_json

ROOT = Path("reports/phase6d-product-identification")
DATA = Path("data/catalogs/abo-phase6d")


def panel(image, size=(230, 190)):
    frame = Image.new("RGB", size, "white")
    result = ImageOps.contain(image.convert("RGB"), size)
    frame.paste(result, ((size[0]-result.width)//2, (size[1]-result.height)//2))
    return frame


def draw_row(canvas, offset, query, candidates, products, caption):
    draw = ImageDraw.Draw(canvas)
    draw.text((15, offset), caption, fill="black", font=ImageFont.load_default(size=17))
    canvas.paste(panel(query), (15, offset+35))
    draw.text((15, offset+230), "Query (different source image)", fill="black")
    for i, result in enumerate(candidates[:5]):
        product = products[result["product_id"]]
        image = next(im for im in product["images"] if im["image_id"] == result["best_image_id"])
        with Image.open(DATA / image["file"]) as source:
            canvas.paste(panel(source), (260+i*245, offset+35))
        text = f"#{i+1} {result['product_id']}\n{result.get('brand') or 'Brand unavailable'}\ncosine {result['similarity']:.3f}; not identity probability"
        draw.multiline_text((260+i*245, offset+230), text, fill="black", spacing=3)


def main():
    evaluation = read_json(ROOT / "evaluation-audited.json")
    excluded = set(evaluation["excluded_queries_after_crop_audit"])
    rows = [r for r in read_json(ROOT / "benchmark-validated/rankings.json")["queries"] if r["image_id"] not in excluded]
    products = {p["product_id"]: p for p in read_json(DATA / "catalog.json")["products"]}
    choices = [next(r for r in rows if r["true_product_in_catalog"] and r["correct_rank"] == 1), next(r for r in rows if r["true_product_in_catalog"] and r["correct_rank"] != 1), next(r for r in rows if not r["true_product_in_catalog"])]
    canvas = Image.new("RGB", (1500, 985), "#f6f5ef")
    for offset, row, title in zip((10, 335, 660), choices, ("Catalog query: true identity ranked first", "Catalog query: incorrect top identity", "Absent-product query: visually similar alternatives; NO VERIFIED EXACT MATCH")):
        with Image.open(DATA / row["file"]) as query:
            draw_row(canvas, offset, query, row["candidates"], products, title + " · true ID " + row["product_id"])
    canvas.save(ROOT / "retrieval-examples.png")
    source = read_json(ROOT / "actual-room-query.json")
    cache = Path(".local-projects/product-cache") / source["cache_key"]
    with Image.open(cache / "crop.png") as crop:
        room = Image.new("RGB", (1500, 330), "#f6f5ef")
        draw_row(room, 10, crop, source["candidates"], products, "Actual OWLv2 room-photo crop to five ABO alternatives; product identity unknown")
        room.save(ROOT / "real-room-product-comparison.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = evaluation["robustness"][0]["baseline_same_subset"]
    labels = ["Original views"] + [r["transformation"].replace("synthetic_", "") for r in evaluation["robustness"]]
    at1 = [base["recall_at_1"]] + [r["metrics"]["recall_at_1"] for r in evaluation["robustness"]]
    at5 = [base["recall_at_5"]] + [r["metrics"]["recall_at_5"] for r in evaluation["robustness"]]
    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x-.17, at1, .34, label="Recall@1")
    ax.bar(x+.17, at5, .34, label="Recall@5")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Same 24 catalog queries: synthetic perturbations, not real wear evidence")
    fig.tight_layout()
    fig.savefig(ROOT / "robustness.png", dpi=160)
    plt.close(fig)
    write_json(ROOT / "visualization-sources.json", {"catalog_examples": choices, "rights": "Images/metadata Amazon.com / ABO, CC BY 4.0; cropped/resized for figures. Room photo Phillip Goldsberry / Unsplash, existing licensed benchmark photograph.", "no_model_inference": True})


if __name__ == "__main__":
    main()
