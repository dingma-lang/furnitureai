"""Multi-image product aggregation around the existing isolated-FAISS catalog API."""
import copy
from pathlib import Path

from .data import digest, read_json
from .retrieval import CatalogIndex


def aggregate_products(image_results, products, preferences=(), retailer_only=False, category=None, k=5):
    """Max cosine per listing identity; preference bonus is not identity confidence."""
    by_id = {p["product_id"]: p for p in products}
    allowed = {str(p).casefold() for p in preferences}
    best = {}
    for row in image_results:
        key = row["dataset_product_id"]
        product = by_id[key]
        if category is not None and product["label"] != category:
            continue
        preferred = product.get("retailer", "").casefold() in allowed
        if retailer_only and allowed and not preferred:
            continue
        if key not in best or row["similarity"] > best[key]["similarity"]:
            best[key] = {"product_id": key, "similarity": float(row["similarity"]), "best_image_id": row["image_id"], "preferred_retailer": preferred}
    results = []
    for key, row in best.items():
        product = by_id[key]
        results.append({**row, "ranking_score": row["similarity"] + (.02 if row["preferred_retailer"] else 0), "identity_verified": False, "classification": "visually_similar_alternative", "exact_identification": "No verified exact match.", "image_ids": [i["image_id"] for i in product.get("images", [])], "name": product.get("name"), "brand": product.get("brand"), "manufacturer": product.get("manufacturer"), "model_number": product.get("model_number"), "label": product["label"], "retailer": product.get("retailer"), "specifications": product.get("specifications", {}), "price": None, "current_availability": None, "source": product.get("rights", {})})
    return sorted(results, key=lambda r: (-r["ranking_score"], r["product_id"]))[:k]


class MultiImageCatalog:
    def __init__(self, index, catalog):
        self.index, self.catalog = index, catalog

    @classmethod
    def build(cls, vectors, catalog, embedding, root):
        rows = []
        for product in catalog["products"]:
            for image in product["images"]:
                rows.append({"product_id": product["product_id"] + "::" + image["image_id"], "dataset_product_id": product["product_id"], "image_id": image["image_id"], "label": product["label"], "file": str((Path(root) / image["file"]).resolve()), "image_sha256": image["sha256"]})
        if len({p["product_id"] for p in catalog["products"]}) != len(catalog["products"]):
            raise ValueError("Dataset product IDs must be unique")
        index = CatalogIndex.build(vectors, rows, embedding)
        index.metadata["multiview_catalog"] = copy.deepcopy(catalog)
        return cls(index, catalog)

    @classmethod
    def load(cls, folder):
        index = CatalogIndex.load(folder)
        if "multiview_catalog" not in index.metadata:
            index.close()
            raise ValueError("Not a multi-image product index")
        return cls(index, index.metadata["multiview_catalog"])

    def save(self, folder):
        self.index.save(folder)

    def close(self):
        self.index.close()

    def search(self, vectors, preferences=(), retailer_only=False, category=None, k=5):
        # Search all image rows for exact identity aggregation on this bounded catalog.
        results, seconds = self.index.search(vectors, k=self.index.index.ntotal)
        image_rows = {r["product_id"]: r for r in self.index.metadata["products"]}
        grouped = []
        for matches in results:
            rows = [{**m, "dataset_product_id": image_rows[m["product_id"]]["dataset_product_id"], "image_id": image_rows[m["product_id"]]["image_id"]} for m in matches]
            grouped.append(aggregate_products(rows, self.catalog["products"], preferences, retailer_only, category, k))
        return grouped, seconds


def load_catalog_images(manifest):
    catalog = read_json(manifest)
    images = []
    for product in catalog["products"]:
        if product.get("rights", {}).get("allowed_for_benchmark") is not True:
            raise ValueError("Catalog lacks licensed-data provenance")
        for image in product["images"]:
            path = Path(manifest).parent / image["file"]
            if digest(path) != image["sha256"]:
                raise ValueError("Catalog image bytes changed")
            images.append(path)
    return catalog, images
