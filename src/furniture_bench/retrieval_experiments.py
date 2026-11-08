"""Small inference-only preprocessing/ranking functions; no identity features."""

import numpy as np
from PIL import Image, ImageOps

from .retrieval import unit_vectors


def prepare_image(image, mode):
    image = image.convert("RGB")
    if mode == "original":
        return image
    if mode == "letterbox":
        resized = ImageOps.contain(image, (224, 224), Image.Resampling.BILINEAR)
        out = Image.new("RGB", (224, 224), (127, 127, 127))
        out.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
        return out
    if mode not in ("white_margin_trim", "margin10"):
        raise ValueError("Unknown preprocessing")
    a = np.asarray(image)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.mean(np.all(border >= 240, axis=1)) < 0.90:
        return image
    foreground = np.any(a < 240, axis=2)
    y, x = np.nonzero(foreground)
    if not len(x) or np.mean(foreground) < 0.03:
        return image
    left, top, right, bottom = int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)
    if mode == "margin10":
        dx, dy = (right - left) * 0.10, (bottom - top) * 0.10
        left, top, right, bottom = (
            max(0, int(left - dx)),
            max(0, int(top - dy)),
            min(image.width, int(right + dx)),
            min(image.height, int(bottom + dy)),
        )
    return image.crop((left, top, right, bottom))


def score_products(queries, gallery, member_indices, method="max", pool=None):
    q, g = unit_vectors(queries), unit_vectors(gallery)
    similarities = np.einsum("qd,gd->qg", q, g, optimize=False)
    if method == "normalized_centroid":
        centers = unit_vectors(np.stack([g[ids].mean(axis=0) for ids in member_indices]))
        scores = np.einsum("qd,pd->qp", q, centers, optimize=False)
    elif method in ("max", "top2_mean", "main_only"):
        values = []
        for ids in member_indices:
            if not len(ids):
                raise ValueError("Product has no gallery images")
            block = similarities[:, ids]
            values.append(
                block.max(axis=1)
                if method == "max"
                else np.sort(block, axis=1)[:, -min(2, len(ids)) :].mean(axis=1)
                if method == "top2_mean"
                else block[:, 0]
            )
        scores = np.stack(values, axis=1)
    else:
        raise ValueError("Unknown product aggregation")
    if pool is not None:
        if pool < 1:
            raise ValueError("Pool size must be positive")
        nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :pool]
        for p, ids in enumerate(member_indices):
            included = np.any(np.isin(nearest, ids), axis=1)
            scores[~included, p] = -np.inf
    return scores


def category_scores(scores, categories, predictions, mode, weight=0.08, minimum_confidence=0.35):
    result = np.array(scores, float, copy=True)
    fallbacks = 0
    for i, pred in enumerate(predictions):
        label, confidence = pred.get("label"), pred.get("score", 0)
        eligible = np.array([c == label for c in categories])
        if (
            label is None
            or not eligible.any()
            or (mode != "oracle" and confidence < minimum_confidence)
        ):
            fallbacks += 1
            continue
        if mode in ("strict", "oracle"):
            result[i, ~eligible] = -np.inf
        elif mode == "soft":
            result[i, eligible] += weight * float(confidence)
        else:
            raise ValueError("Unknown category rule")
    return result, fallbacks


def query_ranks(scores, product_ids, queries):
    if len(set(product_ids)) != len(product_ids) or scores.shape != (
        len(queries),
        len(product_ids),
    ):
        raise ValueError("Query/product identity map is not unique or shape-aligned")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = []
    for i, q in enumerate(queries):
        candidates = [product_ids[j] for j in order[i] if np.isfinite(scores[i, j])]
        ranks.append(
            candidates.index(q["product_id"]) + 1 if q["product_id"] in candidates else None
        )
    return ranks


def align_image_rows(vectors, source_ids, target_ids):
    """Condition vectors must follow explicit image identities, never incidental record order."""
    if len(source_ids) != len(vectors) or len(set(source_ids)) != len(source_ids):
        raise ValueError("Condition source image IDs are duplicate or not aligned")
    if len(set(target_ids)) != len(target_ids) or set(target_ids) != set(source_ids):
        raise ValueError("Condition target identities differ from source identities")
    rows = {key: i for i, key in enumerate(source_ids)}
    return np.asarray(vectors)[[rows[key] for key in target_ids]]


def summarize(ranks, queries, indices):
    ids = [i for i in indices if queries[i]["true_product_in_catalog"]]
    r = [ranks[i] for i in ids]
    return {
        "queries": len(ids),
        "recall_at_1": float(np.mean([v == 1 for v in r])) if r else None,
        "recall_at_5": float(np.mean([v is not None and v <= 5 for v in r])) if r else None,
        "mrr": float(np.mean([1 / v if v else 0 for v in r])) if r else None,
    }


def paired_interval(before, after, indices, samples=5000, seed=6118, k=5):
    ids = np.asarray(indices, int)
    if not len(ids):
        return [None, None]
    delta = np.array(
        [
            float(after[i] is not None and after[i] <= k)
            - float(before[i] is not None and before[i] <= k)
            for i in ids
        ]
    )
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(0, len(ids), size=(samples, len(ids)))].mean(axis=1)
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()
