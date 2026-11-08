"""Freeze nested galleries, incrementally embed, and evaluate an experimental catalog."""

import argparse
import copy
import csv
import hashlib
import json
import os
import resource
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.catalog_expansion import (
    atomic_json,
    incremental_embeddings,
    validate_catalog,
    validate_nested_galleries,
)
from furniture_bench.data import digest, load_image, read_json
from furniture_bench.dataset_quality import fingerprint
from furniture_bench.embedding.siglip2 import Siglip2Embedder
from furniture_bench.product_catalog import MultiImageCatalog
from furniture_bench.retrieval_experiments import paired_interval, summarize

ROOT = Path("reports/phase6d3-dataset-expansion")
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def all_products():
    return (
        read_json(ROOT / "existing-products.json")["products"]
        + read_json(ROOT / "new-products.json")["products"]
    )


def prepare():
    if (ROOT / "scaling-protocol-lock.json").exists():
        raise ValueError("Protocol already frozen; reuse it")
    products = all_products()
    exported = copy.deepcopy(products)
    for p in exported:
        for image in p["images"]:
            image["file"] = os.path.relpath(image["file"], ROOT)
            image["image_role"] = (
                "official_main_whole_object_candidate"
                if image["image_id"] == p["source_record"].get("main_image_id")
                else "alternate_or_detail_unresolved"
            )
            image["role_provenance"] = (
                "Official source ordering only; provisional, not human-verified whole-object label"
            )
    atomic_json(
        ROOT / "catalog-products.json",
        {
            "schema_version": 2,
            "products": exported,
            "identity_scope": "ABO_listing; manufacturer/model unverified",
        },
    )
    catalog = {
        "schema_version": 2,
        "products": [{**p, "images": p["images"][:2]} for p in exported],
        "experimental": True,
    }
    validate_catalog(catalog, ROOT)
    atomic_json(ROOT / "expanded-catalog.json", catalog)
    old_ids = {
        i["image_id"] for p in products if p.get("source_cohort") != "phase6d3" for i in p["images"]
    }
    refs = [i for p in products for i in p["images"][:2]]
    previous = [i for p in products if p.get("source_cohort") != "phase6d3" for i in p["images"]]
    screening = {i["image_id"]: i for i in refs + previous}
    fps = [fingerprint(i["file"]) for i in screening.values()]
    bits = np.stack([f["bits"] for f in fps])
    hashes = {f["sha256"] for f in fps}
    groups, exclusions = defaultdict(list), []
    query_ids, query_hashes, query_fps = set(), set(), []
    for p in products:
        if p.get("source_cohort") != "phase6d3":
            continue
        q = p["images"][2]
        fp = fingerprint(q["file"])
        close = np.flatnonzero(np.count_nonzero(bits != fp["bits"], axis=(1, 2)) <= 6)
        near = fp["sha256"] in hashes or any(
            float(np.mean(abs(fps[j]["rgb"] - fp["rgb"]))) < 14 for j in close
        )
        duplicate_query = (
            q["image_id"] in query_ids
            or q["sha256"] in query_hashes
            or any(
                np.count_nonzero(other["bits"] != fp["bits"]) <= 6
                and float(np.mean(abs(other["rgb"] - fp["rgb"]))) < 14
                for other in query_fps
            )
        )
        if q["image_id"] in old_ids or near or duplicate_query:
            exclusions.append(
                {
                    "product_id": p["product_id"],
                    "image_id": q["image_id"],
                    "reason": "query_to_query_duplicate"
                    if duplicate_query
                    else "prior_query_or_reference_id_hash_or_crop_near_duplicate",
                }
            )
        else:
            groups[p["selection_category"]].append(p)
            query_ids.add(q["image_id"])
            query_hashes.add(q["sha256"])
            query_fps.append(fp)
    rng = np.random.default_rng(6337)
    for category in groups:
        group = sorted(groups[category], key=lambda p: p["product_id"])
        groups[category] = [group[j] for j in rng.permutation(len(group))]
    cycle = ["sofa", "sofa", "chair", "table", "bed", "desk", "cabinet", "shelving", "lamp"]

    def take(count):
        result = []
        while any(groups.values()) and len(result) < count:
            for category in cycle:
                if groups[category] and len(result) < count:
                    result.append(groups[category].pop(0))
        return result

    absent, present = take(20), take(50)
    queries = [
        {
            **p["images"][2],
            "product_id": p["product_id"],
            "label": p["label"],
            "selection_category": p["selection_category"],
            "true_product_in_catalog": exists,
            "identity_scope": "ABO_listing",
        }
        for exists, selected in [(True, present), (False, absent)]
        for p in selected
    ]
    unknown_ids = {p["product_id"] for p in absent}
    known_ids = {p["product_id"] for p in present}
    distractors = sorted(
        p["product_id"] for p in products if p["product_id"] not in unknown_ids | known_ids
    )
    distractors = [distractors[j] for j in rng.permutation(len(distractors))]
    order = sorted(known_ids) + distractors
    by_id = {p["product_id"]: {**p, "images": p["images"][:2]} for p in products}
    sizes = [
        size for size in [100, 300, 600, 1000] if size <= len(order) and size >= len(known_ids)
    ]
    galleries = {str(size): [by_id[pid] for pid in order[:size]] for size in sizes}
    validate_nested_galleries(galleries, queries)
    for size, gallery in galleries.items():
        atomic_json(
            ROOT / f"gallery-{size}.json",
            {"schema_version": 2, "products": gallery, "experimental": True},
        )
    protocol = {
        "seed": 6337,
        "queries": queries,
        "gallery_ids": {size: [p["product_id"] for p in gal] for size, gal in galleries.items()},
        "exclusions": exclusions,
        "known_queries": len(present),
        "unknown_queries": len(absent),
        "query_categories": dict(Counter(q["selection_category"] for q in queries)),
        "policy": "Same fixed fresh query identities across deterministic nested galleries; two source references/product; global max cosine and flat FAISS, unchanged production aggregation. Query/source crop duplicate screening before scores. No filtering by provisional image quality.",
        "untouched_scope": "Fresh listing IDs and source query images not previously evaluated; screened against all prior image IDs/hashes/crop fingerprints. References can overlap related variants; manufacturer/session independence unknown.",
        "unknown_rules": {
            "similarity": 0.85,
            "margin": 0.01,
            "origin": "Frozen historical Phase6D2 diagnostic rules, not retuned or deployed",
        },
        "bootstrap": {"seed": 6338, "samples": 5000},
        "household_verified_queries": 0,
    }
    atomic_json(ROOT / "scaling-protocol.json", protocol)
    atomic_json(
        ROOT / "scaling-protocol-lock.json",
        {
            "protocol_sha256": digest(ROOT / "scaling-protocol.json"),
            "gallery_sha256": {size: digest(ROOT / f"gallery-{size}.json") for size in galleries},
            "scoring_seen_before_lock": False,
        },
    )
    print(
        {
            "known": len(present),
            "unknown": len(absent),
            "sizes": sizes,
            "excluded_query_candidates": len(exclusions),
        }
    )


def embed():
    lock = read_json(ROOT / "scaling-protocol-lock.json")
    assert digest(ROOT / "scaling-protocol.json") == lock["protocol_sha256"]
    products = all_products()
    rows = {}
    for p in products:
        for image in p["images"]:
            if image["image_id"] in rows and rows[image["image_id"]]["sha256"] != image["sha256"]:
                raise ValueError("Image ID maps to different source bytes")
            rows[image["image_id"]] = image
    rows = list(rows.values())
    model_folder = Path(".cache/models/siglip2-base-75de2d55").resolve()
    signature = {
        "revision": REVISION,
        "preprocessor_sha256": digest(model_folder / "preprocessor_config.json"),
        "feature": "model-default-get_image_features-l2",
        "weights_sha256": digest(model_folder / "model.safetensors"),
    }
    cache = ROOT / "features"
    cache.mkdir(exist_ok=True)
    old = Path("reports/phase6d2-data-quality")
    assert read_json(old / "inference-report.json")["model"]["requested_revision"] == REVISION
    old_rows = read_json(old / "vector-image-rows.json")["images"]
    old_values = np.load(old / "image-vectors.npy")
    old_map = {r["image_id"]: (r["sha256"], v) for r, v in zip(old_rows, old_values)}
    seeded = 0
    for row in rows:
        if row["image_id"] in old_map:
            sha, value = old_map[row["image_id"]]
            assert sha == row["sha256"]
            key = hashlib.sha256(
                json.dumps([row["image_id"], sha, signature], sort_keys=True).encode()
            ).hexdigest()
            path = cache / (key + ".npy")
            if not path.exists():
                with path.with_suffix(".tmp").open("wb") as stream:
                    np.save(stream, value)
                os.replace(path.with_suffix(".tmp"), path)
                seeded += 1
    from furniture_bench.runtime import get_torch

    torch = get_torch()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = Siglip2Embedder(
        str(model_folder), device="cpu", allow_download=False, revision=REVISION, batch_size=8
    )

    def encode(paths):
        return model.encode([load_image(p) for p in paths])

    started = perf_counter()
    vectors, report = incremental_embeddings(rows, cache, signature, encode)
    np.save(ROOT / "image-vectors.npy", vectors)
    atomic_json(ROOT / "vector-image-rows.json", {"images": rows})
    report.update(
        model=model.metadata(),
        model_signature=signature,
        seeded_prior_vectors=seeded,
        wall_seconds=perf_counter() - started,
        unique_images=len(rows),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    name = (
        "embedding-report.json"
        if not (ROOT / "embedding-report.json").exists()
        else "embedding-resume-report.json"
    )
    atomic_json(ROOT / name, report)
    print(report)


def evaluate():
    if (ROOT / "scaling-evaluation.json").exists():
        raise ValueError("Evaluation saved; no repeated selection")
    protocol = read_json(ROOT / "scaling-protocol.json")
    lock = read_json(ROOT / "scaling-protocol-lock.json")
    assert digest(ROOT / "scaling-protocol.json") == lock["protocol_sha256"]
    rows = read_json(ROOT / "vector-image-rows.json")["images"]
    values = np.load(ROOT / "image-vectors.npy")
    by_image = {r["image_id"]: (r["sha256"], v) for r, v in zip(rows, values)}

    def matrix(images):
        for image in images:
            assert by_image[image["image_id"]][0] == image["sha256"]
        return np.stack([by_image[i["image_id"]][1] for i in images])

    queries = protocol["queries"]
    qvectors = matrix(queries)
    report, csv_rows, category_rows = {}, [], []
    baseline_ranks = None
    for size in sorted(protocol["gallery_ids"], key=int):
        file = ROOT / f"gallery-{size}.json"
        assert digest(file) == lock["gallery_sha256"][size]
        catalog = read_json(file)
        gvectors = matrix([i for p in catalog["products"] for i in p["images"]])
        start = perf_counter()
        index = MultiImageCatalog.build(
            gvectors, catalog, read_json(ROOT / "embedding-report.json")["model"], Path.cwd()
        )
        build_seconds = perf_counter() - start
        directory = ROOT / "indexes" / size
        start = perf_counter()
        index.save(directory)
        save_seconds = perf_counter() - start
        try:
            start = perf_counter()
            results, search_seconds = index.search(qvectors, k=int(size))
            wall = perf_counter() - start
        finally:
            index.close()
        ranks = [
            next(
                (j + 1 for j, r in enumerate(result) if r["product_id"] == query["product_id"]),
                10**9,
            )
            for query, result in zip(queries, results)
        ]
        if baseline_ranks is None:
            baseline_ranks = ranks
        present = [i for i, q in enumerate(queries) if q["true_product_in_catalog"]]
        absent = [i for i, q in enumerate(queries) if not q["true_product_in_catalog"]]
        delta = paired_interval(baseline_ranks, ranks, present, seed=6338)
        top = np.array([r[0]["similarity"] for r in results])
        margins = top - np.array([r[1]["similarity"] for r in results])
        metrics = summarize(ranks, queries, present)
        sofa = summarize(ranks, queries, [i for i in present if queries[i]["label"] == "sofa"])
        index_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        row = {
            "gallery_products": int(size),
            "reference_images": len(gvectors),
            "known_queries": len(present),
            "unknown_queries": len(absent),
            "recall1": metrics["recall_at_1"],
            "recall5": metrics["recall_at_5"],
            "mrr": metrics["mrr"],
            "sofa_queries": sofa["queries"],
            "sofa_recall1": sofa["recall_at_1"],
            "sofa_recall5": sofa["recall_at_5"],
            "recall5_delta_vs_smallest95ci_low": delta[0],
            "recall5_delta_vs_smallest95ci_high": delta[1],
            "known_identity_gallery_coverage": 1.0,
            "similarity_unknown_false_accept": float(np.mean(top[absent] >= 0.85)),
            "similarity_known_false_reject": float(np.mean(top[present] < 0.85)),
            "margin_unknown_false_accept": float(np.mean(margins[absent] >= 0.01)),
            "margin_known_false_reject": float(np.mean(margins[present] < 0.01)),
            "index_build_seconds": build_seconds,
            "index_save_seconds": save_seconds,
            "faiss_search_batch_seconds_including_ipc": search_seconds,
            "product_search_batch_wall_seconds": wall,
            "index_bytes": index_bytes,
            "evaluation_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        }
        csv_rows.append(row)
        for label in sorted({q["label"] for q in queries}):
            category_rows.append(
                {
                    "gallery_products": int(size),
                    "category": label,
                    **summarize(
                        ranks, queries, [i for i in present if queries[i]["label"] == label]
                    ),
                }
            )
        report[size] = {
            "metrics": row,
            "ranks": ranks,
            "top5": [r[:5] for r in results],
            "top1_similarity": top.tolist(),
            "product_top1_top2_margin": margins.tolist(),
        }
        print(size, metrics, flush=True)
    for name, records in [
        ("catalog-scaling-results.csv", csv_rows),
        ("scaling-category-metrics.csv", category_rows),
    ]:
        with (ROOT / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    largest = max(protocol["gallery_ids"], key=int)
    loaded = MultiImageCatalog.load(ROOT / "indexes" / largest)
    try:
        matches, _ = loaded.search(qvectors[:1], k=5)
    finally:
        loaded.close()
    assert [r["product_id"] for r in matches[0]] == [
        r["product_id"] for r in report[largest]["top5"][0]
    ]
    atomic_json(
        ROOT / "scaling-evaluation.json",
        {
            "sizes": report,
            "queries": queries,
            "protocol_sha256": digest(ROOT / "scaling-protocol.json"),
            "index_reload_top5_preserved": True,
            "production_changed": False,
        },
    )


def search(args):
    model = Siglip2Embedder(
        str(Path(".cache/models/siglip2-base-75de2d55").resolve()),
        device="cpu",
        allow_download=False,
        revision=REVISION,
    )
    vector, seconds = model.encode([load_image(args.image)])
    index = MultiImageCatalog.load(args.index)
    try:
        results, latency = index.search(vector, k=5)
    finally:
        index.close()
    print(
        json.dumps(
            {
                "matches": results[0],
                "inference_seconds": seconds,
                "search_seconds": latency,
                "exact_identification": "No verified exact match.",
                "physical_fit_verified": False,
            },
            indent=2,
        )
    )


def build(args):
    """Rebuild any trusted licensed manifest into a new experimental index folder."""
    if not args.catalog or not args.output:
        raise ValueError("build requires --catalog and a new --output directory")
    manifest = Path(args.catalog)
    catalog = read_json(manifest)
    validate_catalog(catalog, manifest.parent)
    if Path(args.output).exists():
        raise ValueError("Index output exists; preserve previous index")
    rows = read_json(ROOT / "vector-image-rows.json")["images"]
    values = np.load(ROOT / "image-vectors.npy")
    lookup = {r["image_id"]: (r["sha256"], v) for r, v in zip(rows, values)}
    vectors = []
    for product in catalog["products"]:
        for image in product["images"]:
            if image["image_id"] not in lookup or lookup[image["image_id"]][0] != image["sha256"]:
                raise ValueError("Missing or mismatched embedding; run incremental embed first")
            vectors.append(lookup[image["image_id"]][1])
    index = MultiImageCatalog.build(
        np.stack(vectors),
        catalog,
        read_json(ROOT / "embedding-report.json")["model"],
        manifest.parent,
    )
    try:
        index.save(args.output)
    finally:
        index.close()
    print(
        {
            "products": len(catalog["products"]),
            "images": len(vectors),
            "index": args.output,
            "production_changed": False,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "embed", "evaluate", "search", "build"])
    parser.add_argument("--catalog")
    parser.add_argument("--output")
    parser.add_argument("--image")
    parser.add_argument("--index", default=str(ROOT / "indexes/1000"))
    args = parser.parse_args()
    if args.stage == "build":
        build(args)
    elif args.stage == "search":
        if not args.image:
            parser.error("search requires --image")
        search(args)
    else:
        {"prepare": prepare, "embed": embed, "evaluate": evaluate}[args.stage]()
