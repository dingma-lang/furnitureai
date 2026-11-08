"""Build reproducible Phase 6D.2 reports and failure views without model execution."""

import csv
import hashlib
import html
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from furniture_bench.data import digest, read_json, write_json

ROOT = Path("reports/phase6d2-data-quality")


def readable_prose(text):
    parts = re.split(r"(`[^`]*`|https?://\S+)", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", parts[i])
        parts[i] = re.sub(r"(?<=\d)(?=(?:s|GB|MB)\b)", " ", parts[i])
    return "".join(parts)


def main():
    result = read_json(ROOT / "fresh-evaluation.json")
    protocol = read_json(ROOT / "fresh-protocol.json")
    quality = read_json(ROOT / "quality-summary.json")
    inference = read_json(ROOT / "inference-report.json")
    gallery = read_json(ROOT / "fresh-gallery.json")
    products = {p["product_id"]: p for p in gallery["products"]}
    image_root = Path(gallery["data_root"])
    browser = {}
    for p in ROOT.glob("*-browser/*tests.json"):
        value = read_json(p)
        browser[p.parent.name] = {
            "passed": value.get("passed", len(value["checks"])),
            "page_errors": value.get("page_errors", []),
        }
    review_browser = read_json(ROOT / "review-browser-tests.json")
    assert sum(v["passed"] for v in browser.values()) == 59
    assert review_browser["passed"] == 8
    assert not any(v["page_errors"] for v in browser.values())
    tests = (ROOT / "regression-tests.log").read_text()
    assert "Ran 230 tests" in tests and "\nOK\n" in tests
    prior = read_json(ROOT / "preservation-before.json")
    changed = [p for p, sha in prior.items() if not Path(p).is_file() or digest(p) != sha]
    append_only = []
    for path in list(changed):
        if path != "reports/phase6d-product-identification/server-current.log":
            continue
        prefix = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for line in stream:
                prefix.update(line)
                if prefix.hexdigest() == prior[path]:
                    append_only.append(path)
                    changed.remove(path)
                    break
    assert not changed, changed
    lock = read_json(ROOT / "protocol-lock.json")
    assert digest(ROOT / "fresh-protocol.json") == lock["protocol_sha256"]
    assert digest(ROOT / "fresh-gallery.json") == lock["gallery_sha256"]
    assert digest(ROOT / "unknown-selection-lock.json") == result["unknown_selection_sha256"]
    write_json(
        ROOT / "verification-summary.json",
        {
            "python_baseline": 214,
            "python_final": 230,
            "existing_browser": browser,
            "review_browser": review_browser,
            "browser_total": 67,
            "prior_report_and_production_files_hash_checked": len(prior),
            "prior_files_changed": changed,
            "existing_operational_logs_append_only_prefix_verified": append_only,
            "production_retrieval_changed": False,
            "owner_quality_annotations_created": 0,
            "household_verified_queries": 0,
            "model_weights_downloaded": False,
            "source_hash_validation": "All1140 source images checked during inference assembly and manifest audit",
            "protocol_and_selection_locks_valid": True,
        },
    )
    detector = read_json(ROOT / "detector-predictions/predictions.json")
    owl = sum(r["timing_seconds"].get("owlv2", 0) for r in detector["images"])
    write_json(
        ROOT / "runtime-costs.json",
        {
            "siglip": inference,
            "owlv2_images": len(detector["images"]),
            "owlv2_inference_seconds": owl,
            "owlv2_image_stage_total_seconds": sum(
                r["timing_seconds"].get("total", 0) for r in detector["images"]
            ),
            "owlv2_errors": [r["errors"] for r in detector["images"] if r["errors"]],
            "cached_scoring_seconds": result["ranking_seconds"],
            "units": "seconds, RSS bytes",
            "limits": "RSS is whole SigLIP process peak including reused vectors, not individual config peaks. OWLv2 peak was not sampled. Cached scoring excludes inference, preprocessing, browser, IPC and worker startup.",
        },
    )
    queries, ranks = result["queries"], result["query_ranks"]
    failures = [
        i
        for i, q in enumerate(queries)
        if q["split"] == "heldout" and q["true_product_in_catalog"] and ranks[i] > 5
    ]
    folder = ROOT / "failure-gallery"
    html_rows = []
    for number, i in enumerate(failures):
        query = queries[i]
        ids = [query["product_id"]] + result["top5"][i]
        paths = [Path(query["file"])] + [
            image_root / products[pid]["images"][0]["file"] for pid in ids
        ]
        labels = ["Query " + query["product_id"], "Correct listing " + ids[0]] + [
            f"Top {n + 1}: {pid}" for n, pid in enumerate(ids[1:])
        ]
        canvas = Image.new("RGB", (1400, 290), "#f5f4ed")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 5),
            f"Fresh held-out {query['label']} | correct rank {ranks[i]} | listing identity only, not household ground truth",
            fill="black",
            font=ImageFont.load_default(size=15),
        )
        for n, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as im:
                thumbnail = ImageOps.contain(im.convert("RGB"), (190, 195))
            canvas.paste(thumbnail, (n * 200 + (190 - thumbnail.width) // 2, 36))
            draw.text((n * 200 + 3, 245), label, fill="black", font=ImageFont.load_default(size=10))
        name = f"heldout-failure-{number + 1}.png"
        canvas.save(folder / name)
        html_rows.append(
            f'<p>{html.escape(query["product_id"])}; true rank {ranks[i]}</p><img style="max-width:100%" src="{name}" alt="Query, correct reference and five candidates">'
        )
    (folder / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen held-out errors</title><body><h1>Listing retrieval errors</h1><p>Generated after final scoring; no re-selection. Images © Amazon.com, CC BY 4.0. Product IDs remain distinct.</p>'
        + "".join(html_rows)
        + "</body></html>"
    )
    # Reuse the examined development ablation, not a new evaluation or selection.
    rows = read_json("reports/phase6d1-retrieval-optimization/development-results.json")[
        "individual_and_combination"
    ]
    with (ROOT / "reference-quality-development.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scope",
                "configuration",
                "queries",
                "recall1",
                "recall5",
                "sofa_recall5",
                "new_selection",
            ],
        )
        writer.writeheader()
        for row in rows[:4]:
            writer.writerow(
                {
                    "scope": "historical_examined_Phase6D1_development",
                    "configuration": row["name"],
                    "queries": row["metrics"]["queries"],
                    "recall1": row["metrics"]["recall_at_1"],
                    "recall5": row["metrics"]["recall_at_5"],
                    "sofa_recall5": row["sofa"]["recall_at_5"],
                    "new_selection": False,
                }
            )
    counts = protocol["counts"]
    categories = sorted({p["label"] for p in products.values()})
    table = "| Category | Gallery IDs | Dev present / absent | Held-out present / absent |\n|---|---:|---:|---:|\n"
    for label in categories:
        table += f"| {label} | {sum(p['label'] == label for p in products.values())} | {counts['development'].get(label + '_present', 0)} / {counts['development'].get(label + '_absent', 0)} | {counts['heldout'].get(label + '_present', 0)} / {counts['heldout'].get(label + '_absent', 0)} |\n"
    (ROOT / "evaluation-protocol.md").write_text(f"""# Phase 6D.2 frozen protocol

## Preserved baseline and rights

Production remains Google's cached `google/siglip2-base-patch16-224`, pinned revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The official fixed224 checkpoint declares SigLIP architecture and the existing AutoModel adapter is retained. Preprocessing is the cached model-default RGB resize224×224, bilinear, mean/std0.5; aspect ratio is warped. Image features are L2-normalized float32. Existing isolated FAISS IndexFlatIP retrieves images and maximum cosine aggregates into distinct ABO listing IDs. No training, quality filtering, identity merging, or deployed threshold change occurred. User choices and photo-to-3D links are untouched. The offline global baseline disables application preference/category restrictions to isolate retrieval. The unchanged application requires a confirmed observation and uses its confirmed category; benchmark scores are not full application accuracy.

Official [ABO dataset terms](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) publish CC BY 4.0 for images/metadata. Amazon.com attribution, full license, metadata, source URLs and hashes are retained. Source images are official small/max256 derivatives, not high-resolution originals. No retailer pages were scraped. Code licensing for another dataset is not assumed to license its images.

## Fresh cohort

100 previously unused identities / 300 images were selected without retrieval scores from cached official shard0 metadata. Both previous 140-product selections are excluded by listing ID and source image ID; no previous byte hash overlaps. Download cost: 2,039,230 image bytes; metadata/checkpoints reused. Generic downloader's catalog/query files are provisional and are NOT this study's frozen roles: use `fresh-gallery.json` and `fresh-protocol.json`.

Crop-normalized duplicate screening excludes four products similar to prior images and one within-product duplicate, leaving 95 identities. This heuristic uses SHA256 or dHash Hamming<=6 AND mean32×32 RGB absolute difference<14, after a near-white foreground bounding crop; it is not proof against every semantic duplicate. Query-to-entire-gallery screening excludes no additional queries. Exclusions are preserved in the protocol. No replacement is selected after scoring. All downloaded sources, including excluded images, remain in the review audit.

Seed6227 stratifies category then assigns disjoint development/held-out identities; about one fifth per category/split are absent entirely from gallery. Present identities contribute two independent source photo files to the gallery and a third to queries. Distinct source images are not proof of independent physical capture sessions; those are unknown. The gallery has 79 identities / 158 images; dev40 present+8 absent, held-out39 present+8 absent. No bed/desk identities remained eligible in the unused portion of this shard; prior audits still include those categories. This is a limited five-category catalog study.

{table}

`protocol-lock.json` hashes protocol/gallery before inference or scoring. Unknown rejection was selected only on development scores and locked in `unknown-selection-lock.json` before final evaluation. No fresh held-out ranking informed parameters. The baseline is frozen; no new optimized retrieval method was selected or deployed. Fresh final evaluation is saved once and now examined; future tuning needs another untouched set.

## Metrics and category scenarios

Recall@1/5 and MRR use distinct listing identities, present queries only. Report category and sofa counts. Absent queries are excluded from recall denominators and evaluated separately. Bootstrap uses seed6228 and5,000 paired resamples for category differences; main Recall@5 uses resampling of query identities. Small samples and diagnostic multiplicity limit inference.

Before OWLv2 inference, select at most two present query photos per category/split: 10 development+10 held-out. Use the highest-scoring actual detection. Strict diagnostic and soft0.08 use global fallback for missing category, score<0.35, or unavailable category. Catalog categories simulate correct user confirmation only as an oracle; **actual human confirmations=0**. Synthetic incorrect confirmation uses another category; unconfirmed category evidence falls back globally. These are conditioned catalog-query experiments, not a household full-pipeline detection/localization benchmark. No GT boxes or verified household identities exist.

Unknown diagnostics compare maximum product cosine and top1-minus-top2 **distinct-product** margins. The predeclared similarity grid is0.70/0.75/0.80/0.85/0.90/0.95; margin grid0/.005/.01/.02/.04/.08. Development objective is balanced known-accept/unknown-reject rate; ties prefer known acceptance then grid order. Selected thresholds:0.85 and0.01. No held-out re-selection. Similarity is not calibrated identity probability. Existing no-verified-match/user-decision behavior stays enabled; no threshold is deployed.

## Quality and household evidence

All1,140 images /380 listings are audited across three **separate** cohorts. Quality suggestions use nearest saved SigLIP visual reference per class from20 prior assistant-described sofa queries. Fixed minimum cosine0.65 and class margin0.03 otherwise mark ambiguous. This is an untrained, sofa-biased, provisional review prioritizer; classes for occlusion/multiple objects have no reliable prototypes. It is not an independent quality classifier benchmark, and cannot establish how many images truly show the complete object. No owner-verified labels exist. All visibility and suitability fields remain unknown until attested review.

Quality/retrieval associations use the **examined development** cohort only and are descriptive, not causal. Using the same embedding space for provisional quality and retrieval also confounds these associations. Historical main-vs-multiview ablations are reused as historical evidence, with their held-out failure documented. High-similarity listing pairs/shared reported brand-model values enter a queue; no identities are merged.

The manifest explicitly separates active catalog references, catalog queries, unknown queries, reserved images, household queries and synthetic data. No household or synthetic example is substituted into the fresh accuracy denominator. Household verified queries=0; camera/view, physical sessions, real condition labels, and independent manufacturer identity remain unknown. See `real-world-evaluation-status.md` and `household-import-template.json` for acquisition/import controls, explicit permission, identity evidence, session splits and immutable-source review history.
""")
    held = result["metrics"]["heldout"]
    quality_rows = "\n".join(
        f"| {label} | {count} |" for label, count in sorted(quality["provisional_counts"].items())
    )
    category_rows = "\n".join(
        f"| {r['scenario']} | {r['queries']} | {r['recall_at_1']:.1%} | {r['recall_at_5']:.1%} |"
        for r in result["category_diagnostics"]
        if r["split"] == "heldout"
    )
    unknown_rows = "\n".join(
        f"| {r['family']} | {r['threshold']} | {r['known_accept_rate']:.1%} | {r['unknown_false_accept_rate']:.1%} |"
        for r in result["unknown_rejection"]
        if r["split"] == "heldout"
    )
    (ROOT / "RESULTS.md").write_text(f"""# Phase 6D.2 — dataset quality and retrieval evaluation

The production system is preserved. This phase adds a provisional catalog audit, an owner-review/export/import workflow, documentary identity/rights validation, and a fresh catalog-only evaluation. **Verified household product-identity queries:0. No real-home exact-product accuracy is claimed.**

## Verified work and independent metrics

Audited1,140 unchanged source images across380 ABO listing identities (140 historical+140 Phase6D.1+100 fresh). Source hashes were checked. The fresh download used2.04 MB of official CC BY4.0 images, cached metadata and the existing SigLIP checkpoint. Four prior-near-duplicate products and one within-product duplicate were excluded before scoring. The fresh gallery contains79 identities/158 references, with40 present+8 absent development queries and39 present+8 absent held-out queries. Bed/desk are absent from this fresh shard remainder, not silently scored as success.

| Fresh frozen baseline | Present queries | Recall@1 | Recall@5 | MRR | Sofa R@1 / R@5 |
|---|---:|---:|---:|---:|---:|
| Development |40|70.0%|87.5%|0.7568|50.0% /70.0% (10)|
| Held-out |39|64.1%|84.6%|0.7216|70.0% /80.0% (10)|

Held-out Recall@5 bootstrap95% interval: {held["recall5_bootstrap95ci"][0]:.1%}–{held["recall5_bootstrap95ci"][1]:.1%}. See `fresh-category-metrics.csv` for all category counts, Recall@1/5 and MRR. These numbers measure **official listing association across catalog photographs**, not verified manufacturer/model recognition. The smaller79-product gallery and new identities make them unsuitable as before/after gains over119-query Phase6D or56-query Phase6D.1 results. No new production retrieval improvement is claimed.

## Image and gallery quality

| Provisional suggestion, not ground truth | Images |
|---|---:|
{quality_rows}

The audit distinguishes staged interiors, parts, material close-ups and diagrams. However, **owner-verified whole-object count is unknown**, with zero owner-verified quality annotations. All1,140 whole-visibility/query-suitability/gallery-suitability fields stay unknown. A small, sofa-biased reference bank and embedding similarity generate provisional suggestions;529 images are ambiguous. Occlusion and multiple-object classification are unsupported rather than fabricated. These suggestions must not be used to discard references automatically.

`dataset-quality-audit.csv` includes image/product IDs, dimensions, source/license, hashes, roles, confidence semantics and annotation provenance. `product-gallery-audit.csv` records active references, retained main photos, unresolved whole-object availability, unverified view angles, reported brand/model and possible variant issues. {quality["possible_duplicate_listing_flags"]} listings have a high-similarity alternate flag; shared reported brand/model values also enter the review queue. These are possible variants/duplicates, **not verified equivalences**; zero IDs were merged. Metadata values are reported, not independently authenticated.

Historical assistant visual review of20 sofas found4 material close-ups with0/4 R@5 hits and2 whole-object queries with2/2 hits. This is descriptive, tiny and not owner-verified. On examined Phase6D.1 development data, provisional material-closeup queries have1/12 R@5 hits, staged9 queries have3/9, and the single provisional whole-object query hits. The classifier and retrieval share embeddings, so this association is confounded and cannot prove image quality causes the difference. `exploratory-quality-retrieval.csv` gives every group. Whole-object ground truth must be reviewed before a quality-filter experiment is justified.

`reference-quality-development.csv` reuses earlier main-only/max/centroid/top-two ablations; it is explicitly historical. Main-only improved that development set but regressed on its held-out sofas. This phase does not retune on that examined held-out set. Missing viewpoints, close-up-only evidence and similar variants remain plausible limitations; the evidence does not isolate all errors as a SigLIP model defect.

## Category information

| Fresh held-out scenario | Queries | Recall@1 | Recall@5 |
|---|---:|---:|---:|
{category_rows}

These are the same10 preselected catalog queries, not the39-query full result. Oracle categories simulate correct user confirmation; actual human confirmations=0. Oracle R@5 increases by10 points, with paired95% interval0–30 points, insufficient to establish a dependable gain. Actual OWLv2 filtering changes one top-one outcome but leaves R@5 unchanged; this differs from the previous small subset and is not evidence to deploy strict unconfirmed filtering. Deliberately wrong confirmed categories exclude all correct IDs. Missing/unconfirmed categories fall back globally. No household full-pipeline accuracy or detection precision/recall can be measured without identity/box ground truth.

## Unknown products

| Development-selected rule | Threshold | Held-out known accepted (39) | Held-out unknown false accepts (8) |
|---|---:|---:|---:|
{unknown_rows}

Cosine threshold0.85 rejects7/8 unknowns but also8/39 known products. The product-margin threshold0.01 accepts6/8 unknowns and rejects4/39 knowns. Neither small-sample result supports dependable unknown recognition or calibrated probabilities. No rejection threshold is deployed. Top-five visually similar alternatives, explicit “No verified exact match”, independent user confirmation/rejection and saved decisions remain intact.

## Review tools and household gap

`image-review.html` provides search, filtering, pagination, named-reviewer fields, explicit human-attestation export, persistent **unverified drafts**, and source links. The importer checks source hashes/signatures, permissions, roles, duplicate bytes, session/identity splits and evidence, preserves original quality history and refuses overwrites. Automation exported a draft and verified that it is rejected as human review. No human annotations were invented. `manual-review-queue.csv` has {quality["manual_queue_rows"]} entries covering all images, possible variants and the household identity gap.

Real licensed room photos remain useful unknown-identity demos only. No real wear, lighting, camera-angle, stain, scratch or upholstery-change retrieval accuracy is established. No new synthetic stress test is presented as real household evidence. `real-world-evaluation-status.md` documents sources considered, rights limitations and a concrete acquisition plan; `household-import-template.json` is ready to fill but intentionally fails until actual image permission and documentary identity evidence exist. No permission messages were sent or retail review pages scraped.

## Resource costs and regression evidence

Real pretrained SigLIP2 inference encoded386 missing images and reused754 saved vectors: {inference["inference_seconds"]:.2f}s model inference, {inference["wall_seconds"]:.2f}s job wall time (load/preprocessing included), {inference["peak_process_rss_bytes"] / 1e9:.2f}GB peak process RSS. CPU, batch8, four threads. Actual OWLv2 category inference on20 images took {owl:.2f}s; no errors. Its peak memory was not separately sampled. Cached scoring of all95 queries against79 identities took {result["ranking_seconds"] * 1000:.2f}ms; this is not cold-start/browser latency. FAISS/direct cosine max difference: {result["faiss_vs_cosine_max_error"]:.2g}. `runtime-costs.json` retains definitions and timings.

Baseline214 Python tests passed; final **230 Python tests and67 browser checks passed** (59 existing +8 review checks), with no page errors. Browser suites exercise real licensed uploads and existing model/cache paths, category corrections, product confirmation/rejection, photo-to-3D identity, replacement, save/reopen and uncalibrated-fit restrictions. Review screenshots are `image-review-before.png` and `image-review-after-draft.png`; previous-workflow screenshots/recording are in `*-browser/`. {len(prior) - len(append_only)} previous report/production files match their pre-phase hashes; the existing live server log only appended browser requests, with its original prefix hash verified. Original source hashes remain valid. No neural training or new model weights were introduced.

## Recommendation

Retain production ranking and no-verified-match behavior. Prioritize owner-reviewed whole-object/reference annotations and independently documented household identity/variant examples with explicit commercial evaluation rights, multiple real sessions/views and true out-of-catalog products. Acquire independent untouched evaluation data before selecting any quality filter or category policy. Better references may help, but current provisional labels cannot establish the cause or magnitude. Do not begin fine-tuning until evidence, permissions and session-level splits are adequate; no additional GPU/model download is needed for this review phase. Physical placement remains subject to existing calibration/measurement safeguards.

Startup/import/reproduction: `../../docs/dataset-quality.md`. Full created/modified paths: `FILES.txt`.
""")
    for name in ["RESULTS.md", "evaluation-protocol.md"]:
        path = ROOT / name
        path.write_text(readable_prose(path.read_text()))
    print(
        {
            "python": 230,
            "browser": 67,
            "preserved_files": len(prior),
            "retrieval_errors_visualized": len(failures),
        }
    )


if __name__ == "__main__":
    main()
