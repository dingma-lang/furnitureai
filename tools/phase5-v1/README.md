# Phase 5 isolated detection preparation

Use `/home/tristary/furniture-ai-ml/.venv/bin/python -B`. No dependencies were installed. These tools contain no downloader, trainer or production deployment operation. All JSON outputs use exclusive creation. Existing Phase 2B review gates are unchanged.

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B -m unittest discover -s tools/phase5-v1 -p 'test_*.py' -v
```

The completed, no-inference local run is in `reports/phase5-detection-preparation-v1/local-diagnostics-v1/`. Open `gallery.html`. To repeat it, choose a **fresh** output directory:

```bash
.venv/bin/python -B tools/phase5-v1/diagnose_rooms.py \
  --out reports/phase5-detection-preparation-v2/local-diagnostics-v1
```

`detection_prep.quality` measures crop geometry, border distances, overlaps, detector score, image contrast and adjacent-pixel variation. Its flags and thresholds are unfitted heuristics. No whole-object annotation, calibrated confidence or automatic error label is produced. Disagreement alone does not exclude a crop. Original boxes and classifier outputs remain in Phase 4.

## Approval-gated metadata stage

**No Open Images CSV/JSON has been downloaded.** Exact eight objects, HEAD evidence and byte budgets are in `reports/phase5-detection-preparation-v1/metadata-acquisition-proposal.json`. Wait for explicit metadata-download approval. No command here downloads anything.

After an approved acquisition has saved the specified files under `data/open-images/metadata/phase5-v1/`, run this offline audit (paths are future inputs, not existing data):

```bash
.venv/bin/python -B tools/phase5-v1/audit_openimages.py \
 --dictionary data/open-images/metadata/phase5-v1/oidv7-class-descriptions-boxable.csv \
 --hierarchy data/open-images/metadata/phase5-v1/bbox_labels_600_hierarchy.json \
 --boxes data/open-images/metadata/phase5-v1/validation-annotations-bbox.csv data/open-images/metadata/phase5-v1/test-annotations-bbox.csv \
 --labels data/open-images/metadata/phase5-v1/validation-annotations-human-imagelabels-boxable.csv data/open-images/metadata/phase5-v1/test-annotations-human-imagelabels-boxable.csv \
 --images data/open-images/metadata/phase5-v1/validation-images-with-rotation.csv data/open-images/metadata/phase5-v1/test-images-with-rotation.csv \
 --out reports/phase5-open-images-metadata-audit-v1
```

It resolves exact display names to MIDs, checks hierarchy membership, checks bbox schemas/flags/coordinates, retains human positive and negative image-level labels, counts boxes/images, joins image rights/source records and stops on source partition or label contradictions. The audit does not assert that images are indoors or that a historic licence claim establishes current rights. Missing exact names remain unresolved. It does not expand native taxonomy ancestors/children or calculate official Open Images mAP.

## Future approved pilot and bundle contract

Target **at most 150** photographs: 60 training and 30 development from official validation, 60 frozen evaluation from official test. This repurposing is an explicitly custom pilot, never an official Open Images leaderboard result. Select rare furniture categories first using deterministic seven-category rounds; no quota is filled with ambiguous mappings or uncertain identities. `select-pilot` operates **after** image rights, indoor scene suitability and identity/duplicate screening, not before images are available:

```bash
.venv/bin/python -B tools/phase5-v1/detection_prep.py select-pilot \
 --candidates APPROVED_PILOT/candidates.json --out FRESH_SELECTION.json
```

Candidates contain image_id, actual SHA-256, source_identity (canonical source-photo ID), conservative identity_group (source photographer/scene family), source_partition, licence/indoor/annotation eligibility, identity_uncertain and box_counts. Absent evidence is not permission to set eligibility true. Near duplicates, unknown orientation and unresolved scene/product relationships must be quarantined before final grouping. Selection is immutable after freezing. `assert_isolation` rejects cross-partition image hashes, source photo IDs and known groups, and protected ABO hashes/IDs/groups read from metadata only.

A future `phase5-public-boxes-v1` bundle must contain:

- `source: "Open Images"`; `class_dictionary: {file,url,sha256}`.
- `annotation_files`: records with `partition: validation|test`, `kind: boxes|human_labels`, local relative `file`, official URL and SHA-256. Four annotation records are needed for a full pilot.
- `images`: local relative photo path, `image_id`, `sha256`, `source_identity`, `identity_group`, `split`, `source_partition`, `annotations`, `verified_classes`, rights/attribution, `license_verified`, `annotations_verified`, `orientation_verified`, `identity_uncertain`, `annotation_provenance: published_open_images`.

`annotations` are the exact rows returned by `parse_annotations`; `verified_classes` is the union of positive and negative exact native labels returned by `parse_image_labels` for that image. The validator reconstructs both from hashed source CSVs. Changing GT or setting a boolean cannot substitute for these records. Box coordinates are normalized xyxy in the annotation-consistent orientation. Resolve Open Images Rotation versus EXIF/CVDF orientation once, record the transform, and verify overlays on a small sample before setting `orientation_verified`; do not rotate twice. Licence verification is source-access/attribution evidence, not a human furniture label. No Phase 2B human-review flags are synthesized.

## Baseline and future detector commands

These commands are prepared for approved, audited future data and deliberately fail today because no such bundle exists:

```bash
.venv/bin/python -B tools/phase5-v1/run_baseline.py \
 --bundle APPROVED_PILOT/bundle.json \
 --out reports/phase5-open-images-baseline-v1 --device cuda

.venv/bin/python -B tools/phase5-v1/detection_prep.py evaluate \
 --bundle APPROVED_PILOT/bundle.json \
 --predictions reports/phase5-open-images-baseline-v1/predictions.json \
 --out reports/phase5-open-images-baseline-v1/metrics.json
```

The default detector is the saved original OWLv2 with its pinned SHA-256, seven prompts, threshold .15 and NMS .5, identical to Phase 4. Future adapted models use the same command with `--checkpoint LOCAL_MODEL_DIRECTORY --sha256 EXPECTED_WEIGHT_SHA`, in a **different output directory**. The local directory must include compatible OWLv2 processor/config/tokenizer and safetensors. No hub fallback. Hashes of every local model/processor file are bound in `input-lock.json` and checked after inference.

On CUDA failure, fix visibility/memory availability and repeat the same inference command with `--resume`. It preserves completed immutable image stages and rejects input, code, processor, checkpoint, device or checksum changes. A stage interrupted during writing is preserved for audit; use a fresh output directory rather than deleting it. Error paths propagate, with no empty-detection substitution. No automatic CPU fallback or silent score threshold changes.

Metrics: confidence-ordered, same-class, one-to-one IoU .5 matching; precision/recall and mean matched IoU by category; duplicate false positives; small contained unmatched fragment candidates; counts of unknown-class and group/depiction-ignored predictions. Unknown class/image pairs are unscored. Verified negative detections count as false positives. Unknown/group/depiction boxes cannot create ordinary-instance true positives. This is a **flat partial-label pilot metric**, not hierarchy-expanded/group-credit official Open Images AP. An ordinary match gets priority over ignore regions. Zero denominators return null, never fabricated zero accuracy.

No scoring is performed on the current unannotated room gallery. No official Phase 2B images, feature arrays or decisions are loaded. A future frozen run should occur once per frozen model comparison; tuning uses training/development only.
