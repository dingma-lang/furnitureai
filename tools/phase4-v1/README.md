# Phase 4 isolated room crop comparison

This experimental mode uses the existing OWLv2 detector, pinned SigLIP 2 image encoder and the Phase 3 matched baseline/expanded cosine heads. It does not change production routes, configured models, detector labels, bounding boxes, user confirmations, segmentation/depth, retrieval indexes or historical predictions.

Completed local run: `reports/phase4-room-comparison-v1/comparison.html`. It contains side-by-side full-room overlays and the exact shared crop gallery. All three authorized photographs completed on the RTX 4070: 25 detections/crops, four baseline/expanded disagreements, five expanded/OWLv2 label differences, zero inference errors. These are behavioral counts, **not accuracy**. Assistant-provisional boxes in the source manifest were not used for scoring.

## Run / recover

From `/home/tristary/furniture-ai-ml`:

```bash
# Existing completed report: validates inputs/preservation and skips all inference.
.venv/bin/python -B tools/phase4-v1/compare_rooms.py --resume

# A genuinely new version, using the existing authorized three-photo manifest:
.venv/bin/python -B tools/phase4-v1/compare_rooms.py \
  --output reports/phase4-room-comparison-v2

# Recover that same run after diagnosing an inference failure:
.venv/bin/python -B tools/phase4-v1/compare_rooms.py \
  --output reports/phase4-room-comparison-v2 --resume
```

Use `--manifest <local-room-manifest.json>` for 1–3 existing authorized room photographs. Follow the schema of `data/detection-pilot/rooms.json`: image IDs/files, actual SHA-256, `rights.allowed_for_benchmark=true`, and honest licence/source records. Do not invent authorization. Paths under `data/abo/` and hashes from official/frozen evaluation sources are forbidden. The tool does not fetch a URL. For a larger room study, create a separately reviewed bounded protocol rather than removing the cap.

The runner refuses an existing output without `--resume`; resumed code/settings/source fingerprints must match. Atomic successful detector stages, crop features and per-room results are reused. Inference errors are explicit by stage, with detections retained when embedding/classification fails. OOM reduces only embedding minibatch size or retries the same detector operation at most twice; it does not alter boxes, resolution, thresholds or use a CPU fallback. Generic model/kernel failures propagate into visible partial/error records; no fake detections/features are substituted. Completed runs skip all model inference when resumed. Do not edit completed artifacts to force a retry; use a fresh version if inputs/code change.

## Checkpoint contract

Expanded head:
`data/abo/experiments/phase3-gpu-v1/final-selected/best.pt`

SHA-256:
`51097ebf696a7765d023c8c02a9e4ea14965e0da04877cdd65be4416b9c13f0d`

Matched previous cosine recipe refit:
`data/abo/experiments/phase3-gpu-v1/final-baseline/best.pt`

SHA-256:
`2c61fe77229a67b1cdd2dc4cd3109fb0a9f3675ff5348519fddd8385dd9b6755`

These are the two heads behind the 79.13%→88.96% Phase 3 controlled panel comparison. The baseline is identified explicitly; this does not imply the application previously deployed that head. Both contain `state_dict.weight` shaped `[7,768]`, `config.kind=cosine`, a class tuple and `feature_model_fingerprint`. Load with `torch.load(..., weights_only=True, map_location='cpu')` after SHA verification. Reject incorrect class order, architecture, dimensions, nonfinite weights or embedding fingerprint. No checkpoint is written by this mode.

Pinned encoder:
`models/siglip2-base-patch16-224-phase2a-v1/`, original revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, model/processor fingerprint `b5d0736ddf111580ada6d0bd9d207c423a2324d250962ddbbb6d7095229dee91`.

Preprocessing: EXIF-transposed RGB; integer crop bounds from the existing `crop_bounds` helper; the saved slow SiglipImageProcessor (`use_fast=False`) resizes directly to 224×224 with bilinear sampling, rescales 1/255 and uses RGB mean/std 0.5. `AutoModel.get_image_features` produces 768-dimensional float32 vectors; NumPy L2 normalization matches training. Both heads receive the **same** vector matrix. Each crop records PNG SHA-256, preprocessed pixel-tensor SHA-256 and shared embedding SHA-256. No separate classifier-specific crop preprocessing occurs.

Class indices: `0 sofa, 1 chair, 2 table, 3 bed, 4 desk, 5 cabinet, 6 lamp`.

Cosine-head logits: `10 * normalized_image_features @ normalized_head_weights.T`. Results expose prototype cosine, logits and seven-way softmax. `s7` is **uncalibrated closed-set relative score**, not correctness probability, product identity similarity or human-verified accuracy. There is no learned unknown class. A non-furniture/mixed/partial crop can still receive a furniture label; no abstention threshold was tuned on these three photos.

## Detection and memory policy

The existing `furniture_bench.detection.owlv2.Owlv2Detector` runs from the immutable original local checkpoint, SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`. This run freezes the historical detector baseline configuration: seven default `a photo of a <category>` prompts, threshold 0.15, class-aware NMS IoU 0.5, native checkpoint 960×960 processing. It does not retune thresholds or reproduce the UI's separate custom prompt settings.

OWLv2 runs first on all rooms, then unloads before SigLIP loads. The GPU allocator is capped at 75%. Embedding batch starts at four and halves on OOM to one. At most 64 crops/room are classified; any excess remains explicitly budget-skipped with its original detection record. This run classified all 25 boxes. Tiny cosine heads run on CPU against the shared GPU-extracted vectors; this is not a fallback for detector/encoder inference. Maximum observed PyTorch allocated VRAM was 2,185,781,248 bytes; total end-to-end inference/artifact time was about 11.56 seconds (initial integrity snapshot excluded).

Detector boxes remain their exact postprocessed xyxy values. Integer floor/ceil crop extents are stored separately, and only visualization coordinates are scaled. Classification never adds, removes, moves, suppresses or relabels original detection records. Raw candidates and the existing filter audit are preserved separately. Duplicate proposals, localization errors and missed objects remain detector issues; a changed crop category cannot fix them.

## App integration audit

- `furniture_bench/pipeline.py` already creates crops after OWLv2 and optionally calls SigLIP/FAISS retrieval; optional SAM 3.1 masks are associated with detections by IoU. Failure of that optional mask branch does not block crop retrieval.
- `room_viewer/photo_worker.py` invokes the existing detector worker; `photo_jobs.py` queues detection, ADE20K UPerNet semantic segmentation and Depth Anything V2 Small relative-depth stages. Semantic room labels are not furniture instance masks, and relative depth is not metres.
- `room_viewer/product_worker.py` uses SigLIP/FAISS for similar-product retrieval, not a trained seven-way head. `product_matching.py` uses a reviewed detection/category to filter catalog candidates and expressly does not verify exact product identity.
- No deployed trained-head loading hook was found in these paths. Existing classification is primarily detector text-query labels and reviewed labels. The new head can be added later as advisory crop-classification fields after detection/cropping, separate from detector/user-reviewed categories, before an optional retrieval decision. Do not silently feed its uncertain result into catalogue filters or scene geometry as ground truth.
- UI code expects `.cache/models/...`, example prompts and catalogue/index assets that are not present in this WSL snapshot. This experiment uses the available pinned `models/...` paths explicitly. It did not launch or change a production application. Segmentation/depth/SAM weights were not downloaded or invoked.

## Outputs and interpretation

`comparison.html` and three `side-by-side.png` files are the visual entry points. `results.json` contains every box, crop, score vector, disagreement, error and provenance. Per-room `detector.json` retains original/raw candidates; `crop-features.npy/json` binds identical preprocessing and feature reuse; original source photographs remain unchanged. `preservation-after.json` verifies historical results/models/source. No official Phase 2B image was forwarded through a model; official image/feature paths receive stat-only preservation checks.

The three photographs are an exploratory convenience sample with existing rights/attribution. Their old assistant-provisional annotations are not independent ground truth. The report separates detector/proposal/crop ambiguity from classifier behavior and includes a blank optional crop-review sheet; it never claims detection/classification accuracy or automatically fills human labels.

Open Images proposal: `reports/phase4-room-comparison-v1/open-images-pilot.md`. It proposes at most 150 photographs, a metadata/rights audit and a frozen 30/120 image-grouped protocol; exact MID availability/counts and remote sizes remain pending an approved metadata audit. No external dataset files or images were acquired. OWLv2 fine-tuning is conditional on a proper supervised detection loss/trainer and separate annotations/approval; the installed detection forward API has no `labels` loss interface.

Tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase4-v1 -p 'test*.py' -v
```

Thirteen tests passed, including actual-head formula parity against the training interface, SHA/category/embedding-space checks, identical shared preprocessing, score comparison, immutable boxes/checkpoints/images, no historical overwrite, no official holdout inputs, bounded OOM recovery and propagation of other inference errors. The real GPU run completed without errors; completed-run resume was tested without repeating inference. No dependencies were installed and no model was trained.
