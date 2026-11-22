# Phase 10D — Original-classifier-preserving box-only experiment

One authorized diagnostic arm. No model/data downloads, production replacement, new labels, negative supervision, reserved scoring or automatic follow-up experiment. Historical C21 configuration remains disabled and unchanged; this phase has its own explicit execution binding.

## Actual execution

Completed 5 epochs, 75 optimizer updates, 322 durable forward intents / 322 observed detector calls and 295 training backwards. GPU-process wall time including startup/smoke/final diagnostic: 142.976 seconds. Peak allocated VRAM: 2025887744 bytes (1.887 GiB); peak reserved: 2654994432 bytes.

FP32 throughout; batch 1, accumulation 4 (last group 3); fixed seed 6106 and unchanged five-epoch C8/C20 data ordering. AdamW LR 1e-5, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01, global norm clip 1. One fixed endpoint; no epoch/threshold/hyperparameter selection.

The 322 calls were 4 zero-update smoke, 295 training, 4 post-training FP32 invariance checks and 19 final internal-development calls. No additional original-baseline inference was performed; saved original/C20 outputs were replayed with identical metrics.

## Actual parameter and data scope

Original base `models/owlv2-original-baseline-20261103/model.safetensors`, SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`. All 418 persisted model tensors matched this base before GPU execution and again on GPU initialization. No adapted heads were loaded.

Exactly six trainable tensors, 1,184,260 scalars:

- `box_head.dense0.bias` — shape [768]
- `box_head.dense0.weight` — shape [768, 768]
- `box_head.dense1.bias` — shape [768]
- `box_head.dense1.weight` — shape [768, 768]
- `box_head.dense2.bias` — shape [4]
- `box_head.dense2.weight` — shape [4, 768]

All other 412 named parameters remain frozen, including classification projection, shift/scale, the full vision/text model, shared layer normalization and objectness. The complete exact frozen-name/shape/hash list and registered buffers are in `architecture.json`. OWLv2's derived `box_bias` is an **unregistered plain tensor**; the executor protects it with an additional original-computation/hash check.

Frozen parameter/buffer fingerprint: `dea9be2a7eb1f73e94ad200a09cf94a16fa73206de4f3a40bc6ddc4da6072126`. It was checked at initialization, all six snapshots and completion; each epoch has a saved fingerprint and per-box-tensor statistics/hashes. Exact box parameter values and optimizer state are in local checkpoints. All four repeated FP32 training-photo checks were bit-identical for classification logits.

Optimization used exactly 59 photos / 56 known source groups / 182 source-supported positive boxes: Chair 114, Sofa 40, Bed 28. Table, Desk, Cabinet and Lamp gained no supervised examples. Chair includes seating/bar stools; native annotations and unresolved seating remain unchanged. The five newly prepared Chair-negative regions, earlier source-negative regions, all ignored regions, new annotations and generic background labels were not activated.

The objective is the unchanged detached original-classification Hungarian assignment with 5×L1 + 2×GIoU on retained positives only. Classification/objectness/negative losses are zero. The frozen classifier can affect matching; localization can alter NMS and final retained detections despite unchanged raw logits.

## Matched internal-development comparison

Exact exposed 19-photo / 61-positive Sofa–Chair–Bed panel; confidence 0.15, IoU 0.50, NMS 0.50, score floor 0.001, cap 300, unchanged 101-point macro AP. Native annotation completeness controls FP scoring. Unknown unmatched predictions are not confirmed mistakes. This panel is not independent acceptance.

| Model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN | Precision at original recall |
|---|---:|---:|---:|---:|---:|---:|
| C20_E5 | 7.07% | 88.52% | 73.15% | 54.74% | 54 / 710 / 7 | 75.00% |
| Phase10D | 62.69% | 68.85% | 74.58% | 50.66% | 42 / 25 / 19 | 68.85% |
| original | 62.69% | 68.85% | 74.99% | 55.28% | 42 / 25 / 19 | 68.85% |

Matched-recall values are descriptive PR-envelope values, not newly adopted thresholds. Recomputed original and C20 baseline metrics exactly matched their saved historical results. Full PR curves are in `precision-recall.json`.

| Category / model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN | Unknown unmatched | Duplicate FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| chair / C20_E5 | 5.98% | 92.31% | 63.82% | 42.10% | 36 / 566 / 3 | 2477 | 2 |
| chair / Phase10D | 55.81% | 61.54% | 54.33% | 31.74% | 24 / 19 / 15 | 640 | 1 |
| chair / original | 55.81% | 61.54% | 55.46% | 31.75% | 24 / 19 / 15 | 634 | 1 |
| sofa / C20_E5 | 26.32% | 76.92% | 76.24% | 55.45% | 10 / 28 / 3 | 308 | 0 |
| sofa / Phase10D | 91.67% | 84.62% | 90.10% | 55.45% | 11 / 1 / 2 | 262 | 0 |
| sofa / original | 91.67% | 84.62% | 90.10% | 69.31% | 11 / 1 / 2 | 264 | 0 |
| bed / C20_E5 | 6.45% | 88.89% | 79.41% | 66.69% | 8 / 116 / 1 | 133 | 1 |
| bed / Phase10D | 58.33% | 77.78% | 79.30% | 64.78% | 7 / 5 / 2 | 59 | 0 |
| bed / original | 58.33% | 77.78% | 79.41% | 64.78% | 7 / 5 / 2 | 59 | 0 |

Table, Desk, Cabinet and Lamp remain unscored for positive-category performance; zero scored FPs in uncovered categories is not evidence of correctness. All seven native category fields, unknown counts and FP-score distributions are retained in `results.json` and `category-metrics.csv`.

## Localization and confidence attribution

| Category | Shared matched GT | Original mean IoU | Adapted mean IoU | Delta | IoU 0.75 up / down | Original-only / adapted-only matches |
|---|---:|---:|---:|---:|---:|---:|
| bed | 7 | 0.87862 | 0.87619 | -0.00243 | 0 / 0 | 0 / 0 |
| chair | 24 | 0.81293 | 0.80910 | -0.00383 | 0 / 0 | 0 / 0 |
| sofa | 11 | 0.86064 | 0.85318 | -0.00746 | 0 / 1 | 0 / 0 |

C20_E5: aligned 19-photo logits bit-identical to saved original = False; mean logit difference 4.11992464, maximum absolute difference 12.1332296. Per-category means and score/logit quantiles are in `results.json`.

Phase10D: aligned 19-photo logits bit-identical to saved original = True; mean logit difference 0, maximum absolute difference 0. Per-category means and score/logit quantiles are in `results.json`.

C20_E5: raw Chair query counts at 0.15 / 0.50 / 0.80: 4039 / 753 / 132; postprocessed heuristics: {'category_disagreement_candidates': 31, 'fragmentation_candidates': 47, 'mixed_object_candidates': 81}.

Phase10D: raw Chair query counts at 0.15 / 0.50 / 0.80: 82 / 3 / 0; postprocessed heuristics: {'category_disagreement_candidates': 14, 'fragmentation_candidates': 4, 'mixed_object_candidates': 16}.

original: raw Chair query counts at 0.15 / 0.50 / 0.80: 82 / 3 / 0; postprocessed heuristics: {'category_disagreement_candidates': 14, 'fragmentation_candidates': 4, 'mixed_object_candidates': 15}.

Four identical before/after training inputs establish numerical classification preservation independently of historical cache precision. If the aligned development logits also match original, retained detection/FP/FN differences are attributable to changed boxes and downstream matching/NMS, rather than trained calibration or classification projection. Fragmentation and mixed-object counts remain heuristics under partial annotation.

## Learning curve

| Epoch | Updates | Mean box loss | Mean L1 | Mean GIoU loss | Mean preclip box-gradient norm | Matched positives |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 0.418619 | 0.032769 | 0.127388 | 56.297200 | 182 |
| 2 | 30 | 0.397188 | 0.031130 | 0.120769 | 50.363401 | 182 |
| 3 | 45 | 0.384465 | 0.029988 | 0.117262 | 48.614228 | 182 |
| 4 | 60 | 0.381326 | 0.029602 | 0.116659 | 54.895876 | 182 |
| 5 | 75 | 0.367611 | 0.028357 | 0.112914 | 57.157264 | 182 |

Final inference forward mean: 0.2197s; median: 0.2172s. Synchronized forward-only timing excludes CPU decode/preprocessing and transfers. No new original-model timing was collected, so a matched baseline latency claim is unavailable.

## Decision and limits

Outcome **B**. Fixed-threshold 60% precision plus original 68.85% recall milestone reached: **True**. Stronger C21 localization/precision/recall gates all passed: **False**. Exact per-gate values are in `decision.json`; the 60% milestone alone is not the complete predeclared success criterion.

Outcome A: localization improved with classification preserved. B: classification preserved without demonstrated localization improvement. C: localization improved but precision/recall regressed. D would indicate insufficient execution/evaluation evidence. These are development diagnostics on an exposed, partially annotated three-category panel, not independent seven-class accuracy.

Recommended next direction: investigate frozen-original-teacher distillation as a separately prepared future experiment;do not start it or return to loss-coefficient sweeps without new evidence.

No fully independently annotated all-seven-class photographs currently qualify (C21 audit). The ultimate 80% precision / 90% recall target remains untested. No reserved/frozen images were used here.

## Checkpoint, preservation and recovery

Final local six-box checkpoint: `data/open-images/experiments/phase10c21-box-only-v1/checkpoints/update-075-277eac43.pt`; SHA-256 `2607165fd021718fed12cb610db6857efe3b0d5b6363dc60b99b3416e574f1ed`. This is a delta over the immutable original base, not a deployed full model or a twelve-head checkpoint.

`preservation.json` records historical image/model/annotation/seal checks, original and new checkpoint integrity, frozen fingerprints and unchanged main. `postrun-tests.json` records the relevant CPU suites. GPU receipts and raw predictions were saved atomically before the separate Torch-free CPU metric process. No interpreter/package change was made.

`resource-accounting.json` records launcher wall time, allocated/reserved VRAM and unique-inode/conservative storage. The 600-second launcher deadline includes initialization and diagnostics; the worker stops cooperatively earlier. Durable forward intents and optimizer intents distinguish attempted operations from committed results. No retry or resume was launched.

Rollback requires no model replacement: production and original weights never changed. Keep local epoch checkpoints, manifests and telemetry. Any future recovery needs separate authorization and must restore the six-tensor optimizer/cursor/RNG binding without replaying committed updates. Legacy twelve-head workers are incompatible. Code/configurations/reports are pushed on the dedicated experimental branch; weights, photos and raw caches stay local.
