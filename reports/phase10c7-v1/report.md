# Phase10C.7 — actual-model-query negative-supervision smoke

**Decision A: technically ready for a separately authorized controlled diagnostic training comparison, with a weak-initial-signal caveat.** No training is authorized by this result. Phase10D remains unauthorized.

Exactly24 original-OWLv2 GPU forwards were executed on the exact24 approved optimization-training photographs/24 source groups containing41 source-supported regions. No warmup/retry/model-comparison forwards occurred. No backward passes or optimizer steps occurred, including in CPU tests. No checkpoint or production defaults were changed. No downloads or validation/reserved/frozen-set scoring occurred.

## Actual coverage

| Measurement | Actual result |
|---|---:|
| Raw model queries | 86,400 (3,600 per photo) |
| Added valid Chair-negative query terms | 292 |
| Photos/source groups with added terms | 19/24 |
| Source regions with nonzero coverage | 24/41 |
| Source regions without coverage | 17/41 |
| Positive/protected overlap incorrectly targeted | 0 |
| Coordinate-membership discrepancies | 0 |
| Changed existing positive targets/legacy masks/non-Chair terms | 0 |

Every added query has exactly one source-region owner. The41 CPU interior probes/352 synthetic grid terms from C.6 were not reused as model predictions. The actual result is292 terms, not352.

All17 zero-coverage regions lack any predicted box fully contained in the verified region. No source rectangle was enlarged, no protected region weakened and no box clipped to force coverage. Zero-coverage IDs: `C015`, `C025`, `C036`, `C077`, `C081`, `C083`, `C103`, `C142`, `C143`, `C144`, `C145`, `C157`, `C189`, `C202`, `C203`, `C204`, `C207`.

Detailed query indices, original pixel/normalized coordinates, padded-square boxes, source-region ownership, Chair logits/scores, matched-positive labels and per-region zero coverage are saved privately per photo. This distinguishes prepared evidence from actual model-query support without fabricating coverage.

## Matching, masks and preprocessing

The original checkpoint is `models/owlv2-original-baseline-20261103/model.safetensors`, SHA-256:

`e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.

The pinned local processor pads bottom/right to max(originalH,originalW), then resizes to960×960 using its saved slow implementation. Inverse mapping uses the same maximum dimension with zero top/left offset. Unclipped predicted boxes are retained. Both padded-square and independent original-pixel containment checks agree. All7 prompts are exactly `a photo of a <category>` in the preserved Sofa/Chair/Table/Bed/Desk/Cabinet/Lamp order.

Weights stayedFP32, with CUDAFP16 autocast matching Phase10B optimization. The entire original model was frozen and in evaluation mode. No head delta or trained checkpoint was loaded. Hungarian matching uses the unchanged Phase10B cost, native GT and label mapping. Legacy and corrected masks are calculated independently on the same raw outputs and matching assignments. Independent replay of the original loss matches the historical implementation.

Additional Chair negatives require full-box containment inside a supported region and exclusion from all known Chair/stool/ignored and newly uncertain scopes. Positive matches always win. Only the Chair column is extended; no other furniture is relabeled as background. Existing Sofa/Bed positives and all incomplete-label masks are preserved. Known source/creator groups remain isolated; all photo/evidence/model/processor hashes passed before execution.

## Learning-signal magnitude

Mean per-image legacy classification loss: **0.8692358943**.

Mean per-image corrected classification loss: **0.8692360061**.

The independently computedFP64 mean added focal contribution is **1.1471792e-7** per photo (sum2.7532300e-6 across24photos). FullFP32 reductions often round away this tiny increment; the slight difference from subtracting aggregatedFP32 losses is numerical reduction rounding, not a changed denominator. Both objectives use the same positive-GT normalization, focal alpha0.25/gamma2 and classification1/L1=5/GIoU2 coefficients.

Added-query Chair scores: median**0.000157964**,95th percentile**0.00165254**,maximum**0.0190509**. None reaches0.05 or0.15; one reaches0.01. These are presently *easy negatives*, not established high-confidence false positives. Raw queries were not thresholded or NMS-filtered for supervision.

The objective genuinely differs on real predictions across19groups, clearing the zero-contrast blocker. However, a positive difference does not establish a useful gradient magnitude, prevention of score inflation or improved detection. No backward pass was allowed; actual gradient contribution/AMP numerical behavior is unmeasured. As training changes classification scores and box sizes, source-region coverage and negative loss could increase, shrink or disappear. A future diagnostic must log those quantities rather than assume stability.

## Per-photo comparison

Indices correspond to the exact frozen24-photo manifest; private identities and creator/source records stay local.

| Photo index | New Chair terms | Covered regions | Legacy class loss | Corrected class loss |
|---|---:|---:|---:|---:|
| 1 | 2 | 1/1 | 0.983980596 | 0.983980596 |
| 2 | 1 | 1/1 | 0.087107494 | 0.087107494 |
| 3 | 7 | 1/2 | 0.997012436 | 0.997012436 |
| 4 | 2 | 1/3 | 0.193491951 | 0.193491951 |
| 5 | 27 | 2/2 | 0.414622962 | 0.414622962 |
| 6 | 12 | 2/3 | 0.583403051 | 0.583403111 |
| 7 | 8 | 1/2 | 0.163686380 | 0.163686380 |
| 8 | 0 | 0/1 | 1.654608727 | 1.654608727 |
| 9 | 49 | 2/2 | 0.961190879 | 0.961190879 |
| 10 | 0 | 0/1 | 0.279350787 | 0.279350787 |
| 11 | 34 | 1/1 | 4.179687023 | 4.179687023 |
| 12 | 15 | 1/1 | 1.211456537 | 1.211456537 |
| 13 | 0 | 0/1 | 0.297348231 | 0.297348231 |
| 14 | 0 | 0/2 | 0.217407748 | 0.217407748 |
| 15 | 1 | 1/2 | 1.562079668 | 1.562079668 |
| 16 | 29 | 1/1 | 1.425301433 | 1.425304055 |
| 17 | 64 | 1/2 | 0.694871485 | 0.694871485 |
| 18 | 9 | 2/3 | 0.370470077 | 0.370470077 |
| 19 | 0 | 0/2 | 0.382710785 | 0.382710785 |
| 20 | 2 | 1/2 | 0.245536745 | 0.245536745 |
| 21 | 1 | 1/1 | 0.840927839 | 0.840927839 |
| 22 | 11 | 2/3 | 0.845989704 | 0.845989704 |
| 23 | 4 | 1/1 | 0.228349179 | 0.228349179 |
| 24 | 14 | 1/1 | 2.041069746 | 2.041069746 |

## Resources and preservation

Exactly**24/24 forwards**,**15.67seconds** worker time including setup/state verification (below600seconds), peak Torch allocated**1.55GiB**, reserved**2.17GiB**, sampled whole-device used**3.33GiB**. The limit was6.5GiB. Device sampling ran every50ms; Torch peak counters are separate measurements. A parent hard deadline, forward reservations, allocator headroom and checks enforced the bounds. GPU memory returned to0MiB after worker exit.


All raw arrays were saved atomically with SHA-256 receipts before CPU matching/loss work. CPU analysis ran in a separate CUDA-hidden process in the unchanged virtual environment and completed cleanly. No interpreter/package/environment installation was needed. Full original model-state fingerprints before and after are identical, and all parameters remain frozen with no gradients. Original on-disk checkpoint and processor hashes are unchanged.

19 forward-free CPU tests passed: exact pad/inverse mapping, unclipped boxes, atomic cache/no-overwrite, finite/dtype/schema checks, Hungarian/loss equivalence, category masking, protected positives and hard-budget/no-training code checks. Read-only historical verification passed9,973 dataset/model files,160 Phase10B snapshots,24 report artifacts,27 prior implementation files,78 eligible photo hashes and prior C.2–C.6 seals. Previous experiments/source annotations/partitions and the Mac application remain unchanged.

Private outputs: `data/open-images/experiments/phase10c7-query-smoke-v1/`. `binding.json` pins authorization/data/model/code/environment;24 `forward-*.intent.json`,24queryNPZs and receipts prove the consumed budget. `raw-complete.json` records resources and identical state fingerprints. `cpu-analysis-v1/` contains query/mask attribution, region coverage and loss details. Small aggregate reports are under `reports/phase10c7-v1/`.

## Recommendation and limitations

A controlled diagnostic A/B is technically justified because real source-backed supervision differs safely across19groups, not just synthetic boxes. Any training still needs separate authorization. Keep initialization, source data/groups, prompts, seed, matching, normalization, updates and all coefficients identical; change only the negative mask. Use the previously bounded two-arm proposal only after approval. Monitor actual region coverage, Chair confidence inflation and negative-loss contribution through training; stop on vanishing contrast, nonfinite behavior, frozen-parameter drift or resource-limit violations. Do not quietly increase negative weights or expand regions to obtain an effect.

The original model currently assigns very low Chair scores to these interiors, so an immediate large improvement is not supported by this smoke. Source masks and automatic visual checks remain fallible and do not establish human-reviewed Chair absence. Missing annotations and incomplete native taxonomy coverage remain limitations. A single original-model FP16 eval-mode pass does not establish future-training coverage or gradient behavior.

Independent real-room evaluation remains separate. This phase reports no precision/recall/AP or80%/90% achievement and performs no model selection. Preserve all existing frozen material and exposed-panel limitations. Do not start another GPU run: the approved24-forward budget has been consumed.
