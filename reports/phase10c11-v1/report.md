# Phase 10C.11 — Frozen-calibration GPU ablation

The single authorized arm completed successfully. Freezing pretrained shift and scale reduced broad confidence inflation, but did not solve it or provide a reliable improvement in precision at comparable recall. The classification projection alone recreated substantial shared logit drift. This checkpoint remains experimental; no deployment or Phase 10D activation occurred.

## Execution and comparability

- Parent: `a8835ce934f2c96e388dae965ee5701ac46206ad`, new branch `experiment/phase10c11-frozen-calibration-ablation`.
- Original pretrained OWLv2 initialization, checkpoint SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.
- Exactly one arm, learning rate `1e-5`, seed `6106`, five epochs, **75 committed optimizer updates**, 295 training-image presentations. Same 59 optimization photographs, 56 known source groups, 182 positive boxes and 41 source-supported Chair-negative regions as C8 corrected Arm B. Known source groups are not proof of independent physical identities.
- All 75 update source-order lists exactly match C8 Arm B. The initial 24 optimization-probe and 19 development raw-output arrays were **bit-identical** to the historical original before the first update.
- Same original corrected focal loss, mask rules, positive annotations, AdamW settings, clipping, FP16 autocast/FP32 weights and loss, accumulation of four with final group of three. Only the four specified calibration tensors were removed from the optimizer. Eight tensors/1,577,988 scalars remained trainable: classification projection and box head. No backbone updates.
- Approved implementation/configuration: `authorized-config-v2.json`; original C10 disabled specifications and historical workers remain unchanged.
- Differences documented before training: standalone initialization repeated the existing 19-photo reference inference instead of reusing C8 Arm A's reference; detached Chair-specific diagnostics; comparison of initial saved outputs before optimization. No additional optimization objective was introduced. Tests verify the logging wrapper returns the original loss tensor, parts and input gradients unchanged, including FP16 inputs.
- The first launcher invocation failed on an import-path error **before creating the experiment directory, CUDA initialization, forwards or updates**. An isolated import fix and regression test preceded the one GPU execution. Both original preparation and version-2 execution records are preserved.

GPU worker runtime was **211.01 seconds**; enclosing queue runtime, including initialization and all evaluation inference, **213.65 seconds**. There were **467 total forward passes**: 295 training, 96 optimization-probe and 76 FP32 development forwards at epochs 0/1/3/5. CPU metrics and cached-feature arithmetic ran afterward in separate GPU-hidden processes.

Peak allocated VRAM: **2,024,382,976 bytes / 1.89 GiB**. Peak reserved: 2,327,838,720 bytes / 2.17 GiB. Peak sampled whole-device usage: 3,612,868,608 bytes / 3.36 GiB. All stayed below 6.5 GiB. No nonfinite, masking, integrity, runtime or memory failures occurred in the GPU arm.

## Unchanged evaluation protocol

Exactly the C8 19-photo, source-group-isolated, training-derived internal-development panel: 61 active GT instances (39 Chair, 13 Sofa, nine Bed). It is already exposed, partially annotated and **not independent validation**. Table, Desk, Cabinet and Lamp have no active supported GT and receive no accuracy/AP claims.

Identical original preprocessing, seven prompts/category order, FP32 inference, per-category NMS IoU 0.5 with cap 300, AP score floor 0.001, confidence 0.15, IoU 0.5/0.75 matching and 101-point macro AP over the three supported categories. CPU metric implementation SHA-256 remains `a6a1c8454efca6969620a06739ee0b72d50027a7986f43551917bb88a90e8460`.

Only four photographs support native Chair-FP scoring, one supports Sofa-FP scoring and five support Bed-FP scoring. Unmatched detections elsewhere remain unknown, not confirmed mistakes. Native ignored/non-exhaustive scopes remain unchanged. Project Chair includes seating/bar stools, while historical native taxonomies can separate Stool; this unresolved scope limitation was retained rather than relabelled to improve metrics.

Per-image raw predictions were saved atomically before CPU calculations. No official, reserved or frozen evaluation material was scored. Epoch five was the predeclared endpoint; milestones were diagnostic, not an opportunity to select a better epoch.

## Overall comparison

Percentages; confidence 0.15. FP means scored FP under the retained partial-label protocol.

| Model | Precision | Recall | AP50 | AP75 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original pretrained | 62.69 | 68.85 | 74.99 | 55.28 | 42 | 25 | 19 |
| C8 corrected Arm B | 6.86 | 95.08 | 75.25 | 55.21 | 58 | 788 | 3 |
| C11 frozen calibration | **8.76** | **90.16** | **76.76** | **60.58** | **55** | **573** | **6** |

Compared with B: 215 fewer scored FPs (27.28%), three fewer TPs, precision +1.90 percentage points and recall -4.92 points. Compared with original: precision -53.93 points, recall +21.31 points. This is not evidence that the independent 80%-precision/90%-recall objective has been reached.

Descriptive PR-envelope precision at a common recall target, applied consistently to all models:

| Recall target | Original | C8 B | C11 |
|---|---:|---:|---:|
| Original operating recall, 42/61 = 68.85% | 68.85% (42 TP/19 FP) | 80.77% (42/10) | **56.76% (42/32)** |
| B operating recall, 58/61 = 95.08% | Not attainable | 7.19% (58/749) | **7.28% (58/739)** |

The first row uses PR-envelope operating points, so the original's 68.85% differs from its fixed-0.15 precision of 62.69%. C11's corresponding descriptive thresholds are 0.72980392 and 0.09152289. These are analysis results, **not adopted thresholds**. At high matched recall the reduction is negligible; at original matched recall C11 ranks worse than B and original. No threshold or loss weight was tuned or changed.

## Per-category results

Each cell is precision / recall / AP50 / AP75 (%), followed by TP/FP/FN.

| Category | Original | C8 B | C11 |
|---|---|---|---|
| Chair | 55.81 / 61.54 / 55.46 / 31.75; 24/19/15 | 5.78 / 100.00 / 66.87 / 43.41; 39/636/0 | **6.72 / 89.74 / 61.96 / 39.95; 35/486/4** |
| Sofa | 91.67 / 84.62 / 90.10 / 69.31; 11/1/2 | 28.21 / 84.62 / 79.35 / 55.45; 11/28/2 | **31.58 / 92.31 / 88.37 / 74.64; 12/26/1** |
| Bed | 58.33 / 77.78 / 79.41 / 64.78; 7/5/2 | 6.06 / 88.89 / 79.52 / 66.79; 8/124/1 | **11.59 / 88.89 / 79.95 / 67.16; 8/61/1** |
| Table, Desk, Cabinet, Lamp | Unsupported GT | Unsupported GT | Unsupported GT |

Chair precision improves only 0.94 points over B, while recall loses four of 39 Chairs (10.26 points) and Chair AP50/AP75 regress. Sofa improves over B on all reported measures; versus original, Sofa precision and AP50 remain worse, but recall and AP75 improve. Bed preserves B's recall, halves its FP count and slightly improves AP; Bed precision remains far below original.

Mean matched IoU at 0.15 (a changing TP subset, not an independent localization experiment): Chair 0.81293 → 0.76964 → 0.77702; Sofa 0.86064 → 0.84011 → 0.86197; Bed 0.87862 → 0.87989 → 0.87926 (original → B → C11). Scored duplicate FPs: 1 → 4 → 2 overall. Fragmentation heuristic counts: 4 → 47 → 44; mixed-object heuristic counts: 15 → 79 → 61. These geometry heuristics do not establish the visual cause of individual errors.

Chair contributes **486/573 = 84.82%** of C11's scored FPs. Of these, 340 occupy less than 1% of the photograph, 105 occupy 1–5%, 30 occupy 5–20%, and 11 occupy at least 20%. Only one is an IoU-defined duplicate of a matched Chair. Broad small-box activations remain more prevalent than exact duplicate matches; missing/ambiguous stools and incomplete annotations still limit interpretation.

## Chair score thresholds and unknowns

NMS-retained Chair predictions, not all raw queries. Each cell is TP / scored FP / unknown unmatched.

| Threshold | Original | C8 B | C11 |
|---|---:|---:|---:|
| 0.15 | 24 / 19 / 14 | 39 / 636 / 1,455 | **35 / 486 / 849** |
| 0.50 | 3 / 0 / 0 | 34 / 211 / 252 | **27 / 117 / 113** |
| 0.80 | 0 / 0 / 0 | 28 / 39 / 36 | **24 / 13 / 14** |

At 0.80 C11 Chair precision is 64.86% and recall 61.54%, versus B 41.79%/71.79%. Higher thresholds suppress errors and Chairs together. Across all seven categories at 0.15, unknown unmatched predictions number 217 original, 2,434 B and 1,513 C11; these are excluded from scored FP totals. `results.json` additionally contains unknown counts at the 0.001 AP floor; they are explicitly different from this operating-point table.

## Confidence drift on identical model queries

All 68,400 queries are aligned by the same 19 photographs and 3,600 query indices. No selection by each model's detections is used in these distributions.

| Raw Chair score statistic | Original | C8 B | C11 |
|---|---:|---:|---:|
| Median | 0.0000136020 | 0.00391807 | **0.00117995** |
| 90th percentile | 0.00154509 | 0.103255 | **0.0535891** |
| 99th percentile | 0.0239067 | 0.563222 | **0.409501** |
| Queries ≥0.15 | 82 | 4,855 | **2,714** |
| Queries ≥0.50 | 3 | 926 | **440** |
| Queries ≥0.80 | 0 | 161 | **70** |

C11's median remains about 86.75 times original. The original's 24 GT-matched Chair query indices also rise from median score 0.33385 to 0.94941 in B and 0.92345 in C11. Both weak/background query activation and existing true-positive confidence increase.

| Mean logit increase vs original | C8 B | C11 |
|---|---:|---:|
| Sofa | 5.18727 | 4.20104 |
| Chair | 5.42252 | 4.24799 |
| Table | 3.57298 | 2.31894 |
| Bed | 5.23472 | 3.83622 |
| Desk | 3.88624 | 2.60620 |
| Cabinet | 3.17259 | 1.86536 |
| Lamp | 3.29428 | 2.00150 |
| All-category mean | **4.25294** | **3.01103** |
| Median common shift | 4.22281 | 2.98261 |
| Common-mode squared-drift fraction | 95.45% | 90.32% |

Mean shared inflation decreases **29.20%**, and Chair mean-logit inflation decreases **21.66%**. Nevertheless, **all seven logits increased on every one of the 68,400 queries** in both trained models. The common-mode fraction is a statistical decomposition, not a direct layer attribution.

Actual pre-head image/text activations cached during the authorized initial forwards permit a separate CPU-only functional decomposition. All eight projection/shift/scale head combinations and six-order Shapley averages were checked against saved GPU endpoints with predeclared `atol=1e-4, rtol=1e-5`; the largest candidate endpoint absolute error was 0.00010109 and passes the combined tolerance. No additional encoder/GPU forwards occurred.

| Mean endpoint logit contribution | B projection | B shift | B scale | C11 projection | C11 shift | C11 scale |
|---|---:|---:|---:|---:|---:|---:|
| Seven-category average | 2.90891 | 1.32741 | 0.01662 | **3.01103** | **0** | **0** |
| Chair | 4.07984 | 1.32741 | 0.01527 | **4.24799** | **0** | **0** |

These are endpoint functional attributions conditional on the frozen actual activations, **not causal training-dynamics percentages** or separately trained variants. The zero C11 shift/scale contributions follow their exact original values. The remaining projection produces all of its observed classification-logit drift. Its mean projection contribution is slightly larger than B's, consistent with partial compensation through the remaining trainable pathway; one seed does not establish a general compensatory mechanism.

Projection weight change L2: B 0.14520893 (1.49683% of pretrained norm), C11 **0.14573774 (1.50228%)**. Projection bias change L2: B 0.00584046, C11 **0.00617026**. Small weight-norm changes can substantially alter normalized image/text logits; norm size alone does not establish preservation of pretrained outputs.

## Training signal and frozen-parameter verification

Diagnostic unweighted mean per-image normalized losses (optimizer accumulation rules remained unchanged):

| Epoch | Classification | Weighted box | Chair positive | Chair negative | Added Chair terms |
|---|---:|---:|---:|---:|---:|
| 1 | 0.356271 | 0.462852 | 0.132534 | 0.0000001000 | 294 |
| 2 | 0.103157 | 0.462698 | 0.026034 | 0.0000046094 | 292 |
| 3 | 0.065873 | 0.449779 | 0.008700 | 0.0000139443 | 286 |
| 4 | 0.058745 | 0.426730 | 0.005562 | 0.0000355190 | 282 |
| 5 | 0.042739 | 0.423497 | 0.003641 | 0.0000970162 | 279 |

There were 1,433 additional Chair-negative term occurrences over training; median score 0.00142696, maximum 0.206574. Only three were ≥0.15 and none ≥0.50. At the final fixed probe, 281 terms covered 19 photographs/known groups and 24/41 evidence regions; 17 regions had no valid model-query coverage. Only two final probe terms were ≥0.15. **Zero protected-positive/mask violations** occurred. Most negative evidence remains easy, low-confidence patches rather than the high-confidence confusion responsible for the failure.

Both remaining classification tensors had finite recorded pre-clip gradients at every update; all four calibration gradients remained `None`. Full per-epoch norm quantiles and per-update records are retained. No calibration parameter entered the optimizer, including after train/eval transitions. All 76 snapshots (initialization and every update) have verified receipts, exact original calibration values, consistent frozen/backbone fingerprints and consistent optimizer cursors. Calibration tensor SHA-256 values:

| Parameter | SHA-256 |
|---|---|
| `class_head.logit_shift.weight` | `8ecf31509b44f2dc64e11d1f44b9cd30c9ce18daaace2a04baff6faeb2d4aa6a` |
| `class_head.logit_shift.bias` | `da4fdc6288b5f478c4261b6c93439a0fc3e9f8cb6b8700bcddd44688f1fb3de8` |
| `class_head.logit_scale.weight` | `985588774b8321bd5d419b7fa3e874bc75fac25818a86470f1ede992ba2b07d1` |
| `class_head.logit_scale.bias` | `69be7f155569d897139b6335e3b30357b4c3ebb18ddd60cc280c72b4dd82065c` |

The exact full FP32 values are retained privately in `data/open-images/preparation/phase10c11-v1/calibration-original-values.json` and every full-head snapshot; they are not uploaded as model weights. The public `calibration-verification.json` records all 76 snapshot hashes and unchanged calibration hashes.

## Decision and one next action

1. **Did freezing reduce broad inflation?** Yes, partially (mean +4.253 → +3.011 logits); it did not prevent universal seven-category increases.
2. **Did it improve Chair precision?** Slightly at 0.15 (5.78% → 6.72%), with 23.58% fewer scored Chair FPs; not a reliable overall precision-at-matched-recall gain.
3. **Did it preserve Chair recall?** No: 39/39 → 35/39 at 0.15.
4. **Did Sofa/Bed regress?** Against B, no material recall/AP regression: Sofa improves, Bed recall is unchanged with fewer FPs. Against original, both still have large fixed-threshold precision regressions; Sofa AP50 is lower.
5. **Did projection still inflate confidence?** Yes. Exact frozen calibration plus endpoint-validated cached-feature reconstruction identifies the changed projection as sufficient to recreate the remaining broad drift.

**Recommended next action:** separately authorize a controlled pretrained-projection-preservation experiment (for example, a fixed teacher/output-preservation constraint on optimization-training queries, retaining frozen calibration). Its design must preserve unknown regions rather than declaring them negative, retain actual hard-negative evidence, specify the preservation coefficient and acceptance criteria before running, and keep independent real-room evaluation untouched. This phase only investigates and recommends that direction; no new loss, variant or training was implemented or launched.

One seed, 59 training photographs, only three evaluable categories, few FP-complete photos, potential native Stool ambiguity, prior panel exposure and changing localized TP subsets prevent real-world accuracy or significance claims. An independent, sufficiently annotated seven-class real-room set remains necessary before claiming the target precision/recall.

## Artifacts, tests and preservation

- Experimental checkpoint: `data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt`.
- SHA-256: **`44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f`**.
- The prepared C10 destination name was retained deliberately; this directory contains the one C11 execution. No previous directory was overwritten.
- GPU raw predictions: same directory under `evaluation/epoch-{0,1,3,5}/`; immutable marker hashes in `execution-provenance.json`.
- CPU predictions/metrics and endpoint attribution: `cpu-diagnostics-v1/` under that experiment directory; saved separately from historical inference results.
- Public result tables/data: `results.json`, `confidence-attribution.json`, `training-curves.json`, `epoch-losses.csv`, `precision-recall.csv`, `diagnostics.svg`.
- Tests: 128 preflight CPU tests passed; 13 C11 post-run CPU tests passed (nine repeated instrumentation/import tests plus four new analysis tests; **132 distinct relevant tests**). No environment changes, downloads or additional GPU tests.
- Preservation checks passed for 9,973 historical data/model files, 160 Phase 10B snapshots, 24 Phase 10B report artifacts, 27 historical implementation files, 1,310 C8 sealed artifacts, 33 C10 public/10 C10 private artifacts, all 78 eligible-photo hashes and source-group isolation. Original checkpoint hash is unchanged. Source annotations, corrected masks, historical C8 checkpoints, SigLIP 2, historical datasets and frozen partitions were preserved. The Mac project was not modified.
- Existing `.venv`: Python 3.14.4, Torch 2.14.1+cu132/CUDA build 13.2. GPU: RTX 4070 Laptop. CPU metrics completed successfully with GPU hidden and the same metric implementation. No packages, drivers or environment were changed.
- Small code/configuration/reports only are committed and pushed on the C11 branch; weights, photograph bodies, raw queries and activation caches remain ignored/local. `main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

The queue has finished. The launcher rejects the existing destination; there is no automatic replay or next experiment. No deployment, extra training, threshold change or Phase 10D execution is authorized by this completion.
