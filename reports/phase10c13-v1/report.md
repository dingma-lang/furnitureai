# Phase 10C.13 — Projection preservation GPU ablation

**Outcome B: over-regularized, provisionally under the unchanged internal protocol.** The one authorized arm completed. L2-SP substantially reduced parameter drift and partially reduced shared confidence inflation, but did not improve precision at matched recall. Five additional Chairs were missed and Sofa localization/AP75 regressed. This is not a production candidate or evidence that the independent 80%-precision/90%-recall goal has been met.

## Execution and reproducibility

- Parent: `423dfd0de60b685dccff271c97fe89fa53549412`; new branch: `experiment/phase10c13-projection-preservation-ablation`.
- Exactly one arm, original pretrained OWLv2 initialization, LR **1e-5**, seed **6106**, **five epochs / 75 committed optimizer updates**. No retries or alternative hyperparameters.
- Same 59 optimization photographs, 56 known source groups, 182 positive boxes and 41 source-supported negative regions as C11. These groups do not establish independent physical-scene identities. All 75 source-order lists match C11. Positive annotations, corrected negative supervision, ignored regions, matching, focal coefficients, box loss, optimizer, accumulation, preprocessing and postprocessing are unchanged.
- Exactly the original eight trainable tensors: classification projection weight/bias and six box-head tensors, 1,577,988 scalars. Four calibration tensors and all non-head parameters remain frozen. No backbone updates or Phase 10D activation.
- The sealed C12 `launch12.py`, `future_worker12.py`, `objective12.py`, `preservation12.py` and checkpoint implementation were executed **without modification**. The only intended optimization difference from C11 is the implemented L2-SP term.
- C12 stores lambda `78.92715855598048`; the authorization specifies `78.92715856`. Both become exactly **78.92716217041016 in FP32**. CPU tests established identical loss and gradient arithmetic. The prepared value was preserved and the decimal representation difference documented before execution; no coefficient tuning occurred.
- Initial predictions on the 24 optimization probes and 19 internal-development photos were **bit-identical to C11**, as were all head weights after the first update, when the regularizer was zero. This directly checks comparability before the trajectories diverge.

Worker runtime: **220.92 seconds**. Enclosing queue runtime including initialization and evaluation inference: **224.59 seconds**, below 600. Peak allocated VRAM: **2,026,482,176 bytes / 1.8873 GiB**; peak reserved: 2,325,741,568 bytes / 2.1660 GiB; peak sampled whole-device usage: 3,610,771,456 bytes / 3.3628 GiB. All remained below 6.5 GiB.

There were **467 forward passes**: 295 training-photo presentations, 96 optimization-probe forwards and 76 FP32 internal-development forwards. Epoch-2/4 summaries were computed from existing update records and snapshots, without additional forwards. All GPU work ended before the separate CPU metric/analysis processes. No nonfinite, mask, resource or checkpoint-integrity violation occurred.

## Preserved objective and frozen references

```text
L_SP = 0.5 × (sum((class_head.dense0.weight − original_weight)²)
              + sum((class_head.dense0.bias − original_bias)²))
L_total = L_existing_corrected + 78.92715856 × L_SP
```

This is a joint half-squared **sum**, not MSE. Immutable original FP32 references stay outside the optimizer. The penalty is applied once per optimizer group through the unchanged 4-microbatch averaging (last group of three). It is not divided by positive-box count. There is no generic background class or change to the seven categories.

All **76 snapshots** were checksum-verified. Each preserves the same calibration values, original projection-reference hashes, objective/configuration binding, eight-key optimizer group, update cursor and frozen/backbone hashes. Calibration gradients remained `None`, and reference corruption checks ran throughout the worker. The four calibration hashes are retained in `integrity.json`:

| Frozen parameter | SHA-256 |
|---|---|
| `class_head.logit_shift.weight` | `8ecf31509b44f2dc64e11d1f44b9cd30c9ce18daaace2a04baff6faeb2d4aa6a` |
| `class_head.logit_shift.bias` | `da4fdc6288b5f478c4261b6c93439a0fc3e9f8cb6b8700bcddd44688f1fb3de8` |
| `class_head.logit_scale.weight` | `985588774b8321bd5d419b7fa3e874bc75fac25818a86470f1ede992ba2b07d1` |
| `class_head.logit_scale.bias` | `69be7f155569d897139b6335e3b30357b4c3ebb18ddd60cc280c72b4dd82065c` |

## Identical internal-development evaluation

The exact same **19 already-exposed, training-derived internal-development photographs** were used, with 61 active GT objects: 39 Chair, 13 Sofa and nine Bed. Historical original/C8B/C11 scoring manifests match current row identities, source hashes, FP32 precision and row digest. The new initialization reproduces original metrics exactly.

Unchanged settings: confidence 0.15; category-wise NMS IoU 0.5/cap 300; AP floor 0.001; matching IoU 0.5/0.75; 101-point macro AP over the three supported classes; original prompts, box transforms and category order. Table, Desk, Cabinet and Lamp have no supported active GT and receive **no invented accuracy/AP**.

Only four photos support native Chair-FP scoring, one supports Sofa-FP scoring and five support Bed-FP scoring. Unmatched predictions under incomplete annotation coverage remain unknown, not scored mistakes. Native Chair-versus-Stool limitations are unchanged: project Chair includes seating/bar stools, while footstools/ottomans/step stools remain separate. Original annotations were not relabelled to change apparent performance.

Per-image GPU raw arrays were saved atomically before CPU metrics. Postprocessed predictions were also saved before CPU metric calculations. No official, reserved or frozen independent evaluation material was scored. Epoch five was the predeclared endpoint; no alternate epoch or threshold was selected after inspecting results.

## Four-model comparison

All percentages below use the unchanged partial-label protocol. FP means **scored FP**, not every unmatched prediction.

| Model | Precision @0.15 | Recall @0.15 | AP50 | AP75 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original OWLv2 | 62.69 | 68.85 | 74.99 | 55.28 | 42 | 25 | 19 |
| C8 corrected Arm B | 6.86 | 95.08 | 75.25 | 55.21 | 58 | 788 | 3 |
| C11 frozen calibration | 8.76 | 90.16 | 76.76 | 60.58 | 55 | 573 | 6 |
| **C13 projection regularization** | **14.45** | **81.97** | **75.27** | **52.95** | **50** | **296** | **11** |

Versus C11: precision +5.69 percentage points, recall **−8.20 points**, 277 fewer scored FPs (48.34%), five fewer TPs, AP50 −1.50 points and AP75 **−7.63 points**. Versus original, fixed-threshold precision remains much worse while recall is higher. F1 rises because the original adapted false-positive burden was very large; this does not establish a better precision/recall frontier.

### Precision at matched recall

Consistent PR-envelope comparison at fixed recall targets; thresholds are descriptive only and were **not adopted**.

| Recall target | Original precision | C8 B | C11 | C13 |
|---|---:|---:|---:|---:|
| Original: 42/61 = 68.85% | 68.85% | 80.77% | 56.76% | **56.00%** |
| C13: 50/61 = 81.97% | 15.20% | 23.36% | 18.45% | **15.58%** |
| C11: 55/61 = 90.16% | Not attainable | 15.71% | 11.60% | **Not attainable** |
| C8 B: 58/61 = 95.08% | Not attainable | 7.19% | 7.28% | **Not attainable** |

C13's maximum observed recall under the frozen argmax-label/NMS/cap/0.001-floor protocol is **52/61 = 85.25%**. This is not an intrinsic maximum over all raw queries or hypothetical postprocessing. At the original recall, C13 has 42 TP/33 FP versus C11 42/32. At the C13 recall target, its best envelope point has 50 TP/271 FP versus C11 50/221 and C8 B 50/164. Thus the fixed-0.15 precision gain is mainly reduced activation, not improved useful ranking at comparable recall. Curves can cross: C13 is slightly above original at its 81.97% target, but worse at the original recall and below both adapted baselines at that higher target.

### Per-category comparison

Cells contain precision / recall / AP50 / AP75 (%), followed by TP/FP/FN.

| Category | Original | C8 B | C11 | C13 |
|---|---|---|---|---|
| Chair | 55.81 / 61.54 / 55.46 / 31.75; 24/19/15 | 5.78 / 100.00 / 66.87 / 43.41; 39/636/0 | 6.72 / 89.74 / 61.96 / 39.95; 35/486/4 | **10.83 / 76.92 / 56.25 / 36.19; 30/247/9** |
| Sofa | 91.67 / 84.62 / 90.10 / 69.31; 11/1/2 | 28.21 / 84.62 / 79.35 / 55.45; 11/28/2 | 31.58 / 92.31 / 88.37 / 74.64; 12/26/1 | **44.44 / 92.31 / 88.53 / 54.57; 12/15/1** |
| Bed | 58.33 / 77.78 / 79.41 / 64.78; 7/5/2 | 6.06 / 88.89 / 79.52 / 66.79; 8/124/1 | 11.59 / 88.89 / 79.95 / 67.16; 8/61/1 | **19.05 / 88.89 / 81.02 / 68.10; 8/34/1** |
| Table, Desk, Cabinet, Lamp | Unsupported GT | Unsupported GT | Unsupported GT | Unsupported GT |

Chair loses five previously detected GT objects and gains none at 0.15: recall falls **12.82 points**; Chair AP50 drops 5.71 points and AP75 drops 3.75 points versus C11. Chair contributes 247/296 = **83.45%** of remaining scored FPs, including 167 smaller than 1% of the photograph. Bed retains its recall and improves AP slightly. Sofa retains the same 12 GT detections but has a material localization regression: on those identical GT identities, mean IoU falls **0.86197 → 0.82748**, with **two matches crossing below IoU 0.75** and none crossing upward. Sofa AP75 falls **20.07 points**.

On the 30 Chair GT objects detected by both models, mean IoU is essentially unchanged (0.80409 → 0.80430). Its apparent overall mean-IoU rise mostly reflects losing five other detections; it is not a localization success. Bed common-GT mean IoU is essentially unchanged (0.87926 → 0.87905). The box-loss implementation was preserved, but changed classification gradients affect shared global clipping and subsequent discrete matching; no additional box objective is needed for box updates to diverge.

### Chair thresholds and incomplete coverage

Each cell is TP / scored FP / FN / unknown unmatched, after unchanged NMS.

| Threshold | Original | C8 B | C11 | C13 |
|---|---:|---:|---:|---:|
| 0.15 | 24 / 19 / 15 / 14 | 39 / 636 / 0 / 1,455 | 35 / 486 / 4 / 849 | **30 / 247 / 9 / 279** |
| 0.50 | 3 / 0 / 36 / 0 | 34 / 211 / 5 / 252 | 27 / 117 / 12 / 113 | **24 / 32 / 15 / 29** |
| 0.80 | 0 / 0 / 39 / 0 | 28 / 39 / 11 / 36 | 24 / 13 / 15 / 14 | **19 / 3 / 20 / 3** |

C13 Chair precision/recall at those thresholds are **10.83%/76.92%**, **42.86%/61.54%** and **86.36%/48.72%**. The last precision does not imply the acceptance goal was reached: recall is below half. No threshold changed in the model or application.

Across all seven categories at 0.15, unknown unmatched predictions number **217 original, 2,434 C8 B, 1,513 C11 and 850 C13**. They are separate from the scored FP totals. The saved category `unknown_unmatched` fields within primary metric objects use the 0.001 AP floor; operating-point unknowns in `thresholds` use their named threshold. These counts must not be conflated.

## Projection and confidence drift

| Projection displacement from pretrained | C8 B | C11 | C13 |
|---|---:|---:|---:|
| Weight L2 | 0.14520893 | 0.14573774 | **0.02251288** |
| Bias L2 | 0.00584046 | 0.00617026 | **0.00241840** |
| Joint L2 | 0.14532634 | 0.14586830 | **0.02264240** |
| Weight displacement / pretrained norm | 1.49683% | 1.50228% | **0.23207%** |
| Bias displacement / pretrained norm | 5.80923% | 6.13726% | **2.40547%** |

Joint displacement is **84.48% lower than C11**. This proves that the prior controlled parameter distance, not that it preserved useful detection behavior.

All raw confidence distributions align the same **68,400 photo/query indices**, without selecting each model's detections.

| Raw Chair probability statistic | Original | C8 B | C11 | C13 |
|---|---:|---:|---:|---:|
| Median | 0.0000136020 | 0.00391807 | 0.00117995 | **0.000213820** |
| 90th percentile | 0.00154509 | 0.103255 | 0.0535891 | **0.0188204** |
| 99th percentile | 0.0239067 | 0.563222 | 0.409501 | **0.208788** |
| Queries ≥0.15 | 82 | 4,855 | 2,714 | **1,049** |
| Queries ≥0.50 | 3 | 926 | 440 | **141** |
| Queries ≥0.80 | 0 | 161 | 70 | **26** |

The C13 median is still about **15.72 times original**. Shared confidence inflation is reduced, not removed.

| Mean logit increase vs original | C8 B | C11 | C13 |
|---|---:|---:|---:|
| Sofa | 5.18727 | 4.20104 | **2.51426** |
| Chair | 5.42252 | 4.24799 | **2.66127** |
| Table | 3.57298 | 2.31894 | **1.44033** |
| Bed | 5.23472 | 3.83622 | **2.07133** |
| Desk | 3.88624 | 2.60620 | **1.59227** |
| Cabinet | 3.17259 | 1.86536 | **1.19455** |
| Lamp | 3.29428 | 2.00150 | **1.27545** |
| Mean across seven | 4.25294 | 3.01103 | **1.82135** |
| Median common shift | 4.22281 | 2.98261 | **1.81036** |
| Common-mode squared-drift fraction | 95.45% | 90.32% | **91.58%** |

Mean shared inflation falls **39.51% versus C11**, much less than the 84.48% parameter-distance reduction. **All seven logits still increase at every one of the 68,400 queries.** A small movement in a shared, normalized projection can remain functionally consequential. This statistical common-mode decomposition does not assign independent weight/bias causal percentages; both changed, while calibration and the non-head feature extractor were fixed.

## Actual loss and gradient monitoring

Epoch zero is the fixed 24-photo optimization probe before updates. Epochs 1–5 are means over the 59 training-photo presentations within each epoch; endpoint displacement/penalty are measured after its final update. These are different sampling/time conventions and are labelled accordingly.

| Epoch / updates | Classification | Positive classification | Legacy negative | Added Chair negative | Box loss | Mean weighted L2-SP | Endpoint joint drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 / 0 | 0.869236 | 0.869236 | 0 | 0.000000115 | 0.443301 | 0 | 0 |
| 1 / 15 | 0.405262 | 0.405005 | 0.000256 | 0.000000096 | 0.449457 | 0.019298 | 0.029459 |
| 2 / 30 | 0.232327 | 0.229552 | 0.002773 | 0.000001430 | 0.441355 | 0.029238 | 0.025215 |
| 3 / 45 | 0.152768 | 0.147383 | 0.005382 | 0.000002651 | 0.443980 | 0.021668 | 0.022996 |
| 4 / 60 | 0.140100 | 0.133407 | 0.006687 | 0.000006323 | 0.436417 | 0.020791 | 0.022761 |
| 5 / 75 | **0.129947** | **0.120110** | **0.009822** | **0.000014556** | **0.432101** | **0.021097** | **0.022642** |

At the final snapshot, unweighted L2-SP is **0.0002563392** and weighted penalty **0.02023213**. This differs from the epoch mean because weights change during the epoch. C11's corresponding final-epoch classification mean was **0.042739**, about one-third of C13's value, consistent with constrained fitting. Chair-positive loss mean is 0.02012490 versus C11 0.00364119. Full positive/negative/Chair-specific loss partitions are in `training-diagnostics.json` and `epoch-losses.csv`.

| Epoch | Median projection data-gradient norm | Median regularizer-gradient norm | Median regularizer/data norm ratio |
|---|---:|---:|---:|
| 1 | 20.15729 | 1.82967 | 0.08237 |
| 2 | 11.68275 | 2.10344 | 0.18408 |
| 3 | 9.33832 | 1.81472 | 0.19387 |
| 4 | 8.81063 | 1.81501 | 0.20774 |
| 5 | 6.96261 | 1.82874 | 0.26236 |

Gradient diagnostics are **after AMP unscale and before global clipping**. The regularizer gradient is analytic `lambda × displacement`; the data component is obtained by subtracting it from the accumulated gradient, subject to FP32/AMP rounding. Ratio medians are computed per-update and are not ratios of the separate medians above.

Across all 75 updates, the median ratio is **0.18590**. The regularizer norm exceeds the data norm in **two updates**, exceeds half the data norm in six, and has maximum ratio **2.96103**. It therefore did **not** globally overwhelm the data gradient. It exerted a sustained restoring constraint, and occasionally dominated low-data-gradient updates. The preflight ratio estimate on C11's larger-displacement trajectory was counterfactual; the actual new trajectory stayed much closer to pretrained weights and had smaller regularizer gradients. Norms alone do not measure directional cancellation or AdamW update influence.

There were **1,443 added Chair-negative term occurrences** during training; median score **0.00072649**, maximum **0.12326005**. **None reached 0.15 or 0.50.** Zero protected-positive/mask violations occurred. Thus the valid added negatives remain mostly easy patches, not the high-confidence confusion driving the reported Chair-FP problem. No uncertain region was automatically labelled negative.

Milestone internal diagnostics (not alternate model selection):

| Epoch | Precision @0.15 | Recall | AP50 | AP75 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| 0 | 62.69 | 68.85 | 74.99 | 55.28 | 42 / 25 / 19 |
| 1 | 30.07 | 70.49 | 76.32 | 57.67 | 43 / 100 / 18 |
| 3 | 19.03 | 77.05 | 75.92 | 57.80 | 47 / 200 / 14 |
| 5 | 14.45 | 81.97 | 75.27 | 52.95 | 50 / 296 / 11 |

After epoch one, projection distance decreases while fixed-threshold FPs still increase. Parameter distance alone does not adequately control the relevant classifier function. No earlier checkpoint was promoted after seeing these curves.

## Answers and decision

1. **Did drift decrease substantially?** Yes: joint projection displacement decreased 84.48%.
2. **Did shared inflation decrease?** Yes: mean +3.01103 → +1.82135 logits, a 39.51% reduction, but every aligned query still increases in all seven categories.
3. **Did fixed-threshold precision improve?** Yes versus C11 (8.76% → 14.45%), while remaining far below original (62.69%).
4. **Did matched-recall precision improve?** No versus C11: 56.00% vs 56.76% at original recall; 15.58% vs 18.45% at the C13 recall target. C11's 90.16% recall is not attainable under the frozen C13 scoring protocol.
5. **Did recall decrease significantly?** The observed drop is materially concerning: five lost TPs, all Chairs; overall −8.20 points and Chair −12.82 points. Exploratory paired bootstrap over the 19 known source groups (2,000 resamples, seed 6106) gives a C13−C11 recall-delta 95% percentile interval **[−14.61, 0.00] points**. It touches zero and cannot establish a general-population statistically significant effect. The precision-delta interval is [+2.77, +14.06] points under this same partial protocol, not independent accuracy evidence.
6. **Did categories regress?** Chair recall/AP50/AP75 regress; Sofa AP75/localization regresses despite unchanged recall; Bed recall is preserved and AP improves slightly.
7. **Was the regularization gradient too dominant?** Not generally by measured norm: it exceeds data in only two updates. Nevertheless, the prior strongly constrains adaptation and produces an unfavorable recall/ranking/localization tradeoff at this fixed strength. The outcome label B describes that observed detection tradeoff, not proof of universal gradient domination or an optimal alternative coefficient.

**Decision B is provisional on the existing native-label panel.** The run is reproducibly comparable; its limitations do not invalidate the measured parameter and internal-protocol differences. They do prevent independent real-world accuracy or acceptance claims. There is one seed, only 19 exposed photos, three supported categories, sparse FP-complete coverage, native Stool ambiguity and incomplete annotations. The existing independent real-room proposal remains unfulfilled; no reserved material was consumed to resolve it.

**One recommended next action:** audit saved optimization-training query outputs to locate high-confidence, source-supported Chair-negative examples that the current easy regions do not supervise, then prepare a single controlled hard-negative-supervision experiment for separate approval. Preserve genuine/missing Chairs and ambiguous regions as ignored. Do not automatically launch another run or lower lambda based on this panel. No additional experiment is recommended for immediate execution before that evidence audit.

## Artifacts and preservation

- Experimental full-head checkpoint: `data/open-images/experiments/phase10c12-projection-preservation-v1/checkpoints/update-075-4972103e1d39.pt`.
- SHA-256: **`f74aba1fd4d44a33ffa0caed150db182ad37f50000c94b83fedceda6fc7fd994`**.
- The previously prepared C12 destination was fresh and intentionally retained for the C13 execution. No old experiment directory was overwritten.
- Raw per-image predictions: that directory under `evaluation/epoch-{0,1,3,5}/`; postprocessed predictions/metrics: `cpu-evaluation-v1/`. Training probes and all per-update checkpoints/logs remain separate from historical artifacts.
- Public detailed data: `results.json`, `training-diagnostics.json`, `confidence-drift.json`, `localization-comparison.json`, `operating-point-comparison.json`, `precision-recall.csv`, `epoch-losses.csv`, `diagnostics.svg`, `integrity.json` and provenance/seal records.
- **149 preflight CPU tests passed**. Nine post-run C13 CPU tests passed, including four additional uncertainty-analysis tests: **153 distinct relevant tests**. All 76 actual checkpoint receipts, frozen/reference hashes, optimizer cursors, source order, penalty scaling and zero mask violations were verified.
- Preservation passed for C12's 27 public/13 private artifacts, C11's 31 public/626 private artifacts, 1,310 C8 sealed artifacts, 9,973 historical data/model files, 160 Phase 10B snapshots, 24 prior report artifacts, 27 historical implementation files and all 78 eligible source-image hashes. Original OWLv2 remains SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`; C11 remains SHA-256 `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f`.
- No original annotations/partitions, model defaults, SigLIP 2 weights, historical predictions, or Mac application files were modified. Preservation hashing reads bytes; it does not decode or score frozen photographs.
- Small code/configuration/reports are committed and pushed on the new C13 branch. Weights, photograph bodies, raw queries, activation caches and private logs stay local. `main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

The single queue is complete. No further training, downloads, threshold/loss-weight adjustment, deployment or Phase 10D execution occurred or is automatically scheduled.
