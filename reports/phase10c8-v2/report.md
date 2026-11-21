# Phase 10C.8 — controlled Chair-negative supervision

The correction measurably changed the learned model and reduced scored Chair
false positives by **24 / 660 (3.64%)** relative to its matched control. It did
**not** prevent confidence inflation. At the frozen confidence threshold 0.15,
the corrected model has **6.86% precision / 95.08% recall**, compared with the
original detector's **62.69% / 68.85%**. Neither arm supports deployment or an
independent accuracy claim. No additional training was launched.

## Design and integrity

- Identical original OWLv2 initialization, processor, prompts, seven-category
  ordering, seed 6106, sampling order, optimizer and 75 updates in each arm.
- Original model SHA-256:
  `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.
- 59 optimization photographs / 56 known source groups / 182 positive boxes:
  Chair 114, Sofa 40, Bed 28. All positive and ignored annotations preserved.
  Group counts establish known isolation, not proof of all physical identities.
- AdamW LR 1e-5, weight decay .01, betas .9/.999, epsilon 1e-8; batch one,
  accumulation four (last group three), gradient norm clipping one. Five epochs.
- Exactly 12 class-head and box-head tensors / 1,579,526 trainable parameters.
  Vision, text, objectness and all other parameters stayed frozen and byte-equal.
- FP16 autocast with FP32 weights/loss; GradScaler 128. Classification/L1/GIoU
  weights 1/5/2 and focal alpha/gamma .25/2 unchanged.
- A uses the legacy Phase 10B loss. B adds only the validated source-supported
  Chair-negative mask; unsupported negatives remain ignored. The old training
  pool has zero explicit Chair-negative photographs (two Sofa and one Bed).
- 41 native source-polygon regions in 24 training photographs. This is
  assistant-inspected source evidence under the C6 policy, **not human-reviewed
  evidence**. No reviewer flags or missing human decisions were fabricated.
- Chair includes seating stools and bar stools. Footstools, ottomans and step
  stools stay separate. Positive Chairs, seating and uncertain objects retain
  protected regions; all actual training/probe protected violations were zero.
- Before results, epoch five was fixed as the primary endpoint. Epochs one and
  three are descriptive checkpoints. No adaptive epoch or threshold selection.

The first worker startup failed on a local import path before any model forward
or optimizer step. That v1 failure record is retained. A standalone subprocess
import regression test was added and a fresh v2 preflight passed. No training
updates were replayed. See `startup-recovery.json`.

## Fixed internal diagnostic evaluation

The same 19 already-exposed, training-derived development photographs / 19 known
source groups / 61 supported GT instances were used: Chair 39, Sofa 13, Bed 9.
None entered gradient updates. No reserved, frozen or official holdout was scored.

FP32 inference, original preprocessing and prompts, score floor .001, classwise
NMS IoU .5 / maximum 300, operating confidence .15, matching IoU .5 and .75,
101-point supported-class macro AP. Raw per-image GPU results were saved
atomically, then postprocessed predictions saved before metric calculation in
a separate CPU-only process using unchanged recovery metrics. All seven
original-model headline values exactly reproduce the historical reference.

Partial-label rules remain essential: only four photographs support unmatched
Chair FP scoring, one supports Sofa FP scoring, and five support Bed FP scoring.
Other unmatched predictions remain unknown or ignored. Table, Desk, Cabinet and
Lamp have no eligible GT in this protocol: their precision/recall/AP are **N/A**,
not zero. Native completeness flags were not relaxed. Scored unmatched boxes
can still reflect imperfect source annotations; they are not all independently
confirmed mistakes. Precision here is conditional on native annotation scopes.

| Model, epoch 5 unless stated | P @.15 | R @.15 | AP50 | AP75 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Original, fresh zero-update reference | 62.69% | 68.85% | 74.99% | 55.28% | 42 / 25 / 19 |
| Phase 5C historical reference* | 57.53% | 68.85% | 74.51% | 58.51% | 42 / 31 / 19 |
| Phase 10B historical LR 1e-5 | 6.83% | 90.16% | 74.17% | 60.23% | 55 / 750 / 6 |
| Phase 10B recovered LR 3e-5 | 4.11% | 88.52% | 52.75% | 30.95% | 54 / 1261 / 7 |
| A: matched legacy control | 6.45% | 91.80% | 74.23% | 55.30% | 56 / 812 / 5 |
| B: corrected negatives | 6.86% | 95.08% | 75.25% | 55.21% | 58 / 788 / 3 |

Historical eligibility-manifest SHA exactly matches the current frozen input:
`ec55e01d968b584819c790de2c2a843bd87d9af960642601d65187807800b937`.
The historical and new models use the same evaluation definitions; historical
training seeds/progress differ, so only the new matched A/B estimates the
correction's effect. *Three of these internal photographs overlap Phase 5C's
earlier training: its row is context, not an independent generalization result.

## Did the additional evidence become influential?

The following are means over the same 24 optimization photographs, with no
optimizer update during probes. A's added contribution is counterfactual; only
B applies it to training. Actual initial query arrays and added-gradient norms
are identical. Current-model query boxes can subsequently move, so membership
and coverage are remeasured rather than assumed constant.

| Epoch | A / B added query terms | A added focal loss, FP64 | B added focal loss, FP64 | A / B mean additional-gradient norm |
|---|---:|---:|---:|---:|
| 0 | 292 / 292 | 1.14718e-7 | 1.14718e-7 | .000177 / .000177 |
| 1 | 293 / 293 | 7.40526e-6 | 7.40526e-6 | .01099 / .01099 |
| 3 | 284 / 284 | .00033960 | .00032513 | .47684 / .45863 |
| 5 | 284 / 285 | .00138989 | .00122137 | 1.90850 / 1.71261 |

Initially the ratio of mean additional-gradient norm to mean used-classification
gradient norm is 1.45e-6. By epoch five it is 20.22% for A's counterfactual probe
and 20.05% for B. These are ratios of per-photo mean norms, not a vector-sum
attribution or causal percentage. Gradients were measured at the actual AMP
scale and unscaled numerically. FP64 loss diagnostics distinguish tiny values
from FP32 summation effects without changing the actual optimization loss.

Only 11 photographs have nonzero added AMP gradients at initialization despite
19 with added query terms; by epoch five 18 have nonzero gradients. Early tiny
terms may be rounded/underflowed in FP16 propagation. The later signal clearly
is not zero. At epoch five, B covers 24 of 41 regions across 19 source groups;
17 regions still receive no valid query. Only four added queries exceed .15.
Median/max added Chair probability is .00906/.30996 in B versus .00915/.32498
in A. The supervision remains spatially sparse and mostly easy background.

On those 24 photographs, B's mean positive classification loss changes from
.869236 to .013556; added-negative loss grows from 1.14718e-7 to .00122137;
legacy-negative classification remains zero on this particular probe subset;
weighted box loss changes from .443301 to .461791. The corresponding final A
positive and weighted box losses are .013172 and .458493. Whole-pool per-update
components and actual classification-head gradient norms are in
`training-curves.csv`; all probe components are in `negative-signal.csv`.

Final class-head and box-head tensors differ, while frozen tensors are identical.
For example, class-head dense-weight delta L2 is .002355 (relative .000243).
Box-head differences are compatible with changed matching/global clipping under
the same procedure; no extra box objective or architecture change was introduced.

## Precision, recall and localization effects

Compared with A, B gains 0.404 percentage points precision (6.26% relative),
3.279 points recall (3.57% relative), and 1.017 points AP50. AP75 falls .091
points. It removes 24 scored FPs and gains two TPs. Both models still show
severe confidence inflation compared with original OWLv2.

| Category | A P / R | B P / R | A TP / FP / FN | B TP / FP / FN | A / B AP50 | A / B AP75 |
|---|---:|---:|---:|---:|---:|---:|
| Chair | 5.44% / 97.44% | 5.78% / 100% | 38 / 660 / 1 | 39 / 636 / 0 | 66.93% / 66.87% | 43.68% / 43.41% |
| Sofa | 26.32% / 76.92% | 28.21% / 84.62% | 10 / 28 / 3 | 11 / 28 / 2 | 76.24% / 79.35% | 55.45% / 55.45% |
| Bed | 6.06% / 88.89% | 6.06% / 88.89% | 8 / 124 / 1 | 8 / 124 / 1 | 79.52% / 79.52% | 66.79% / 66.79% |

Chair FP reduction is 3.64% with recall increasing 2.564 points; Sofa recall
increases 7.692 points and Bed recall is unchanged. Chair AP50/AP75 slightly
regress. For the exact same matched GT instances, mean IoU A→B is:
Chair .77428→.77043 (38 instances), Sofa .87383→.87331 (10), Bed .87949→.87989
(8). B's extra Sofa TP has IoU .50817, explaining much of its lower unpaired
mean IoU .84011. Comparing unpaired TP means alone would overstate regression.

Duplicate FP counts are 4 / 4 for A/B, versus 1 original. Fragmentation heuristic
counts are 47 / 47, versus 4 original; mixed-object candidates 80 / 79 versus
15; category-disagreement candidates 27 / 28 versus 14. These are geometric
diagnostics, not visually confirmed failure labels. The correction did not
resolve fragmentation or duplicate behavior.

All 660 / 636 scored Chair FPs concentrate in these four FP-complete photographs:

| Photograph | Original Chair FP | A Chair FP | B Chair FP |
|---|---:|---:|---:|
| `757a1631131f4ba3` | 12 | 242 | 240 |
| `22a5848e375335e3` | 1 | 182 | 174 |
| `coco2017:000000452459` | 0 | 143 | 131 |
| `d224c4e1ea5c3d49` | 6 | 93 | 91 |

`per-photo-chair.csv` includes every development photograph. Zero counted FP on
an unknown scope does not mean no false positives. No historical FP review
decisions were changed or invented for the new predictions.

Post-NMS Chair-score medians are original .00846, A .18604, B .18162; 90th
percentiles .08019/.58307/.57696. Chair predictions above .15 across all scopes
grow from 57 original to 2201 A / 2130 B. Those counts include unknown scopes
and are not interchangeable with scored FPs. Raw all-query Chair median rises
from .0000136 to .004124/.003918. B reduces inflation modestly, not decisively.

At original recall 42/61 = 68.85%, the descriptive PR envelope gives A **82.35%**
precision and B **80.77%**. Thus B is **1.58 points worse at matched recall**,
despite a small fixed-threshold improvement. At recall at least 90%, the best
available precision is 16.42% A / 15.71% B. Neither saved PR curve contains a
joint 80%-precision / 90%-recall operating point. These post-hoc thresholds are
diagnostics only; none was adopted or used to retrain/select a model.

## Resources, artifacts and preservation

| Arm | Epochs / optimizer updates | Worker wall time* | Peak allocated / reserved / observed device VRAM |
|---|---:|---:|---:|
| A | 5 / 75 | 182.31 s | 1.89 / 2.17 / 3.36 GiB |
| B | 5 / 75 | 174.78 s | 1.89 / 2.17 / 3.36 GiB |

*Includes model load, probes, snapshot verification and diagnostic inference.
Whole queue: 360.58 seconds. 590 training backwards; 384 probe `autograd.grad`
calls without optimizer updates; 915 total model forwards including training,
probes and permitted internal diagnostics. No nonfinite event or memory-limit
violation. Device monitoring sampled at 50 ms plus per-boundary allocator checks.

Final immutable head-delta checkpoints (both retained, neither deployed):

- A: `data/open-images/experiments/phase10c8-ab-v2/A-control/checkpoints/update-075-65391d9c26e1.pt`
  SHA-256 `791cc41cacd4c8fad1a3ea8d6a822323c01743f3bb379b519abc94da64a4a44e`.
- B: `data/open-images/experiments/phase10c8-ab-v2/B-corrected/checkpoints/update-075-9164284d514b.pt`
  SHA-256 `986d511ae2c80ced12a4a8542bdd300af0477230246a4df1ef22ffdf8519a930`.

Checkpoint payload epoch cursors are zero-indexed: final cursor epoch 4 means
the fifth completed epoch; both completion records explicitly report five.
150 per-update checkpoints were independently reloaded, SHA-checked and checked
against optimizer step counters; two initial snapshots have identical weights
and pristine optimizer state. Private checkpoint registry, predictions, masks,
source IDs, RNG/scaler states and immutable update intents remain under the
versioned data directory. About 2.74 GiB was allocated at analysis, below the
self-imposed 4 GiB bound. No photographs or weights are included in Git.

235 CPU tests passed across eleven relevant suites. Post-run preservation
reverified 9,973 historical data/model files, 160 Phase 10B snapshots, 24 prior
reports, 27 historical implementation files, all 78 eligible source-image hashes
and 619 additional evidence-seal entries. All original model/processor hashes
and frozen parameter hashes remain intact. Existing partitions, production
defaults and the separate Mac application were untouched. See
`tests-post-run.json`, `preservation-post-run.json` and `execution-provenance.json`.

## Decision and next experiment recommendation

1. **Causality:** the source-negative mask changes real gradient contributions,
   model weights and predictions in this controlled run. The initial conditions
   and sample order match. One seed cannot establish robustness across seeds.
2. **Precision:** a small reduction in scoped Chair FP at .15, but persistent
   extreme inflation and slightly worse matched-recall precision. This is not a
   successful precision repair or an independent accuracy improvement.
3. **Recall:** no aggregate or category recall loss against A; one extra Chair
   and Sofa match. Chair localization/AP slightly regress. Sofa/Bed FP inflation
   versus the original remains substantial and is not concealed by recall.

The current masks provide a real late-training signal but insufficient coverage
to solve the failure. The evidence does **not** justify automatically raising
loss weights: late gradient norms already reach a material scale, while only
four queries in the source-negative probe exceed .15 and 17 regions are uncovered.
Do not infer that every unmatched proposal is safe negative evidence.

Recommended next work is a **no-update coverage audit** linking saved high-score
queries to independently source-supported negative objects, followed by targeted
human/source verification where current evidence is missing. Before another
GPU run, freeze any expanded negative manifest and demonstrate meaningful
hard-query coverage. A future separately approved alternative is a box-head-only
diagnostic with classification frozen, preserving confidence behavior; it tests
localization and must not be presented as a fix for classification inflation.
Do not repeat this same A/B or tune a multiplier on these 19 exposed photographs.

A separate 140-room / 140-source-group real-photo acceptance proposal is in
`independent-evaluation-proposal.json`: complete seven-class annotations,
at least 50 instances per class, independent annotation QA, conservative ignores,
strict train/source/perceptual isolation, predeclared operating points and
source-group uncertainty intervals. It is a proposal only: no acquisition,
annotation campaign or acceptance scoring was executed. The 80% precision /
90% recall requirement remains a future independent target.

No further experiment or Phase 10D is authorized by this completion. The bounded
queue is finished; no production model or threshold was replaced.

Supporting artifacts: `comparison.json`, `category-metrics.csv`,
`per-photo-chair.csv`, `negative-signal.csv`, `training-curves.csv`,
`matched-localization.json`, `operating-points.json`, `diagnostic-curves.svg`.
Reproduction/monitoring commands and recovery boundaries are documented in
`tools/phase10c8-v1/README.md`.
