# Phase 10C.9 — why high-confidence Chair mismatches persist

The leading supported explanation is **broad classification-confidence inflation
under sparse negative supervision**, combined with fragmented/mixed furniture
boxes and incomplete annotations. The correction has a real gradient effect,
but **no added Chair-negative occurrence in the entire C8 training run had a
Chair score at least .50**. This investigation executed zero model forwards,
zero optimizer updates and zero dataset/model downloads.

Dashboard: **http://localhost:8772**. It is read-only and shows original, control
and corrected predictions in identical image coordinates. No production setting,
primary metric, annotation, partition or historical result was changed.

## What was verified

Started from commit `02e7e6a` on the completed C8 branch; new branch
`experiment/phase10c9-confidence-investigation`. Verified all 32 C8 public seal
entries, 1,310 private artifact seal entries, the pinned model/processor/code
binding and all 78 eligible source photographs' hashes. All three models have
complete saved predictions for the same 19 development photographs. No missing
predictions were synthesized.

The corrected model's preserved results reproduce exactly: aggregate P/R at .15
**6.8558% / 95.0820%**, AP50/AP75 **75.2456% / 55.2134%**, TP/FP/FN **58/788/3**;
Chair FP **636**. These are the existing partial-label internal diagnostics,
not independently verified seven-class accuracy.

Each post-NMS prediction was joined back to its exact raw query index by
reproducing the original score-floor .001, classwise NMS .5 and cap 300. The
resulting boxes/scores/categories must equal the saved list exactly. Prediction
IDs contain model, source image and query index, while original list indices
and source SHA-256 remain available privately. Scoring uses the existing saved
FP32 GPU box transformations, including original clipping. New geometry mining
uses unclipped inverse coordinates and rejects out-of-frame boxes instead of
turning clipped remnants into negative evidence.

## Chair score thresholds and annotation limitations

Counts below are **Chair only**, after unchanged original NMS. Unknown means
unmatched but not eligible for FP scoring. Only four of the 19 photographs
permit native Chair FP scoring; 39 active Chair instances supply the recall
denominator. The remaining photographs cannot establish absence of Chairs.

| Model | Threshold | All Chair predictions | TP | Native-scored FP | Unknown unmatched | Chair recall |
|---|---:|---:|---:|---:|---:|---:|
| Original | .15 | 57 | 24 | 19 | 14 | 61.54% |
| Original | .50 | 3 | 3 | 0 | 0 | 7.69% |
| Original | .80 | 0 | 0 | 0 | 0 | 0% |
| A control | .15 | 2201 | 38 | 660 | 1503 | 97.44% |
| A control | .50 | 521 | 35 | 222 | 264 | 89.74% |
| A control | .80 | 106 | 28 | 40 | 38 | 71.79% |
| B corrected | .15 | 2130 | 39 | 636 | 1455 | 100% |
| B corrected | .50 | 497 | 34 | 211 | 252 | 87.18% |
| B corrected | .80 | 103 | 28 | 39 | 36 | 71.79% |

The corrected Chair precision is 5.78%, 13.88% and 41.79% at these thresholds,
respectively, under the conditional native scoring scopes. Original .80
precision is undefined because it emits no Chair predictions. Increasing the
threshold discards real Chairs; it is not evidence of a repaired detector.
All seven categories' threshold counts are in `threshold-counts.csv`; unsupported
category precision/recall stays undefined, not fabricated as zero accuracy.

Every .15-scored Chair FP is concentrated in these four images:

| Photo | Original FP | Control FP | Corrected FP |
|---|---:|---:|---:|
| `757a1631131f4ba3` | 12 | 242 | 240 |
| `22a5848e375335e3` | 1 | 182 | 174 |
| `coco2017:000000452459` | 0 | 143 | 131 |
| `d224c4e1ea5c3d49` | 6 | 93 | 91 |

`photograph-counts.csv` additionally gives .50/.80 counts for every model/photo.
Zero scored FP on a photo with incomplete scope is not evidence of no errors.

The corrected .15 FP boxes have areas: **449 below 1%** of the photograph,
132 at 1–5%, 41 at 5–20%, 14 at least 20%. Thus 70.60% are tiny by the stated
area heuristic and 91.35% occupy under 5%. Even among the 39 FPs scoring at
least .80, 26 are below 1% area. Smallness alone does not establish an error.

Geometry flags among 636: two exact same-category duplicates of already matched
Chairs, 157 fragment candidates, 129 with IoU at least .1 to known Chairs, 48
boxes altered by frame clipping and two queries anchored in padded image areas.
Flags overlap and are not exclusive labels. **Padding is not the dominant
explanation**; no .50/.80-scored FP has a padding anchor. Classwise NMS already
removes many high-IoU overlaps, so Chair parts with low mutual IoU survive.

## Error taxonomy, visual evidence and completeness gap

Taxonomy is preserved as separate evidence layers:

- A: genuine non-furniture confusion.
- B: other furniture classified as Chair.
- C: real Chair/seating stool missing from active annotations.
- D: duplicate of an annotated Chair.
- E: fragment, mixed furniture or poor localization.
- F: unresolved because of incomplete labels or ambiguous content.
- G: another verified mechanism, requiring explicit evidence.

At full-dataset scale, **two of 636** have source-matched duplicate evidence;
the other **634 receive no source-verified mechanism from that automated check**.
The sample below supplies additional judgments, not a complete adjudication of
the set. Automatic overlap and fragment flags remain hypotheses. A source box around a plant/person/table does
not prove that all pixels inside it exclude seating.

Actually inspected a deterministic, targeted sample of **32 scored FPs** across
the four photographs and confidence bands, plus **eight unknown predictions**.
These are not random prevalence samples. The 32 assistant visual judgments are:

| A | B | C | D | E | F | G |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2 | 2 | 1 | 14 | 8 | 0 |

The eight unknown controls yield A1/B1/C2/E2/F2. Source-supported object overlap
is distinguished from assistant judgment; **none is a new human confirmation**.
All 52 visual observations, including 12 training-query examples, are bound to
the exact sample digests and nine actually viewed contact-plate hashes. Changed
plates/queries require new inspection rather than automatic reuse of opinions.

Representative evidence, available as dashboard cards:

- **S001**, score .956: an apparent red-topped seating stool outside active Chair
  boxes. Native metadata calls it only **Furniture**. Project Chair includes
  seating/bar stools (not footstools, ottomans or step stools); the stool's
  function is an assistant judgment, not independently confirmed. The native Chair-completeness claim is insufficient
  for the broader project category. S006/S007 crop its frame/seat. These should
  not automatically become non-Chair negatives.
- **S002/S005/S008/S014/S016/S017/S022/S030/S032**: annotated Chair parts or
  incomplete boxes. Classification can be semantically correct while localization
  fails IoU. Calling them all background would contradict the positive objects.
- **S019**: duplicate office-Chair prediction, independently flagged by the
  unchanged matching implementation and visually inspected.
- **S020/S024**: bed structure and a desk drawer/storage unit. S024 has native
  drawer evidence; no automatic Cabinet/Desk taxonomy change follows.
- **S023**, score .646: container/trash-can confusion with native `trash_can`
  box IoU about .98 plus visual support. **S015/S018/S029/S031** are visual
  hypotheses involving window detail, an instrument and tabletop objects, with
  weaker or absent native object support.
- **S027/S028**: boxes spanning the table and multiple actual Chairs.
- **S033/S034**, already unscored: real wicker/folding Chairs with additional
  native Chair overlap .93/.85 but absent from active project Chair GT. Additional
  native annotations were joined for diagnostics only and remain inactive.
- People/seating mixtures and distant plant/architectural details remain F when
  the photographs cannot support a definite identity.

The diagnostic-only native join streamed existing local COCO/LVIS/Open Images
metadata, retaining 607 annotation records (including cross-source overlap).
Those are **not 607 independent objects**. They provide context, not changes to
primary GT, completeness flags or training supervision. No validation diagnosis
or extra native label was transferred into optimization data.

## What the actual architecture and objective do

Audited installed Transformers 4.57.6 `modeling_owlv2.py` and the executed C8
worker, loss and postprocessor. Source paths/hashes are in `objective-audit.json`.
This is a dense patch detector, not a fixed small set of DETR-style learned
object queries: the saved image produces **60×60 = 3,600 patch features**.

For patch q and text category c:

```
u[q] = normalize(class_head.dense0(frozen_image_feature[q]))
t[c] = normalize(frozen_text_embedding[c])
z[q,c] = (dot(u[q], t[c]) + shift(image_feature[q]))
         * (ELU(scale(image_feature[q])) + 1)
score[q] = sigmoid(max_c z[q,c])
label[q] = argmax_c z[q,c]
```

Shift and scale are image-feature-dependent scalar predictions shared by all
seven category logits for a patch. They are not seven independent class biases.
Text/image vectors are normalized before similarity, but the final affine logits
are not calibrated real-world probabilities.

**Objectness exists** and is computed by a separate frozen head. The actual C8
loss and postprocessing **do not use it**: no objectness multiplication, no
top-objectness query selection, no objectness/no-object training term. Every
patch is eligible for category scoring before the score floor/NMS. Objectness
and intermediate feature vectors were not cached, so retrospective objectness
gating or exact internal-layer attribution would require separately approved
new inference. No missing intermediate values were invented.

Hungarian matching uses `-sigmoid(class_logit) + 5*L1 + 2*(1-GIoU)` against
existing positives. Matched class cells get target one. Other classes on those
queries and unmatched unknown regions remain ignored unless explicit native
negatives/source regions support target zero. Sigmoid, not softmax, allows
multiple category logits to rise together. Increasing shared shift can reduce
matched-positive loss without improving discrimination on ignored objects.

This objective **does not explicitly mark every query positive**. The problem is
parameter sharing: positive-only gradients can raise scores on many ignored
queries/classes without a corresponding loss penalty. Saturating sigmoid scores
can also compress differences in the classification matching cost; saved data
does not identify how much individual assignments changed. Added mask membership
is nondifferentiable; it adds no direct box-regression target. Shared clipping
and changed matching can nevertheless alter subsequent box-head updates.

## Direct evidence of broad confidence inflation

Compared each original/adapted raw logit vector at the same patch index and
photograph, with the same frozen input/feature pipeline. Across **68,400 saved
development queries**, **all seven logits increased at every query** in both
arms. For corrected B, mean increases are:

| Sofa | Chair | Table | Bed | Desk | Cabinet | Lamp |
|---:|---:|---:|---:|---:|---:|---:|
| +5.1873 | +5.4225 | +3.5730 | +5.2347 | +3.8862 | +3.1726 | +3.2943 |

Subtracting each query's seven-category mean logit change leaves a semantic
residual. The shared component accounts for **95.45% of squared logit change**
in B (95.46% A); its median is **+4.2228** logits (A +4.2509). This is a
statistical common-mode decomposition, **not proof that the calibration modules
alone caused 95% of the drift**. Dense projection changes can also shift all
similar furniture prompts. Both projection and calibration parameters changed.

Corrected versus original class-head relative tensor changes: dense weight L2
1.50%, shift weight 1.26%, scale weight .132%; shift bias increases .000361 and
scale bias decreases .000289. Weight-delta signs/norms cannot recover their
input-dependent activation effects. Exact separation is unavailable without
intermediate embeddings or a controlled parameter-freezing ablation.

The correction lowers the mean Chair change only about .0501 logits versus A,
while the original-to-B mean increase is +5.4225. This matches a modest change
in global inflation rather than a wholesale repair. Ranking is not uniformly
destroyed: AP can remain respectable while the fixed .15 operating point fails.
That does not establish improved independent discrimination.

## Why the trusted negative inventory does not cover the failures

Training positives: Chair **114/182 = 62.64%**, Sofa 40, Bed 28; the other four
categories have no active positive target. Explicit native whole-image negatives:
**zero Chair**, two Sofa photographs, one Bed photograph. The seven prompts still
share trained classification parameters.

The 41 source regions span 24 photographs/source groups: person 17, curtain 7,
TV/television 5, laptop 3, Christmas tree 2, potted plant 2, refrigerator 2,
backpack/handbag/bowl one each. Their median area is 1.02% of the image. Scale
alone is not the whole mismatch: small detections dominate both collections,
but many failing boxes are Chair parts, storage/bed structures or loose
multi-object/context crops absent from safe non-Chair interiors.

In the corrected run's actual training logs, 295 photo occurrences / 1,062,000
query occurrences produce **570 matched Chair-positive target occurrences** and
**1,429 additional Chair-negative occurrences**. These are repeated training
occurrences, not independent examples. Added-negative scores have median .00263,
90th percentile .02203, maximum **.30767**. Only **seven** occurrences reach .15;
**zero reach .50 or .80**. The 41 regions therefore never directly supervise a
high-confidence Chair distractor under this experiment's actual matching/mask.

At initialization the negative probe median is .000158. For negative focal
loss with alpha .25 / gamma2, the per-cell value is approximately
`.75*p^2*(-log(1-p))`: about **2.96e-12** at that score, versus **.12997** at .5,
a factor of **4.40e10** before the same normalization. Very easy negatives have
almost no early effect; C8 measured some FP16 gradient underflow/rounding.

Crucially, the implemented loss sums valid focal terms and divides by the
**positive GT count per image**, not by 3,600 queries or number of negatives.
There is no hidden all-query averaging that explains the tiny terms. The same
divisor scales positive and negative cells; images with more GT reduce each
cell's absolute contribution. Coverage, score-dependent focal damping and
parameter sharing are stronger supported concerns than a normalization bug.

The signal becomes real later: B's 24-photo mean added loss grows
1.147e-7→.001221; ratio of mean added/used-classification gradient norms grows
1.45e-6→.2005. These are per-photo mean-norm ratios, not a causal percentage of
the update. B still covers only 24/41 regions in 19 source groups at epoch five;
17 regions have no valid query coverage. A larger loss multiplier is therefore
not automatically justified: sparse mostly easy evidence is a central limitation.

## Training-only hard-negative search

Available metadata: 1,005 native records across the exact 59 optimization photos.
Saved C8 query caches cover **24**, not all 59; the other **35 remain unmeasured**.
No new inference was used to fill that gap. The search uses only corrected
epoch-five saved training queries whose top category is Chair and score ≥.50:

| Stage | Raw query count |
|---|---:|
| Initial high-score queries | 1363 |
| Out of original frame, rejected | 160 |
| Overlap positives, ignored or uncertain protected regions | 1095 |
| Unprotected but still unverified | 108 |
| New full source-polygon certificates | **0** |
| Newly trustworthy training-eligible hard-negative regions | **0** |

Certification requires an entire integer-enclosed query rectangle inside an
eligible existing native non-Chair polygon with 2px clearance, at least 32px
per side, outside every positive/ignored/uncertain region and existing negative
region. No loose source box, predicted category or partial polygon intersection
is accepted as proof of absence. Duplicated query proposals are not counted as
new independent objects. No rule was weakened to produce a nonzero count.

Inspected 12 diverse high-score unverified training queries: four appear
non-Chair, one appears to be a real unannotated Chair, seven remain uncertain.
Examples include a glass, basket, person/clothing region, plant, luggage-rack-like
structure, furniture edges and mixed TV/storage context. They **remain inactive**
because complete source/spatial proof is missing. T009, apparently a genuine
Chair outside protected annotations, demonstrates the danger of automatically
mining every unmatched training prediction as background.

The versioned private `hard-negative-proposals.json` contains zero certified
new regions and a separate 108-query unverified tier. It cannot activate
supervision. Development, frozen and reserved images were never mined.

## Alternative corrections and one recommended ablation

| Option | Feasibility and evidence | Main restriction/risk |
|---|---|---|
| A. Verified hard-negative mining | Infrastructure implemented; zero new strict certificates from current saved-query coverage | Need source-supported hard-object masks/verification or separately approved additional training-only inference; real seating must stay protected |
| B. Balanced positive/negative objective | Technically simple; separate components can be measured | Raising weights cannot create missing hard evidence; class-conditional normalization/weight changes need their own one-change control |
| C. Different sampling | Stratify verified regions by score, source and scale; cap correlated repeated terms | Do not use predicted non-Chair labels as truth or silently label other portions of a photograph negative |
| D. Selected classifier components | Freeze shared shift/scale; continue semantic dense projection with unchanged corrected loss | **Recommended minimal next ablation**; dense projection may still cause common drift, so success is not assumed |
| E. Preserve pretrained behavior | Teacher-logit regularization on optimization images is technically possible | Teacher outputs for all 59 are not cached; preserve genuine positives, and do not mix distillation and calibration freezing in one untested change |
| F. Calibration/threshold | Saved PR curves and .15/.50/.80 diagnostics are available | Thresholds lose recall; no joint80/90 point in C8 internal PR curves. Fit any later calibration on a separate approved training-derived calibration group, never acceptance images |
| G. Postprocessing | Saved-box re-suppression can reduce overlaps | Re-suppressing existing cap300 boxes at NMS .4 yields Chair TP38/FP497; .3 gives TP33/FP400. This is not a fresh full-raw NMS benchmark and is not deployed. Fragmented parts need not overlap enough for NMS |
| H. Box-only adaptation | Existing preparation supports freezing classification and updating box head | Tests localization, not the observed logit drift. New box locations/NMS can alter reported detections even when patch logits remain fixed. Phase10D stays unauthorized |

Recommend **one calibration-frozen head ablation**, initialized from original
OWLv2: freeze `class_head.logit_shift` and `class_head.logit_scale` weights and
biases (four tensors). Keep `class_head.dense0` and the box head trainable:
**eight tensors / 1,577,988 parameters**. Keep encoders, objectness and all other
parameters frozen. Reuse the same 59/56-group/182-positive training pool, same
41 corrected regions, LR1e-5/AdamW, seed6106, order, loss, batch1/accum4 and
original19-photo diagnostic protocol. No new labels, weights or threshold tuning.

Proposed bound: **one configuration, five epochs / 75 updates, ten minutes,
6.5 GiB**, separate immutable outputs. The existing C8 B run is the matched
reference, so no automatic extra control retraining is requested. Fix final
epoch before results. Predeclare an evidence threshold of at least 20% fewer
native-scored Chair FPs at .15, Chair/aggregate recall drop at most 3 percentage
points, and matched-recall precision drop at most 2 points. Report Sofa/Bed
regressions, localization and uncertainty explicitly. These are proposed
diagnostic criteria, not independent acceptance or permission to optimize away
valid unannotated seating. The taxonomy/completeness caveat remains.

Stop on nonfinite/resource/data/frozen-weight/checkpoint failure. If no material
precision gain at comparable recall, do not extend, retune or add variants.
`next-experiment-proposal.json` is **NOT_AUTHORIZED**; the preparation-only scope
helper is CPU-tested. C9 contains no training launcher. This ablation would test
the shared-calibration hypothesis; it would not prove its cause in advance.

## Dashboard, tests and independent evaluation

Open **http://localhost:8772**. Select pool/photo, .15/.50/.80, Chair/all classes,
TP/scored-FP/unknown status and box limit. Original/control/corrected panels share
the same image. Native context is optional; source object identities and exact
query/pixel coordinates are available in box details. Cards contain the actual
assistant observations, including uncertain cases and source support. The
training view shows existing negative coverage separately. No manual drawing,
confirmation or model execution is required.

The browser interface passed **16 synthetic interaction checks**, including
filters, query identity, native/unknown separation, cohort isolation, evidence
navigation, charts, image loading and rejection of training endpoints. Installed
Windows Chrome, with GPU disabled, also rendered the actual three-panel room
dashboard; the screenshot remains private. A first live-test assertion assumed
the portrait image's width incorrectly; the UI had rendered correctly. The test
now reads the pinned actual dimensions, and the failed DOM/log remain preserved.

**102 CPU tests passed across four suites.** They cover coordinate transforms, exact prediction reconstruction,
source-only certification, group/development isolation, unknown preservation,
focal math, shared-shift gradients, frozen calibration scope, read-only HTTP
security and actually viewed observation bindings. Final totals are in
`tests-final.json`; earlier stage test records remain intact. Post-investigation
preservation verified 9,973 historical files, 160 older training snapshots,
24 prior reports, 27 historical implementation files, 78 image hashes and all
1,310 C8 private artifacts. Original OWLv2, adapted heads, SigLIP2 models,
partitions and the Mac application are unchanged. No reserved/frozen image was
decoded, mined or scored.

The independent real-room proposal remains **140 unseen source groups/photos**,
complete seven-class annotation (including explicit seating/bar-stool coverage),
at least 50 instances per class, separate annotation QA, source/hash/perceptual
isolation, predefined .15/NMS.5/IoU.5/.75 protocol, uncertainty intervals and a
single preselected-checkpoint acceptance comparison. No photo is selected by
model performance. Missing/ambiguous instances are ignored, not negatives. The
80%-precision / 90%-recall target must occur simultaneously at the frozen
operating point on that future independent data; current internal counts cannot
establish it. No future evaluation photograph was acquired or consumed.

No additional GPU inference, training or Phase10D was started. Separate approval
is required for the single proposed calibration-frozen experiment, or any other
new inference/data action. Code and small aggregate reports are intended for
the experimental Git branch; source photographs, raw predictions, contact sheets
and checkpoints stay private and local.

Supporting records: `prediction-summary.json`, `query-space-audit.json`,
`objective-audit.json`, `hard-negative-summary.json`, `visual-findings.json`,
`threshold-counts.csv`, `photograph-counts.csv`, `sample-findings.csv`,
`diagnostic-summary.svg`, `next-experiment-proposal.json`,
`independent-evaluation-protocol.json`, `preservation.json` and browser reports.
Start/restart commands are in `tools/phase10c9-v1/README.md`.
