# Phase 10E.1 — Original-teacher distillation preflight

**CPU preparation is complete; GPU execution and training remain disabled.** No coefficient is selected. The next justified step is a separately approved zero-update projection-gradient smoke, not a training run. No evidence here establishes improved detection.

## Historical evidence

Original and completed C8/C11/C13/C20/10D artifacts were inspected and hash-verified. The following are saved exposed 19-photo, partially annotated, three-positive-category diagnostics at confidence 0.15. They are contextual evidence, not independent acceptance or coefficient-selection data.

| Model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN |
|---|---:|---:|---:|---:|---:|
| 10D | 62.69% | 68.85% | 74.58% | 50.66% | 42 / 25 / 19 |
| C11 | 8.76% | 90.16% | 76.76% | 60.58% | 55 / 573 / 6 |
| C13 | 14.45% | 81.97% | 75.27% | 52.95% | 50 / 296 / 11 |
| C20 | 7.07% | 88.52% | 73.15% | 54.74% | 54 / 710 / 7 |
| C8 | 6.86% | 95.08% | 75.25% | 55.21% | 58 / 788 / 3 |
| original | 62.69% | 68.85% | 74.99% | 55.28% | 42 / 25 / 19 |

C8 shared mean logit drift was +4.253; C11 calibration freezing retained +3.011 and C13 projection regularization retained +1.821. C20 retained +4.120. In contrast, all 478,800 aligned development logits in 10D were bit-identical to original, but AP75 regressed 4.62 percentage points. Thus preserving classification alone did not repair localization, and projection adaptation can still inflate scores even with calibration frozen.

## Exact teacher compatibility

Verified **59 optimization images / 56 known source groups**, 212,400 patch queries and 1,486,800 cached logits. Each cache has `(3600,7)` logits and `(3600,4)` square-normalized cxcywh boxes. All source/receipt/archive hashes, finite FP32 arrays and original model identity pass. These are FP16-forward values saved as FP32 arrays, not FP32 inference references.

Original checkpoint SHA-256: `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`. Nine original model/tokenizer/processor files and the historically pinned OWLv2 modeling/preprocessing source files are unchanged. Native image dimensions, source/creator groups, EXIF orientation, RGB conversion, 960-square bottom/right-pad preprocessing, original seven prompts/class order, no augmentation and row-major 60×60 patch layout are bound explicitly.

The original 24 C8 probes agree bit-for-bit with the C17 teacher and all 24 C11 original-start probes, despite the eight-tensor calibration-frozen scope. Another 35 teacher images are already local. Frozen shared image/text features and identical inputs support **patch-index correspondence**, not persistent object identity. Updated student boxes do not redefine query indices; both teacher and student boxes must remain eligible for the same evidence region.

Historical per-photo processed-pixel/token tensor hashes were not recorded. This phase reconstructed and saved their hashes using the pinned local processor, without model inference. Those are current reconstructions, not invented historical observations. Four original replay checks are proposed in the gradient smoke; all 59 original outputs must pass live numeric equality before any future training updates. A mismatch stops execution; there is no automatic cache refresh or relaxed tolerance.

No optimization-photo pre-head image/text feature cache is available. The 19-photo development activation cache must not be used for strength selection. Seven scalar logits per query cannot reconstruct the original feature vectors/projection Jacobian, so feature distillation and parameter-gradient reconstruction are unsupported. Logit-level distillation is the one implemented method. A conditional original-only 59-forward cache-repair proposal exists if replay fails; it is disabled and has not run.

## The one selective objective

`L = L_C11_corrected + lambda * mean((student_Chair_logit - teacher_Chair_logit)^2 over admitted query terms)`

The base is the unchanged C11/C8 corrected focal classification objective plus 5×L1 + 2×GIoU: same detached Hungarian assignment, original 182 positives, original 41 source-supported Chair-negative grants, alpha 0.25, gamma 2, positive-count normalization and accumulation convention. No BCE modification, source balancing, projection penalty, new labels, new sampling or temperature search is added. At lambda zero, CPU tests reproduce the base loss and positive/box gradients exactly.

The teacher is an immutable cache with detached copied tensors, no live teacher optimizer and no teacher gradients. The consistency term is computed in FP32 and averaged over admitted terms in each photo; an uncovered photo has differentiable zero, not 25,200 background negatives. Group losses are averaged by the actual four/three microbatches. Regions are not multiplied by source-group weighting.

Required mask conditions:

- Exact image/source-group/processor/tokenizer/prompt/class/precision/grid compatibility.
- A previously source-verified visible non-Chair region; low teacher confidence alone grants nothing.
- Full containment of both boxes in the same original-coordinate grant and frame.
- No overlap with positive furniture, ignored regions, protected seating, uncertain native furniture/parts or mixed evidence.
- Every matched positive query has priority over distillation for all seven outputs, even when the teacher disagrees with its label.
- Teacher Chair score at least 0.15 inside a trusted non-Chair grant is a conflict and is excluded; the old detection threshold is used as a conflict rule, not tuned here.
- Only the Chair column is distilled. Six other labels remain unknown, and a query is counted once.

Of 41 historical source grants, six overlap conservative furniture/part protections and are excluded from **distillation** without relabelling or declaring old evidence false. 35 grants remain; 16 have no valid original-query coverage. Actual original-query coverage is **278 terms / 15 photos / 15 source groups**. Initial teacher conflicts: 0. The five C21 additional regions remain inactive. Chair includes seating/bar stools; footstools, ottomans and step stools are not automatically Chairs or generic background.

The additional conservative exclusions belong only to the new consistency term. The historical C11 base loss/grants are preserved for an interpretable comparison; source negatives remain Chair-specific, not seven-category background. Native completeness flags and all original positive/ignored annotations remain unchanged.

**This mask cannot guarantee global confidence control.** A student can increase an unanchored Lamp logit without increasing this loss; a CPU test explicitly demonstrates that limitation. A shared projection may couple the anchored Chair gradients to other queries/classes, but the magnitude and direction must be measured. Unmatched predictions on partial annotations are not confirmed false positives.

## Student scope and comparison control

The actual original OWLv2 CPU instance matched all 418 persisted original tensors. The trainable scope is **eight tensors / 1,577,988 scalars**:

- `box_head.dense0.bias` — shape [768]
- `box_head.dense0.weight` — shape [768, 768]
- `box_head.dense1.bias` — shape [768]
- `box_head.dense1.weight` — shape [768, 768]
- `box_head.dense2.bias` — shape [4]
- `box_head.dense2.weight` — shape [4, 768]
- `class_head.dense0.bias` — shape [512]
- `class_head.dense0.weight` — shape [512, 768]

The four logit-shift/scale tensors, complete vision/text model, shared normalization/representations, objectness, buffers and unregistered box-bias tensor stay frozen. Exact frozen names, shapes and hashes are in `parameter-scope.json`. Projection tensors remain trainable to permit classification changes; box tensors retain the original localization objective. No claim is made that box adaptation will generalize after the 10D regression.

This is the same parameter scope as C11. **C11 is the closest control**: the added selective consistency term is the intended intervention. C8 trains calibration too, C13 adds L2-SP, C20 changes negative-gradient composition and 10D freezes projection; those are contextual baselines, not controlled distillation ablations.

## Strength: deliberately unset

CPU autograd used actual saved C11 epoch-5 **training-photo** logits/boxes on 24 photos. Unit-distillation loss median was 9.7467; positive loss median was 0.006137. Corresponding output-gradient L1 medians were 6.2098 and 0.013425. These are separate medians, not a median ratio or parameter-gradient measurements.

An unweighted penalty could dominate positive output gradients. It is not defensible to infer a projection-gradient ratio from output-gradient mass: normalization, image/text features and the current projection change the Jacobian and gradient directions. At identical initialization, MSE and its gradient are zero, so an initialization-only gradient check cannot select a useful fixed weight. **lambda remains null.** No coefficient search or development-panel tuning occurred.

## Prepared worker and exact next approval

The default worker and hard-deadline launcher reject execution before CUDA/model construction/directory creation. A separate exact proposal/input approval is required. Training additionally requires actual passing gradient-smoke evidence, an explicitly approved finite positive fixed coefficient and a strength justification. Smoke does not automatically pick a coefficient.

**Next authorization requested: one zero-update diagnostic**, not training:

- Four original FP16 replay forwards on existing optimization photos.
- Load the existing immutable C11 epoch-5 heads in memory only as a diagnostic reference, then 24 forwards on the exact historical source-negative optimization-photo list. Actual future student training still initializes from original.
- Maximum 28 forwards, 48 projection-gradient VJPs, 180 seconds, 6.5 GiB allocated VRAM and 128 MiB new output under `data/open-images/experiments/phase10e-gradient-smoke-v1/`.
- Zero optimizer construction/steps or learned weight updates; no validation/reserved scoring, downloads, cache repair or automatic retry.
- Compare the same positive and unit-distillation losses on actual outputs: projection norms/directions, output mass, source-group variability, teacher conflicts, coverage and potential Chair suppression. Verify both original and existing C11 cached outputs without extra forwards.

The reference C11 checkpoint SHA-256 is `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f`. Loading that saved model for diagnostics is not the initialization of a new training arm. The fixed original base and both checkpoint files must remain unchanged.

One later **disabled training configuration** is implemented: original initialization, 59 photos / 56 known groups / 182 boxes, LR 1e-5, seed 6106, AdamW (0.9,0.999), eps 1e-8, weight decay 0.01, batch 1 / accumulation 4 (last 3), global clip 1, FP16 autocast/FP32 loss/GradScaler matching the teacher cache. Maximum five epochs / 75 updates / 600 GPU seconds / 6.5 GiB allocated VRAM / 2 GiB output; maximum 373 forwards = 59 initial original-cache checks + 295 training + 19 final FP32 diagnostics. There is no intermediate epoch selection or automatic follow-on run.

The worker records positive/distillation projection gradients, loss normalization, seven-category drift, teacher agreement/conflicts, selected negative scores, Chair-positive score reductions, calibration/frozen hashes and updated projection/checkpoint state. It saves atomic checkpoints/RNG/cursors and prediction files before a separate Torch-free CPU evaluator. Nonfinite values, skipped AMP steps, correspondence errors, unauthorized gradients, changed frozen state or any budget breach stop the attempt. CPU execution/import/gate tests have run; actual GPU behavior of this new worker remains untested.

Evaluation stays on the exposed 19-photo panel at confidence 0.15 with identical native completeness scopes, IoU/AP50/AP75, NMS and cap. Report fixed/matched-recall precision, TP/FP/FN, Chair errors, Sofa/Bed localization, all-seven logit drift and gradient influence. The immediate requirement is precision at least 60% **and recall above original 68.85%**, with no localization regression; existing original-level results do not establish improvement. Neither the preflight nor zero-update smoke can meet a detection milestone.

## Independent evaluation

Actual qualifying fully annotated independent rooms: **zero**, and zero complete independent instances for each of the seven categories. Existing 156 local readiness photos, three exposed room photos and historical native metadata are useful development/provenance resources, not automatically eligible independent acceptance. Reserved/frozen memberships remain locked and unscored.

The practical plan in `independent-evaluation-plan.json` proposes 140 newly authorized real rooms from 140 source/capture groups, at least 50 complete instances per category, prioritizing Cabinet/Lamp/Bed/Desk. Prefer owner-authorized captures; record photograph-specific permission/license/source/attribution before intake. Exclude historical IDs, hashes, rooms, creator groups and near-duplicates. A blind all-seven-category annotation/completeness pass, occlusion/truncation/ignore policy and second review must find unlabelled objects independently of model predictions. Double-review at least 20% of boxes. Freeze dataset/protocol hashes before scoring; use a separate calibration set for any threshold selection.

Estimated photo transfer is 28–420 MB, with a proposed 1 GiB image cap; full independent annotation is estimated at 23–47 person-hours. These are projections, not acquired or verified counts. No exact new source manifest, permission grants, download, installation or annotation work is approved here. The joint 80% precision / 90% recall acceptance target remains untested.

## CPU verification and preservation

84 CPU tests passed across the new loss/cache/scope/gates and existing scope/base-loss/source-protection suites. They establish contracts and gradient routing on cached outputs or synthetic modules, not detection improvement. Historical backup-inventory files, C8/C11/C13/C17–C20/C21/10D seals/checkpoints and original OWLv2/SigLIP artifacts passed the read-only checks recorded in `preservation-v2.json`.

No source annotations, human-review flags, checkpoints, production defaults, environment packages, historical predictions or Mac simulation files changed. All current artifacts are under fresh E1 preparation/report roots. Future training and gradient-smoke directories remain absent. Original source photos and large caches/weights stay local; code/configurations/tests/compact reports are committed on the E1 experimental branch.

The exact approval-ready scopes are `gradient-smoke-disabled.json` and `gpu-training-disabled.json`. **Approve only the 28-forward zero-update gradient smoke next if desired. Training needs a later separate coefficient/configuration approval.**
