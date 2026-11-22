# Phase 10C.21 — Hard-negative and pretrained-preservation preparation

Completed CPU-only preparation on November 22, 2026, on
`experiment/phase10c21-hard-negatives-pretrained-preservation`, from C20 commit
`b0b74ddea184cc9065b37a6a70fe4d03210dbefa`. No real-model forwards, training,
downloads, deployment or Phase 10D activation occurred. Focal/BCE balancing was
not extended. Source annotations, historical predictions and weights remain
immutable. The Mac application was not accessed.

**Recommendation:** request one original-initialized, box-head-only localization
experiment with the entire pretrained classifier frozen. Keep teacher
distillation separate. The new negative evidence remains too narrow to justify
another loss-balancing experiment or a claim about independent room accuracy.

## What C20 established

These are saved, identically scored results from the exposed 19-photo,
61-positive, three-category internal panel. They are not independent acceptance
results. Precision and recall below use the existing confidence threshold 0.15.

| Model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN |
|---|---:|---:|---:|---:|---:|
| Original OWLv2 | 62.69% | 68.85% | 74.99% | 55.28% | 42 / 25 / 19 |
| C8 corrected Arm B, update 75 | 6.86% | 95.08% | 75.25% | 55.21% | 58 / 788 / 3 |
| C20 recovered, update 75 | 7.07% | 88.52% | 73.15% | 54.74% | 54 / 710 / 7 |

At original recall 68.85%, the saved PR envelopes give 80.77% precision for C8
and 75.00% for C20; the original envelope gives 68.85%. These are descriptive
curve comparisons, not adopted thresholds. C20 did not improve on C8 at matched
recall; its small fixed-threshold precision gain cost four true positives.
Chair scored FPs decreased from 636 to 566, with three additional Chair misses;
Sofa gained one miss and Bed recall stayed unchanged. On commonly matched Chair
ground truths, mean IoU fell by about 0.0166.

The source-gradient guard activated 13 times without a genuine bound violation.
It constrained gradients as designed, but shared mean logit drift remained
large: +4.253 in C8 versus +4.120 in C20 relative to original. In the fixed
training-only C20 epoch-5 probe, only 3 of 288 trusted-negative query terms
scored at least 0.15 and none reached 0.50; median Chair score was about 0.0068.
The narrow evidence was largely easy for the trained model while high-confidence
unmatched predictions persisted elsewhere.

This supports a coverage/discrimination problem, not another arbitrary balancing
coefficient. It does **not** prove that missing hard negatives uniquely caused
the regression. Positive-label bias, shared classification adaptation,
localization and incomplete annotations remain relevant. Unmatched predictions
on incompletely annotated regions remain unknown, not confirmed errors.

## Local annotation and data audit

| Inventory | Photographs | Status |
|---|---:|---|
| Exact optimization pool | 59 | 56 known source groups; only pool mined |
| Internal development | 19 | Excluded from negative mining and gradients |
| Other locally available historical training photos | 42 | Not assigned or activated |
| Prospective historical training total | 120 | 59 + 19 + 42 |
| Native annotation join | 101 | 59 optimization + 42 inactive |

All locally available relevant COCO/LVIS JSON/ZIP and Open Images box CSVs were
streamed, hash-checked and joined to exact source IDs and dimensions. The join
contains 2,272 native annotation records across 101 photos; 1,386 are on the 59
optimization photos. The older category-name-filtered inventory contained only
1,005 optimization records. The **381 omitted records** include body parts,
headboards, cushions, cabinetry, lampshades, a beanbag and a piano. No native
geometry was rejected. COCO/LVIS records can describe the same object; these are
annotation records, not 2,272 independent objects.

Retained optimization positives remain **114 Chair, 40 Sofa and 28 Bed boxes**,
182 total. Table, Desk, Cabinet and Lamp have no newly activated positive
targets. Native dataset split names do not change project assignments: an LVIS
validation annotation attached to a historical project-training photo is retained
as provenance, not a reason to move that photo. No evaluation photograph was
added to optimization.

Chair includes ordinary chairs, seating stools and bar stools. Footstools,
ottomans, step stools, beanbags, furniture parts, mixed boxes and uncertain
seating remain protected or unresolved rather than automatically becoming Chair
negatives. Newly discovered seating was not silently relabelled or activated.
Other furniture categories were protected from generic-background targets.
COCO positive annotations do not establish all-seven-category completeness;
LVIS native negative/not-exhaustive flags are preserved without extending them
to the broader Chair-plus-stool taxonomy.

The source-image hashes and historical photograph-specific rights evidence were
replayed locally. This is not fresh online verification of ownership or releases.
Known source/creator grouping is conservative evidence, not proof that every
physical room or product identity is independent.

## New hard-negative evidence

Mining used saved actual query outputs only: original C17 caches on 59 training
photos and four trained-state caches on 24 training photos each, 155 image-state
records total. No development predictions were used for training-negative
selection. A saved Chair score at least 0.15 was a mining trigger, not a label.

Each candidate needed full query-box containment inside a native visible-instance
polygon with a two-pixel margin, exact original-image registration, preserved
source/annotation hashes, all positive/ignored/native-furniture protections, and
actual query-local visual inspection. Bbox intersection alone was insufficient.
Two bounded contact sheets were inspected; no rectangle drawing was required.

| Tier / evidence | Count | Meaning |
|---|---:|---|
| A, additional non-overlapping source-supported regions | 5 | 4 photos / 4 source groups; Chair-only eligibility |
| C, inspected alternatives or uncertainty | 13 | 12 overlapping/duplicate alternatives and 1 dark TV reflection |
| B, contextual annotation associations | 373 | Automatic proposals, no Chair-negative grant |
| C, protected/uncertain contextual associations | 285 | Furniture, parts, conflicts or unresolved overlaps |

The 658 contextual records are a separate association inventory and overlap the
candidate search; do not sum them as independent negatives or false positives.

| Native visible object for new A region | Regions | Photos | Source groups |
|---|---:|---:|---:|
| Person: hair/neck/clothing pixels | 2 | 1 | 1 |
| Upright piano upper panel | 1 | 1 | 1 |
| Television display showing a face | 1 | 1 | 1 |
| Handbag surface/hardware | 1 | 1 | 1 |

All five are assistant-inspected, native-source-supported evidence, **not human
confirmations**. Their saved Chair scores range from 0.151 to 0.440; none covers
a score at least 0.50. They support only the exact visible rectangles, not the
surrounding photo or another furniture category. They are spatial additions to
the old 41 regions, but remain on existing probe photographs: old plus new would
be 46 regions across the same 24 photos. No combined training mask was activated.
The five regions have seven distinct cached state/query observations, not seven
independent scenes. Native body/plant/decorative boxes elsewhere do not prove
Chair absence or resolve occlusion.

Private evidence and inspection plates:

- `data/open-images/preparation/phase10c21-v1/hard-negative-manifest-v1.json`
- `data/open-images/preparation/phase10c21-v1/query-local-visual-observations.json`
- `data/open-images/preparation/phase10c21-v1/contact-sheets/query-local-1.png`
- `data/open-images/preparation/phase10c21-v1/contact-sheets/query-local-2.png`

The remaining deficit is high-score, protected-overlap-free query evidence on
people/body parts, plants, architecture and furniture fragments. First qualify
the 42 already local inactive training photos for source semantics and group
assignment; do not automatically activate them. The present five regions cannot
establish broad hard-negative coverage.

## Pretrained-preserving alternatives

| Approach | Evidence and benefit | Limitation | Decision |
|---|---|---|---|
| A: original classifier, box-head-only adaptation | Six box tensors can learn positive localization while preserving original query classification | Cannot repair wrong original category logits; changed boxes/NMS can still change precision | Primary diagnostic proposal |
| B: frozen original-teacher distillation | Original outputs already cached for all 59 optimization photos | Teacher mistakes are not truth; masks/strength/precision introduce extra choices | Separate future alternative |

The formerly reported 24-photo teacher limit is obsolete. Verified C17 caches
cover all 59 photos: the original 24 are bit-identical to C8 original caches and
35 more are already local. These were FP16 forwards saved in FP32 arrays, **not
an FP32 inference baseline**. Their 60×60 query indices identify fixed patches,
not persistent objects. Teacher correspondence requires identical source bytes,
orientation, preprocessing, prompts, frozen backbone and numeric precision.
Low teacher confidence is not absence. Pre-head features/objectness and an
independently justified distillation mask/strength remain uncertified. No
distillation objective is combined with proposal A.

The previous box-only/Phase 10D preparation was reviewed, not activated. The new
adapter reuses its unchanged positive Hungarian assignment and box loss while
adding persistent original-classifier/frozen-state fingerprints, exact optimizer
checks and configuration-bound six-tensor checkpoints. CPU tests use original
head tensors in a synthetic module, plus the earlier real-head CPU fixtures;
they do not substitute for a future authorized detector smoke test.

## One disabled training proposal

Exact machine-readable configuration: `box-only-disabled.json`; execution
approval binding: `next-training-approval.json`. Both remain disabled.

- Start from `models/owlv2-original-baseline-20261103/model.safetensors`, SHA-256
  `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.
- Train only `box_head.dense0`, `dense1`, `dense2`, their weights and biases:
  six tensors, 1,184,260 scalars. Freeze classification projection, all four
  shift/scale parameters, vision/text backbone, objectness, all other parameters
  and buffers. Initialize neither classifier nor boxes from C20.
- Same 59 photos / 56 known groups / 182 retained positives. Unmatched,
  ignored, ambiguous and absent annotations do not become background targets.
  No new negatives or inactive LVIS positives are consumed by this box-only arm.
- Detached original-classification Hungarian matching, `5×L1 + 2×GIoU` only.
  Classification, objectness and negative-classification loss are zero.
- One fixed seed 6106; FP32; AdamW LR `1e-5`, betas `(0.9,0.999)`, epsilon
  `1e-8`, weight decay 0.01; batch one, accumulation four (last group three),
  global gradient clipping one. No new augmentation or threshold choice.
- Maximum five epochs / 75 updates, 600 GPU seconds including diagnostics,
  6.5 GiB allocated VRAM and 2 GiB new output. Maximum 322 forwards: four
  zero-update smoke checks + 295 training + four post-training fixed-query
  checks + 19 exposed-panel inference calls. Stop rather than extend a limit.
- Proposed fresh destination:
  `data/open-images/experiments/phase10c21-box-only-v1/`, currently absent.
  Runtime 120–400 seconds and peak 2–4 GiB are estimates, not measurements.
- Require unchanged frozen tensor/buffer hashes and original per-query FP32
  classification on four identical before/after training inputs. The old FP16
  teacher cache cannot certify FP32 equality. Stop on classification drift,
  unexpected gradients/optimizer entries, invalid boxes, nonfinite values,
  preservation failure or any resource limit.

Evaluate only once at the fixed endpoint against original on the existing
19-photo exposed panel, using unchanged scoring and confidence 0.15. No epoch,
hyperparameter or threshold selection from this panel. Predeclared gates:
precision at least 62.69%, recall at least 68.85%, AP75 gain at least two
percentage points, no category AP75 drop over two points, TP floors Sofa 11 /
Chair 24 / Bed 7, and matched-original-recall precision at least three points
above the original PR envelope. Report PR curves, AP50/AP75, scored and unknown
unmatched predictions, localization, duplicate/fragmented boxes, per-category
regressions, latency and VRAM. These diagnostic gates are not evidence of
80% precision / 90% recall on independent rooms.

An execution worker still needs to be wired to the tested adapter under separate
GPU approval; legacy twelve-head workers are incompatible. This phase provides
the tested scope/loss/checkpoint code and a precise proposal, not an executable
authorization to invoke old training scripts.

## Independent seven-class evaluation

**Actual eligible completely annotated independent photographs: zero.** Complete
independent instances are zero for each of Sofa, Chair, Table, Bed, Desk, Cabinet
and Lamp. None of the 156 readiness source rows establishes independent
all-seven completeness. The three Phase 4 room photographs are model-exposed
with provisional annotations. Existing reserved/frozen images are not
automatically promoted or scored.

The new intake validator and empty review template require actual image/rights
hashes, independent creator/source/capture evidence, all-seven completeness,
whole-object taxonomy, Chair-plus-seating-stools policy and independent human
evidence blind to model predictions. Historical source IDs, photo hashes,
catalogue groups, creator groups and frozen references are excluded by a
metadata-only isolation registry v2. Unknown relations require quarantine;
exact hashes alone cannot establish perceptual independence.

The proposed future collection remains 140 real rooms from 140 distinct known
source groups, at least 50 completely annotated instances per category. Freeze
source membership before examining model outputs; verify individual photo
rights; annotate occlusion/truncation and explicit ignore regions; independently
review completeness and double-annotate at least 20%; adjudicate blind to model
outputs. Freeze annotation/protocol hashes before one baseline versus
preselected-model comparison. Report all-seven metrics and 2,000-resample
source-group bootstrap uncertainty. Any threshold tuning needs a separate
calibration set. No new exact-source collection has been authorized or acquired.

## Additional data proposal, separately gated

The preferred immediate data step needs no download: qualify the 42 existing
inactive photos. In parallel, an exact deterministic **64-ID metadata shortlist**
is available at
`data/open-images/preparation/phase10c21-v1/additional-rights-candidates-v1.json`.
It excludes historical image/source IDs and previously attempted or paused
COCO/LVIS rights IDs, including the 296 paused requests. Native indoor/context
metadata ranks candidates; no exposed-development error mining was used.

**Currently individually rights-verified new candidates: zero.** COCO license-ID
4/CC BY 2.0 metadata is not photograph-specific clearance. The proposal is
therefore rights-first, not a claim that 64 licensed photos are available.

Separate stage-A approval would allow evidence retrieval for exactly these 64
IDs: 32,000,000 response-body bytes, 128 MiB storage, at most one evidence
attempt per ID, minimum three-second spacing, respect Retry-After and stop
affected requests on HTTP 429. No photo bodies, identity rotation, bypasses or
automatic substitutions. Verify exact photograph, creator, source-specific
license/version, attribution, evidence timestamps/hashes, source grouping and
HEAD sizes. A later photograph proposal would require its own approval and exact
verified membership/sizes; current estimate is 9.6–25.6 MB for up to 64 images,
conditional cap 100 MB transfer / 512 MiB storage. No requests were made here.

## Verification, preservation and recovery

`tests-v2.json` records **63 passing CPU tests**: 32 new preparation/scope/intake
tests, four original box-adapter tests and 27 source-geometry/evidence/mask tests.
They cover exact partition isolation, provenance, mask/positive protections,
uncertain/duplicate rejection, original-only initialization, six-parameter
optimizer scope, frozen gradients/buffers, checkpoint binding and atomic
write-once outputs. The synthetic CPU optimizer test does not update a saved
detector or constitute an authorized model-training run.

`preservation-v2.json` records rehashing of the historical backup inventory,
original/selected checkpoints, C8 private seal and C17–C20 code/public/private
seals. All source and evidence hashes passed. `artifact-seal-v2.json` records this
phase's private manifest and code hashes, with conservative and unique-inode
storage accounting; `SHA256SUMS-v2` binds the small public reports. The initial
report-relative seal snapshot is retained as superseded; v2 uses the same
repository-relative format as the historical seal verifier. Private
photographs/contact sheets/cache records remain local. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

Recovery is non-destructive: retain all generated receipts and immutable inputs;
do not rerun write-once generators into completed destinations. Recheck seals
and CPU tests using `tools/phase10c21-v1/README.md`. A future interrupted worker
must checkpoint counters, optimizer/cursor/RNG state, base/config hashes and
frozen fingerprints, enforce cumulative limits, and save predictions atomically
before separate CPU scoring. No automatic new arm or C20 continuation is
permitted. Production remains on its existing model, so no deployment rollback
is needed.

**Next model authorization requested:** one original-initialized box-head-only
smoke plus fixed 75-update experiment under `next-training-approval.json`, with
the limits above and one descriptive 19-photo endpoint comparison. No teacher
distillation, classifier adaptation, new downloads, Phase 10D activation or
independent/reserved scoring. The 64-ID rights audit is an optional separate
approval, not a prerequisite or bundled permission.
