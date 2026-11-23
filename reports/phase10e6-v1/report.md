# Phase 10E.6 — Saved-query hard-negative audit

Prepared November 22, 2026. Parent commit `11bc9fad9c0197e679f8b5eedf6b6ec2dca7e7bb`;
branch `experiment/phase10e6-hard-negative-expansion`. This phase performed CPU
analysis and inspection of existing optimization photographs only. No downloads,
detector inference, backward calls, optimizer updates or production changes ran.

**Do not start another training cycle. This audit found zero new trustworthy
high-confidence hard negatives.** It created a source-bound candidate dataset,
documented why candidates remain inactive, and prepared separate missing-cache,
conditional rights-audit and independent-evaluation requests.

## 1. E5 evaluation is distinct from C11

The E5 worker evaluated its live original-initialized student after 75 updates;
it did not reload C11 or another checkpoint for evaluation. The saved epoch-5
trainable-state fingerprint reconstructs from the persisted optimizer parameter
order and weights and equals the live post-evaluation fingerprint. Frozen state
was unchanged, all 19 prediction receipts passed source/file checks, and each
prediction file postdates the final checkpoint. This is saved-artifact verification,
not a new inference run or a claim based solely on matching metrics.

| Artifact | SHA-256 |
|---|---|
| Original OWLv2 `models/owlv2-original-baseline-20261103/model.safetensors` | `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` |
| E5 `data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt` | `416620a947a96579bdd20c85e7adeb8b60f65c8983e73d3b3943a58ceabef4f0` |
| C11 `data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt` | `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f` |

The 19 development prediction files and arrays differ in every photograph.
Across 68,400 raw queries, E5 minus C11 logit RMS is 0.000751634,
maximum absolute difference 0.00399780, mean difference -0.000625282,
with 28 argmax changes. Maximum square-model box-coordinate difference is
0.0000336766. This last value is in the raw padded-square coordinate system;
the prior E5 report's 0.0000216961 used its transformed coordinate comparison.
Projection-weight L2 difference is 0.0000734602 and bias difference 0.00000159033.

Saved, unchanged 19-photo protocol metrics at confidence 0.15:

| Model | Precision | Recall | AP50 | AP75 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Original | 62.6866% | 68.8525% | 74.9881% | 55.2776% | 42 / 25 / 19 |
| C11 | 8.7580% | 90.1639% | 76.7608% | 60.5837% | 55 / 573 / 6 |
| E5 | 8.7580% | 90.1639% | 76.7598% | 60.5837% | 55 / 573 / 6 |

AP50 is **not exactly equal**: E5 0.7675977768894984 versus C11
0.7676075391876376; both round to 76.76%. AP75 and fixed-threshold counts
match. The saved E5 telemetry records 73 nonzero KD VJPs, all conflicting with
positive-gradient directions. Epoch-5 median weighted KD/positive gradient norm
ratio was 0.003647 and maximum 0.445817; weighted KD loss was about 0.051% of
classification loss. Thus the effect is numerically measurable and behaviorally
negligible on this panel. Numerical backward differences prevent attributing all
parameter differences exclusively to distillation. No evaluation bug was found.

These exposed, partly annotated development results are not independent
seven-category accuracy. The 19 photographs were used only for this requested
artifact comparison and never for candidate mining or new training evidence.

## 2. Saved optimization-query coverage

The index verifies **617 source-bound raw cache files**, collapsing 190 exact
source-image/logit/box aliases into **427 distinct snapshots** while retaining all
state, receipt and hash lineages. The initial and expanded indexes are retained;
the final 96 added epoch-zero aliases added no new numerical snapshots.
All 59 optimization images have original teacher arrays. Most longitudinal
trained-state series cover the same 24 images; the final inventory has 25 images
with a non-original-state alias, including additional gradient diagnostics.
It is not a complete endpoint census of any latest model across all 59 images.

All 19 otherwise unindexed experiment NPZ files resolve exactly to existing
internal-validation recovery IDs. Their bodies were not opened for mining.
**E5 has zero complete optimization-image raw prediction caches.** Its 295
training presentations contain scalar/coverage diagnostics, which cannot replace
query boxes and all-category logits. Its 19 development arrays are excluded.

Counts below span saved states and raw queries, including real furniture,
suppressed queries and fragments. They are **not confirmed false positives,
independent objects or newly eligible training examples**.

| Target | Raw score >=0.50 | Raw score >=0.80 | Argmax + NMS >=0.50 | Argmax + NMS >=0.80 | Photos / known source groups with raw >=0.50 |
|---|---:|---:|---:|---:|---:|
| Chair | 7,807 | 937 | 4,281 | 702 | 24 / 24 |
| Sofa | 2,459 | 283 | 1,122 | 241 | 29 / 29 |
| Bed | 838 | 85 | 335 | 70 | 31 / 29 |

Chair contributes the most observations. Among its 7,807 observations,
6,916 have original-teacher confidence below 0.15 at the same query index;
7,302 overlap protected furniture/ignored regions. Same patch index is a
comparison anchor, not a guarantee that changed box heads describe the same
object. Context, exact coordinates, original/student scores, native annotation
keys, protected-overlap counts and existing-region relationships are preserved
for every observation.

There are 2,934 Chair, 834 Sofa and 386 Bed raw observations with IoU >=0.5
to another reported same-category query. These are spatial-redundancy flags,
not proof of duplicate physical objects or classification errors. Native source
bbox context and low teacher confidence cannot establish a negative label.

Of the high-score Chair observations, 886 overlap an existing one of the 41
source-supported negative rectangles, but **zero are fully contained** in such a
rectangle. Overlap must not be inflated into eligible full-query supervision.
The five prior C21 regions remain inactive. All uncertain image areas remain
ignored; no whole-image absence labels were created.

## 3. Actual visual inspection and candidate tiers

Three contact sheets show 24 targeted existing optimization crops, source
photographs, original positive/ignored boxes and actual query coordinates.
The assistant inspected all three sheets; these are automatic/assistant visual
assessments, **not fabricated human review or new source annotations**.

Representative findings:

- V08: a real-looking framed bar stool/chair despite native Bed-bbox context.
- V10/V11/V13/V14: actual chair frames, mixed chair/table crops or a clear
  molded chair. Low whole-object IoU does not make these negative examples.
- V17/V18: people plus upholstered seating/booth; a person bbox cannot certify
  absence of furniture behind that person.
- V20: an upholstered armchair predicted Sofa; native Sofa context requires a
  taxonomy decision, not automatic negative supervision.
- V22–V24: bed/headboard/bedding fragments and mixed person/bed regions.
- Remaining plant, screen-like, appliance-boundary and small-object crops have
  protected overlaps, incomplete visible-mask support or unresolved semantics.

No query passed the combined conservative requirements: original-image bounds,
at least 32 pixels per side, full visible source-polygon containment with a
two-pixel boundary margin, and no overlap with any protected furniture,
ambiguous part or ignored source region. No visually assessed crop overrides
these safeguards. This is a finding under that policy, not proof that no useful
nonfurniture region exists anywhere in the photographs.

| Tier | Distinct image/target/query-footprint candidates | Activation |
|---|---:|---|
| A: trusted new high-confidence negatives | **0** | None |
| B: automatic, source-bbox-supported but unverified | **183** | Inactive |
| C: uncertain, protected, known furniture, fragmented or otherwise rejected | **9,781** | Inactive |

The 183 B candidates span nine photographs/nine known source groups:
174 Chair and nine Sofa, zero Bed. Native bbox contexts are 98 person/body,
49 appliance/screen, 33 plant/decoration and three other native objects.
Eleven B footprints have a reported Chair score >=0.80; these also remain
unverified and inactive.
None has certified whole-query absence evidence. Tier C is not a collection of
confirmed mistakes. Repeated snapshots sharing an exact image/target/pixel
footprint are consolidated; different nearby footprints are not silently merged
into independent physical objects. Full observation lineage remains available.

Private dataset root: `data/open-images/preparation/phase10e6-v1/`.
`dataset-index-v1.json` links hashed components:

- `training-candidate-manifest-v1.json`: A/B/C IDs, unchanged original context,
  labels/taxonomy/partition safeguards and zero activation.
- `candidate-evidence-v1.json` plus `high-confidence-observations.jsonl`:
  exact coordinates, scores, source support, state and overlap evidence.
- `source-lineage-v1.json`: photo hashes, source groups, native annotation
  checksums, licence/creator evidence records and original verification times.
- `inactive-local-candidates-v1.json` and
  `native-furniture-candidates-v1.json`: additional local audit candidates only.

Contact sheets: `contact-sheets/queries-1.png`, `queries-2.png`, `queries-3.png`.
`inspection-samples.json` binds every sample to source, query and page hashes;
`visual-observations.json` records findings with human-review flags false.

## 4. Existing local annotation expansion

The existing rights-evidenced training pool has 101 photographs: 59 active
optimization photographs and 42 inactive/unassigned photographs from 39 known
source groups. Rights-evidence files and image bytes were rehashed offline for
all 101; no new online rights verification or ownership claim was made.
All local official annotation-source files were also rehashed.

There are 2,272 native records: 1,386 on the active 59 and 886 on the inactive
42. The active pool's older inventory contained 1,005 records. All **381 omitted
records** were revisited, including 313 mask-bearing records and 17 native
ignored records. Context counts, preserving exact native labels, are:

| Context heuristic | Omitted records |
|---|---:|
| Furniture fragments/ambiguous parts | 126 |
| Other furniture | 32 |
| Seating/ambiguous seating | 1 |
| Person/body | 16 |
| Plant/decoration | 12 |
| Appliance/screen | 11 |
| Other native objects | 183 |

These are annotation records, not independent instances: COCO and LVIS may
annotate the same physical object. Source masks are retained without interpreting
missing LVIS/COCO annotations as negatives. Part labels such as cushions,
headboards, lampshades, knobs and drawers remain protected. Whole Cabinet
versus fronts, Table versus Desk, and whole Lamp versus lighting parts remain
unresolved where native semantics do not establish the project target.

The active unchanged training context has 59 images, 56 known source groups
and 182 positive boxes: **114 Chair, 40 Sofa, 28 Bed**. Its sources are 36
Phase6D COCO, 13 legacy Open Images, five Phase6A Open Images, three Phase7
LVIS and two Phase8C LVIS images. Other native furniture labels are not silently
promoted into a seven-class training configuration. Ordinary chairs, seating
stools and bar stools are Chair; footstools, ottomans and step stools remain
separate/protected when uncertain.

The 42 inactive images show zero known source-group overlap with the current
internal-development panel. That alone does not prove complete independence:
capture relationships, native semantics and query coverage still need checks
before any partition assignment or activation. ABO catalogue weak labels are
not complete real-room detection negatives. No catalogue images, splits or
training artifacts were changed.

## 5. Missing inference and conditional licensed-data proposal

**Primary next approval request:** fill the latest E5 optimization-query cache,
without training. `inference-request-disabled.json` freezes all 59 current
optimization IDs, E5 SHA-256 above, original base, source context checksum and
existing pinned preprocessing/prompts. Proposed ceilings: **59 forwards,
180 seconds GPU runtime, 6.5 GiB allocated VRAM, 128 MiB new output**, zero
backward/optimizer calls, zero teacher forwards and zero downloads. Destination:
`data/open-images/experiments/phase10e6-training-query-cache-v1/`.
All 19 development and all reserved/acceptance photos are excluded. Save full
raw queries atomically before separate CPU analysis. This request is disabled;
it is not an executed or approved experiment. Cache completion would not itself
authorize training or convert a high score into a negative label.

If sufficient source-supported confusers remain unavailable after the local
audit/cache gap is resolved, the conditional route preserves the exact 64
previously proposed, unrequested COCO training source IDs in
`conditional-rights-candidates-v1.json`. Current verified new-photo count is zero.
Prior attempted and paused IDs are excluded, and the 296 paused requests are
not resumed. This route is **not photograph-acquisition-ready**.

Separate Stage A approval would cover source-specific rights/creator evidence
only: 32,000,000 bytes transfer, 128 MiB storage, one attempt per ID, at least
three seconds between requests, stop on access restriction/429 and honor
Retry-After. Stage B would require another exact rights-verified manifest and
explicit photo approval: at most 64 survivors, projected 9.6–25.6 MB images
(unverified estimate), 100,000,000-byte transfer cap, 3,145,728 bytes/image,
two attempts/image, 512 MiB storage. No automatic substitutions or requests ran.

Required images are licensed real indoor scenes with high-confidence confusing
people/body parts, plants, screens/appliances, small household objects and
furniture boundaries, alongside whole-object furniture positives. Require
photo-specific CC0/CC BY evidence, attribution and source/capture isolation;
annotation licensing is separate. Native visible masks plus actual semantic
verification are needed for proposed negative regions. Missing annotations,
catalogue labels or licence IDs alone are insufficient.

## 6. Independent seven-category acceptance workflow

Actual verified acceptance photographs: **zero**; complete accepted instances
per category: **zero**. Exposed development material and reserved/frozen panels
were not promoted, scored or mined. Neither candidate tier B nor an assistant
visual observation is an independent acceptance label.

Proposed first collection batch is 20 new, independently authorized real-room
photos; this is a proposal, not an available count or a replacement for the
longer 140-photo / minimum 50-instances-per-class collection target. Select
source/capture groups without viewing detector outputs, obtain photo-specific
rights/credits, record hashes and group evidence, and quarantine any historical
source/creator/capture or duplicate relationship. Actual collection needs approval.

A blind annotator must explicitly review **all seven categories**, including
zero/absent categories, whole-object extent, occlusion/ignore flags and the
Chair/stool taxonomy. Resolve Cabinet/front, Table/Desk and Lamp/fixture
ambiguities; incomplete photos remain ineligible. A second reviewer checks
completeness and at least 20% of boxes. Freeze photo, annotation and scoring
hashes before a separately authorized single comparison, with fixed confidence,
PR/AP50/AP75, matched recall, duplication/fragmentation and source-group
uncertainty. Acceptance photos must never generate training masks or tune models.

`independent-review-template-v2.json` contains **zero submitted records**, no
prefilled human decisions, and null per-category unreviewed fields.
`acceptanceE6.py` validates explicit human evidence, source isolation, individual
CC0/CC BY policy, hashes, dimensions/orientation, normalized boxes, explicit
completeness counts and actual source files. It cannot verify ownership from a
URL alone; independently reviewed rights evidence remains required.

Future submitted review validation, CPU only:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e6-v1/acceptanceE6.py /path/to/submission.json
```

## 7. Verification, preservation and decision

The final CPU suites cover new identity/coordinate/evidence/intake safeguards,
existing E5 numerics, E1 masks/cache handling, C11 parameter scope and detection
metrics. `CPU-tests-v3.json` records **83 passing tests** (26 new E6 tests plus
57 relevant existing tests), no skips and no GPU calls. All earlier test
transcripts remain available. `dataset-integrity.json` rechecks cache/source
membership, hashes, dataset components, tier isolation and zero activation.

Historical verification passed for 9,973 earlier data/model files and the
subsequent C8/C11/C13/C17–C21/D/E1–E5 private/public seals. Original OWLv2,
SigLIP 2, adapted models, source images and partitions remain unchanged.
`main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.
Only the new experimental tools, small reports and ignore allowlist are committed;
images, private candidate manifests, caches and weights remain outside Git.
The separate Mac application was not accessed or changed.

CPU reruns should write a fresh versioned destination; immutable emitters reject
existing outputs. Read-only checks can be repeated without altering artifacts:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e6-v1/validateE6.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e6-v1/verifyE6.py check
```

**One concrete training recommendation: do not train until genuinely new,
source-supported high-confidence negative queries improve coverage across
multiple independent source groups.** The next action is the separately approved
59-photo E5 inference-only cache completion and subsequent CPU evidence audit.
Then reassess local unassigned images before conditional rights acquisition.
Do not reuse the ineffective E5 coefficient, launch another focal/BCE balance
variant, or propose a loss ablation on the same unchanged evidence. Preserve
pretrained classification behavior as a design constraint for any later,
separately approved experiment; this audit supplies no basis to promise the
60% precision / above-68.85% recall target, much less independent 80%/90%.
