# Phase 10E.14 — Synthetic end-to-end reliability verification

Completed 2026-11-24. Parent: `743c39180868162cb009740a95d8e528b8513537`.
Branch: `experiment/phase10e14-synthetic-pipeline`.

**Decision: the tested CPU detection/data workflow is sufficiently reliable to
return to separately approved data acquisition and bounded model experiments.**
No release-blocking defect remains in that tested scope. This is not a claim that
a real detector, GPU training, every historical/UI adapter or photograph-rights
verification was exercised. No next experiment or acquisition was started.

## 1. Active data flow and adoption boundary

The existing CLI detection path is `cli.main -> pipeline.run_experiment`:

1. `data.load_manifest` validates identities, rights declarations, boxes and any
   declared image hashes.
2. `data.load_image_with_digest` captures the actual compressed bytes, hashes that
   snapshot, then applies EXIF transpose and RGB conversion. Coordinates refer to
   that oriented image.
3. `Owlv2Detector` uses its pinned processor and ordered query texts/categories;
   processor-specific resize and tensor preparation happen inside the detector.
4. `Owlv2Detector.decode` clips candidates and calls the shared
   `postprocessing.filter_detections` with explicit `FilterConfig`.
5. The old runner writes crops and prediction records through shared atomic JSON.
   It refuses existing output directories; it has no general resume protocol.
6. `evaluation.evaluate` validates annotations/predictions and uses the shared
   confidence-ordered, one-to-one same-category matcher and ignored-region rule.

The new **opt-in** library entry point `pipeline.run_cached_detection` adds v2
cache reuse and resumable publication for detection-only runs. Its implementation
is `cached_pipeline.py`, using the existing E13 `cache.py`, `persistence.py`,
image loader, postprocessor and evaluator. It is callable from active library
code; it does not silently change the CLI default or historical phase scripts.
Retrieval and segmentation are outside this new adapter.

The adapter verifies caller-enumerated local model/processor file identities,
model/query metadata, image bytes, orientation, source groups and partitions. It
stores pre-NMS candidates at the detector's declared acquisition score floor in
E13 v2 envelopes. It never claims these are all model queries if the underlying
detector already discarded lower scores. Available query IDs/logits are retained;
no missing real-model data is fabricated. Supplied logits must agree with the
ordered queries, selected category and score. Requesting postprocessing below
the acquisition floor is rejected.

Real model loading must still prove the live model uses its declared files and
has the expected state in its separately authorised GPU preflight. Callers must
enumerate all relevant weight/head/config/processor/tokenizer and implementation
dependencies. A declared identity is not an in-memory model attestation.

Existing Phase 10 model/training scripts and their source snapshots are preserved
historical paths. Legacy PhotoJobs and room-layout stage caches were not converted
or certified as v2. Fresh v2 roots reject unmarked legacy output directories and
schema-1/raw artifacts rather than treating filtered predictions as complete raw
inference. Nothing migrates or rewrites thousands of old caches.

## 2. Deterministic synthetic fixture and baseline

`tests/synthetic_pipeline_fixture.py` generates three distinct test-only PNGs,
including EXIF orientation 6, across three source groups and two explicitly
synthetic partitions. There are eight known positive boxes covering all seven
categories, an incompletely annotated image, protected seating rectangles,
ignored regions, duplicate boxes and below-threshold candidates.

The synthetic detector returns **17 fixed candidate records containing 119 class
logit values**, scores, boxes and per-image query IDs. It loads no model. All
images, dummy checkpoint/config files and run artifacts are generated inside
temporary directories, completely separate from actual datasets.

Before active code changed, `baseline14.py` ran the existing runner and evaluator
and recorded `baseline-before.json`. The new entry point reproduces its semantic
predictions, hashes, oriented dimensions, matches and metrics **exactly**. The
rotated image is correctly 80 x 120 instead of 120 x 80. Reloading a completed
run produces identical committed results and zero additional synthetic detector
calls.

Baseline synthetic matching gives 8 TP / 0 FP / 0 FN, with three unmatched
predictions ignored. These are deliberately constructed unit-test results, not
furniture-recognition accuracy. Editing one generated ground-truth box produces
7 TP / 1 FP / 1 FN; a subsequent synthetic NMS test produces 7 TP / 3 FP / 1 FN.
Both create distinct derived runs while total synthetic detector calls remain
three. No production threshold was selected or adopted. See
`synthetic-equivalence.json` and `synthetic-derived-results.json`.

## 3. Precise dependency/invalidation results

The following matrix is computed in `cache-invalidation-matrix.json` and checked
by the synthetic tests. Supervision here is a geometry/provenance descriptor,
**not** a generated model-query training-target tensor.

| Changed dependency | Raw candidates | Postprocessed predictions | Evaluation | Supervision descriptor |
|---|---|---|---|---|
| Image bytes | Recompute | Recompute | Recompute | Recompute |
| Orientation/preprocessing | Recompute or reject unsupported transform | Recompute | Recompute | Recompute |
| Checkpoint or processor identity | Recompute | Recompute | Recompute | Reuse geometry-only descriptor |
| Prompt or category ordering | Recompute | Recompute | Recompute | Reuse geometry-only descriptor |
| Taxonomy description only | Reuse | Reuse | Recompute | Recompute |
| Ground-truth boxes | Reuse | Reuse | Recompute | Recompute |
| Completeness | Reuse | Reuse | Recompute | Recompute |
| Ignored/protected regions | Reuse | Reuse | Recompute | Recompute |
| Source-group assignment within same partition | Reuse | Reuse | Recompute | Recompute |
| NMS / final confidence policy, above captured floor | Reuse | Recompute | Recompute | Reuse |
| Matching IoU | Reuse | Reuse | Recompute | Reuse |
| Unsupported schema | Reject without implicit migration | Reject | Reject | Reject |
| Image/source/family partition reassignment | Reject within the v2 root | Reject | Reject | Reject |

A change to category meaning or actual text queries is an inference change;
changing only an annotation-taxonomy description is a context change. Raw keys do
not include NMS or the final confidence policy. They do include the acquisition
floor because the underlying detector may discard lower scores. Filter/evaluator
implementation identities and matching settings are bound to derived keys.

For partially annotated images, the current matcher cannot represent independent
per-category completeness masks. The new adapter therefore conservatively ignores
all unmatched predictions for an image with any partial/unknown category. Known
positives match first. This explicit policy is recorded in derived inputs and
keys; it is not an improvement claim and cannot justify measuring precision on
unlabelled background. Original annotations and historical scoring definitions
are unchanged. Training-query target construction remains in its own versioned
trainer; this descriptor does not activate new negative labels.

## 4. Interruption, resume and multi-file publication

Each raw cache is validated and atomically published with the existing v2 utility.
Each derived run has five immutable artifacts: `input.json`, `predictions.json`,
`postprocessed.json`, `supervision.json`, and `evaluation.json`. Only after all
five exist does `completed.json` publish their sizes/hashes and the request key.
`load_completed_run` validates that commit before returning results. Directly
opening an intermediate file through a legacy reader is not this v2 protocol.

Actual spawned test processes terminate inside a partial write using
`os._exit(23)` at each of these stages:

- Initial adapter/root manifest.
- Raw prediction-cache publication.
- Derived input-manifest publication.
- Evaluation-result publication after earlier files are complete.
- Final completion-manifest publication.

Readers reject absent/incomplete commits. Resume reuses only validated compatible
raw files and republishes missing derived artifacts. Completed sibling files
retain their hashes. Orphan temporaries remain as evidence and are never accepted
as results. An additional process exits after the first photograph's cache has
committed: resume computes only the other two photographs, and the final result
still contains three images and eight positives, not duplicated counts.

Corrupt raw payloads, incompatible schemas, damaged committed evaluation files,
stale completion requests and source-manifest changes during processing all fail
explicitly. They are not silently repaired, overwritten or retried with a new
model. Successful cache files from an interrupted run remain usable.

This is a completed-manifest protocol, not a filesystem-wide multi-file atomic
transaction. Readers must use it. Existing single-file fsync failure guarantees
and `AtomicCommitError(committed=True)` behavior remain as documented in E13.

## 5. Concurrent writers and verified fixes

The new shared `artifact_lock` uses POSIX/WSL advisory process locks, with an
explicit contention timeout. Process death releases ownership; stable lock files
are retained to prevent different writers locking different inodes at one path.
Native Windows has no unsafe implicit fallback. Atomic no-clobber publication is
still the final write safeguard.

Observed tests:

- Two simultaneous identical requests commit one run and invoke the synthetic
  detector exactly once per image: three calls, not six.
- Different image identities can commit separate runs concurrently in one root.
- Different immutable JSON identities can publish concurrently; conflicting
  values at one identity yield one winner and one explicit conflict. Scheduling
  does not select a preferred writer, but conflicting bytes are never merged.
- Readers during an active partial write cannot read that temporary as a valid
  raw cache or completed run.
- Interrupted writers release ownership and permit recovery; timeouts are
  explicit rather than triggering duplicate work or a hidden fallback.

Regression-first fixes during adapter development are preserved in the logs:

| Verified problem | Minimal correction |
|---|---|
| A valid manifest omitting optional `ignore_regions` raised `KeyError` | Normalize an empty list before combining protected regions |
| Group checks within one request allowed a later run to reassign a previously cached image/group across partitions | Immutable image-ID, content-hash, source-group and known-family partition bindings under a shared root lock |
| Caller mutation of `FilterConfig` or taxonomy during a callback changed predictions without matching the original request snapshot | Copy spec and reconstruct an owned policy before invoking caller code |
| A crash during the very first root-marker write left an orphan that was misidentified as a legacy directory | Permit only recognized regular root-marker temporaries during initialization; preserve them and continue rejecting real unmarked legacy data |

These are problems captured in the new adapter's regression cases, not claims
that historical metrics were secretly wrong. No old architecture, classification
loss or evaluation threshold was changed. Native image rights and incomplete
annotations were never inferred from synthetic successes.

Partition bindings cover known relationships supplied to the same root. They do
not discover unknown physical identities or replace the authoritative historical
dataset/frozen-holdout registry across different roots. A crash can leave a
restrictive prefix of bindings, never an automatically permissive reassignment.

## 6. Final CPU verification

| Check | Result |
|---|---|
| New end-to-end, invalidation, interruption and process-concurrency tests | **34 passed** |
| Broader available CPU suite | **458 run: 425 passed, 33 explicit skips, zero errors/failures** |
| Two real TinyTransformers forward tests | Excluded, outside authorization |
| Native PDFKit/Swift/PyPDF2 integration | Remains explicitly unavailable on WSL |
| Pre-change semantic baseline | Exact match |
| Legacy cache formats and existing E13 tests | Preserved; broader suite passes |

Authoritative transcripts are `CPU-targeted-final2.*` and `CPU-broad-final2.*`.
Earlier `final` records are retained as preliminary evidence before the final
root-manifest interruption test; they are not the final totals. No blanket skip
or exception handling hides a failing source test. No real model forward, GPU
allocation, optimizer update, download or package installation occurred.

## 7. Historical preservation and next action

Before/after SHA-256 inventories cover 29,268 / 29,270 files. The only changes to
already inventoried code are the additive persistence lock and opt-in pipeline
entry point; the cached adapter and its test file are new. The generated fixture
was created before baseline capture and included in the before-inventory.
**29,266 original inventoried files are unchanged**, including all **28,342 data,
model and historical-report files**. No historical artifact was deleted. Phase
tools/reports and Git ignore rules are separately sealed. Private inventories
remain local and ignored.

Prior E13 code/report seals match the parent Git blobs. Its live-source seal
checker belongs at its recorded commit because this phase deliberately extends
`pipeline.py`, `persistence.py` and `.gitignore`; no old seal was rebound. Original
and adapted OWLv2, SigLIP 2, source annotations, frozen partitions and historical
predictions remain unchanged. The Mac project was not accessed or modified.
`main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

See `preservation-final.json`, `historical-source-replay-final.json`,
`summary-final.json`, and the [verification/API instructions](../../tools/phase10e14-v1/README.md).

**No critical CPU reliability blocker remains for the tested opt-in workflow.**
Legacy UI cache adoption, native PDF support, multi-host locking and live-model
state attestation are explicit boundaries, not reasons to repeat general cleanup
before acquiring data. Do not silently route new experiments through an
unvalidated legacy cache. Use the versioned adapter or the experiment's preserved
preflight, known-group registry and completion checks as appropriate.

**Recommended next action:** return to a narrowly scoped, commercially compatible
real-room data acquisition proposal targeting the known furniture coverage gaps,
with exact source IDs, rights evidence and independent development grouping for
separate approval. Retain a bounded GPU smoke test as the prerequisite for any
later authorised training run. This phase starts neither task automatically.
