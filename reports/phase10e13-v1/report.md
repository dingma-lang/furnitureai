# Phase 10E.13 — Core reliability and reproducibility hardening

Completed 2026-11-23 on `experiment/phase10e13-reliability-hardening`, based on
Phase 10E.12 commit `5e9c53d446c5659c2ef82bb17a91372bc88cda4d`.

This phase made four focused improvements: deterministic test fixtures, shared
atomic persistence, explicit versioned cache identity, and binding decoded images
to their recorded hashes. Geometry was reviewed and measured, then left unchanged.
There were no model forwards, GPU operations, optimizer updates, downloads,
installations, annotation edits or production deployment.

## 1. All 11 former integration errors accounted for

The original IDs come from the unchanged E12
`preexisting-fixture-errors.json`. Exact per-test causes and outcomes are in
[eleven-test-status.json](eleven-test-status.json).

| Former error(s) | Cause | Resolution and actual status |
|---|---|---|
| Five `PhotoProjectTests` | Fake-runner tests called real configuration loading, requiring missing `examples/furniture-prompts-v2.json` and local model inventory | Deterministic configuration at the existing test seam; all five pass |
| `OwnedInventoryTests.test_catalog_reported_fields_not_verified_or_length_invented_as_depth` | Assumed external `data/catalogs/abo-phase6d/catalog.json` existed | Explicit synthetic product metadata; pass; reported length never becomes invented depth |
| `ReplacementTests.test_candidate_selection_uses_saved_production_results` | Same absent external catalogue | Synthetic saved-candidate fixture through a scoped read seam; real action/ID validation exercised; pass |
| Two `RoomViewerTests` for partial presets and untrusted external paths | Assumed local `reports/phase5-public-room/quick-room.json` existed | Valid temporary synthetic room JSON and preset registry; real lookup/presentation exercised; both pass |
| `TransferTests.test_invalid_uploads` | Optional PyPDF2 import occurred before corrupt-PDF error handling; package absent | Keep invalid-image checks; inject a deliberately failing synthetic parser for the corrupt-PDF unit path; pass |
| `TransferTests.test_pdf_pages_locally_rendered` | Genuine native PDF integration needs PyPDF2, macOS PDFKit and Swift | Explicitly unavailable on this WSL environment; one narrow platform/dependency skip |

There were no incorrect path spellings or broken historical fixture generators.
The tests assumed machine-local data or dependencies. No historical dataset,
review decision or expected model output was invented to satisfy them. Fixtures
use conspicuous synthetic IDs and temporary directories. The PDF parser seam is
not a substitute for proving actual PDF parsing/rendering works. The production
optional-import error remains a documented limitation, with no dependency install
or separate Mac-project change.

Result: **10 now pass; one genuinely unavailable integration is explicitly
documented. Zero of these 11 remain runtime errors.**

## 2. Confirmed reliability problems and fixes

### Atomic persistence

Previous active JSON writes and incremental feature writes could expose truncated
targets after interruption. New `persistence.py` centralizes single-file
publication: create a unique sibling temporary, write, flush/fsync, validate, then
publish and fsync the parent directory. `data.write_json` and the active catalogue
writer delegate to it. Successful JSON formatting remains equivalent to each
caller's old formatting. Strict JSON rejects non-finite values before publication.

New immutable prediction/feature artifacts use atomic no-clobber publication.
Existing mutable writer APIs retain replacement semantics; nothing in this phase
rewrote an existing result. Symlink destinations are rejected, existing permission
bits are preserved, and new temporary files use private permissions.

Pre-publication failures leave an existing target unchanged. If publication
succeeds but directory fsync fails, `AtomicCommitError(committed=True)` makes that
distinction explicit. This is single-file consistency, not a multi-artifact
transaction, concurrent-writer lock or guarantee against every filesystem/hardware
failure. POSIX directory fsync was tested under WSL; native Windows uses its
documented platform fallback.

Tests inject partial writes, validation failures, file-fsync failures, replacement
failures, corrupt feature files, and post-commit directory-fsync failures. A real
child process exits with `os._exit(17)` while writing a temporary file: the old
target remains intact; the orphan is not accepted as a result; a subsequent fresh
write succeeds. Only a call's own temporary is cleaned up. Unrelated crash orphans
are preserved, not deleted automatically.

FAISS writes remain confined to a fresh catalogue directory; metadata is published
only after the native index is saved and hashed. This phase does not make all
historical per-phase writers or multi-file operations transactional.

### Cache identity, with explicit schema 2

Legacy keys omitted dependencies needed to prove reuse across model, processor
or supervision changes. An opt-in v2 contract now records:

| Dependency | Identity affected |
|---|---|
| Image ID and image content SHA-256 | Raw inference |
| Model ID, checkpoint/head identity and processor identity | Raw inference |
| Ordered categories, prompts and explicit prompt-to-category map | Raw detection inference |
| Preprocessing, crop/coordinate frame, precision and relevant inference settings | Raw inference |
| Native annotations, ignored regions, taxonomy, completeness flags and source group | Supervision/context |
| Cache schema version | Both |
| Full inference/context identity and explicit metric protocol | Evaluation fingerprint |

`file_identity` computes content digests for an explicit logical-role map of local
files without loading models. Moving identical files does not invalidate identity;
changed bytes do, including same-size content with restored mtime. Normal
size/mtime changes during hashing are rejected. Callers must list every relevant
encoder/head/config/processor/tokenizer dependency. Mask files likewise need
content digests in supervision; a path alone cannot identify their contents.

The contract owns a canonical JSON snapshot and validates its keys on read.
Malformed schemas, non-JSON/non-finite values, mismatched image/model bindings,
changed inference, and corrupt payloads are rejected. A label/group-only change
can reuse raw vectors/predictions while explicitly invalidating their evaluation
context. If annotations change the actual crop, that transformation must also be
included in preprocessing. No source group is silently merged or reassigned.

`incremental_embeddings(..., cache_contracts=...)` stores each fresh v2 vector,
vector hash and inference provenance together in one atomic NPZ. Image-only
embeddings do not claim fictional text queries. Legacy callers retain their
existing NPY bytes and filename formulas; no historical migration or cache
replacement occurred. Corruption is rejected and preserved for investigation.
Catalogue/query comparisons fail closed when strong fingerprints are missing
from one side or differ. Both-legacy comparisons retain their previous behavior.

The generic v2 raw-prediction API is available and tested; this is not a claim
that every historical prediction producer now emits v2. Legacy PhotoJobs and
room-layout adapters still need explicit adoption in a future fresh run. A
declared contract cannot independently prove a running encoder used its declared
files, that rights are cleared, or that incomplete annotations are exhaustive.
No loss, negative mask, scoring threshold or annotation definition changed.

### Decoded-image provenance

Hashing a path separately from decoding it could record different bytes if a
source changed between those reads. `load_image_with_digest` hashes and decodes
one immutable compressed-byte snapshot, then applies the same EXIF transpose and
RGB conversion as before. It checks a declared input SHA before processing.
Active pipeline/catalogue code uses that captured SHA, rather than hashing an
unrelated later read. Catalogue embedding also rejects source changes observed
after the encoder callback before publication.

Synthetic race tests cover changes after manifest validation and during the
encoder callback. This fixes provenance binding; it is not an adversarial file
lock. Capturing compressed bytes also uses additional memory proportional to the
source file, so no unmeasured RAM or speed improvement is claimed.

## 3. Recheck of the five E12-modified modules

`data.py` retains E12's source-checksum validation, with the consistent snapshot
and atomic writer additions above. E12's `geometry.py`, `postprocessing.py`,
`evaluation.py` and `dataset_quality.py` were not changed. Protected/ignored-region
rules, matching and score thresholds remain unchanged.

Exact before/after comparisons against E12 recorded:

- 25,000 geometry/containment/ignore-boundary cases.
- Nine saved filtering cases.
- Twelve EXIF-orientation and colour-mode fixtures.
- Three synthetic matching-policy results.

All matched exactly; see
[historical-output-equivalence.json](historical-output-equivalence.json).
Additional tests verify degenerate boxes are rejected or filtered at the
appropriate public boundary and zero-union IoU remains zero. No ignored or
protected region was promoted into supervision.

## 4. Representative CPU measurements

Both benchmarks alternated original inline geometry and current shared geometry
in the same process, required exact output equality, and kept settings fixed.
They used saved outputs; they did not perform model inference or calculate new
holdout metrics.

| Saved CPU workload | Inline median | Shared median | Difference |
|---|---:|---:|---:|
| Three authorised room captures, nine configuration/capture cases, 1,800 filtering calls per round | 27.895 ms | 28.712 ms | Shared about 2.9% slower |
| 5,700 saved predictions over 19 exposed internal-development images, three repeats per round | 93.982 ms | 94.621 ms | Shared about 0.7% slower |

The first used ten alternating rounds; the second eight. The second benchmark
read partition metadata to verify membership, decoded zero photographs and used
no ground truth for filtering/scoring. Saved predictions were already capped at
300 per image, so this does not benchmark all raw model queries or total detector
latency. These small workload differences do not justify another geometry
implementation; **no optimization was applied**. Raw measurements and exact
output digests are in the two performance JSON reports.

## 5. CPU verification results

| Verification | Before / E12 | Final / E13 |
|---|---|---|
| Broader available CPU suite | 379 run: 336 pass, 32 skip, 11 errors | 424 run: **391 pass, 33 explicit skips, zero errors/failures** |
| Original 11 failing IDs | 11 errors | **10 pass, one documented native-PDF skip** |
| Focused new reliability tests | Not present | **45 pass** |
| Direct new-test-file entry point | Not present | **45 pass** |
| Historical output comparisons | Recorded E12 behavior | Exact matches described above |

Two TinyTransformers tests that perform real model forwards were excluded from
the broader CPU run under this phase's no-inference restriction, not reported as
passes or included in its 424 executed tests. The existing 32 optional/platform
skips remain; the additional skip is the native PDF test. No blanket exception or
blanket skip was added to hide source failures. Intermediate iteration transcripts
are retained; `CPU-reliability-final.*` and `CPU-broad-final.*` are authoritative.

Tests cover content/prompt/category/processor/annotation/group invalidation,
legacy cache equivalence, corruption rejection, partial-write recovery, atomic
no-clobber behavior, output permission handling, EXIF/RGB consistency and source
replacement. Model behavior is exercised through deterministic stubs and saved
predictions. This is strong CPU regression evidence, not a new GPU experiment.

## 6. Historical preservation and replay

Full content inventories recorded **29,225 files before** and **29,229 after**.
The exact delta is nine intended active-code/test modifications and four new
active-code/test files. **29,216 original files are unchanged**, including all
**28,310 data/model/historical-report files** in the inventory. No existing file
was deleted. New phase tools/reports are separately sealed. Detailed inventories
remain local and ignored; only digests and aggregate preservation evidence are
committed. The separate Mac project was not accessed or modified.

Parent E12 code blobs match their original recorded seals. The E12 live-source
checker must run at its recorded commit because E13 intentionally changes
`data.py`; old seals were not rebound to new code. Historical source snapshots,
reports, checkpoints and inputs remain intact for replay. This phase did not
rerun training or prove GPU reproducibility. Earlier E11 preservation verification
also passed. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

New files and public evidence have a phase-specific artifact seal and SHA-256
list. See [preservation.json](preservation.json),
[historical-source-replay.json](historical-source-replay.json), and the
[verification instructions](../../tools/phase10e13-v1/README.md). Code/report
commits are confined to the experimental branch; model weights, photographs,
datasets and private inventories are excluded from the push.

## 7. Remaining debt and recommended next step

1. Adopt explicit v2 identities in fresh active query-cache adapters, binding
   contracts to actual runtime files/configuration. Legacy mtime-oriented
   PhotoJobs/room-layout caches remain legacy; they were not silently certified.
2. Replace repeated whole-file prediction rewrites with a separately designed
   per-image journal/commit protocol. Current atomic JSON avoids partial targets
   but does not remove repeated serialization cost or coordinate concurrent
   writers. Append-only correction logs also need a dedicated recovery policy.
3. Validate the genuine optional PDF integration on a supported machine with its
   approved dependencies. Improve the production missing-dependency diagnostic
   in a separately scoped change rather than implying it works on WSL.
4. Audit other legacy writers, native multi-file index transactions, downloader
   redirect/source enforcement and large-image snapshot memory with appropriate
   synthetic workloads before extending these changes.

**Recommended next action:** one CPU-only end-to-end synthetic adapter test that
adopts v2 provenance for a fresh query-cache run, exercises interruption/resume,
and proves label-only changes trigger metric recomputation without rerunning
inference. No new acquisition or model experiment is needed for that step.
