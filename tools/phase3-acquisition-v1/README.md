# Approved Phase 3 acquisition and automated audit

The user explicitly approved the immutable plan with SHA-256:

`a47e31d3ee7762781b54608d98ff0d446b2f2a8d760fe7558b4f5cfbe5713d51`

Scope: 500 metadata listing IDs / 1,484 exact image objects, plus official ABO README and CC BY 4.0 licence. Body cap 2,000,000,000 bytes; 512,000 bytes/image; two attempts/image; 2 GiB destination allocated storage. No substitutions, alternative sources, training, model downloads or official holdout scoring.

The approval record is `data/abo/experiments/phase3-preparation-v1/acquisition-approval-v1.json`. The destination is `data/abo/image-pilots/phase3-expansion-v1/`.

`download.py` imports the frozen prepared downloader without altering it. It verifies the official licence/README against Phase 1A hashes, verifies all source HEAD responses before image GETs, handles interrupted commits, and propagates unavailable/integrity/limit failures immediately. Only recognised transport errors can retry within two attempts; corruption is never automatically retried. Image completion requires SHA-256/size, full decoding and single-part ETag/MD5 verification when available. Different bytes that decode are not silently accepted as matching the source ETag.

The ledger precharges a worst-case 512,000 bytes before each image attempt and 64 KiB before each documentation transfer. Partial/failed reservations are never refunded. Observed body counts and conservative charges are distinct; crash uncertainty cannot weaken the conservative cap. New transient partial files are retained on failure; completed image files are not overwritten. The immutable plan, licence/README records, source preflight and transfer ledger preserve provenance. Do not start a second process against the same destination; an exclusive lock prevents it.

Approved execution:

```bash
.venv/bin/python -B tools/phase3-acquisition-v1/download.py
```

A completed acquisition should not be unnecessarily rerun. Inspect any stop record first; do not use a resume command to ignore an unavailable source, licence change, corrupted file, exhausted attempt or cap violation. The existing prepared downloader checkpoints transfers; safe transport interruption can resume the exact same approved image plan and ledger. Unrecoverable conflicting final files require an audit, never replacement.

Automated verification:

```bash
.venv/bin/python -B tools/phase3-acquisition-v1/screen.py
```

This verifies all acquired bytes/hashes/decodes, compares exact and perceptual signatures against both new images and all existing local references, checks known-family/holdout isolation, then uses the existing frozen SigLIP encoder and fixed prompts to screen only new images. No classifier or encoder fitting runs. Existing holdout image pixels/hashes may be compared for leakage prevention; no holdout is forwarded through a model. Original model files, dataset partitions and reviewer declarations remain unchanged.

Labels and physical product/capture identity remain provisional. Statuses distinguish accepted weak candidates, rejected quality/model disagreements, quarantined duplicate/identity suspects, and uncertain low-margin agreement. Hamming-distance suspects can be false positives; identities are never automatically merged. Nonduplicate images are not declared independent captures. Model cosine margins are not calibrated probabilities. The audit's acceptance is not human verification or a completed training dataset.

Embeddings are atomically cached at `screening-cache-v1/` under the new destination for screening recovery. Reports use fresh `audit-vN/` directories and include `report.md`, `summary.json`, `screening.json`, `products.json`, `provisional-candidates.json`, pair diagnostics, SHA-256 sums and historical preservation checks. Current exact disk use is recorded in `final-storage.json`; response-body totals are in `acquisition-summary.json` and the transfer ledger. Source photographs and old experiments are never modified.

Tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase3-v1 -p 'test*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase3-acquisition-v1 -p 'test*.py' -v
```

Sixteen existing Phase 3 tests cover source/approval/limit guards, interrupted download accounting, safe commit recovery, leakage and frozen evaluation preservation. Two additional synthetic tests cover full-decode/ETag integrity and explicit provisional acceptance/status handling. Tests perform no real downloads or human confirmations.

## Product-level guard and authoritative completion report

`product_guard.py` reads the original image audit without changing it, then conservatively excludes every otherwise accepted view of a product with a cross-product exact/perceptual similarity suspicion. Both identities are retained as distinct unresolved records; they are never merged. Each invocation creates a new `audit-vN/` report and records its code/source hashes. This prevents another view of a possibly overlapping product from bypassing image-level quarantine.

The completed acquisition's authoritative report is `audit-v3/report.md`, with `audit-v3/provisional-candidates.json` as the conservative image subset. Audit v1 is the image-only first pass; v2 is retained as earlier generated report history. Do not substitute v1's 434 image-only candidates for the final 350 product-aware candidates. The authoritative report has 196 listing IDs, 605 rejected images, 304 quarantined images and 225 low-margin unresolved images. All identities/captures remain provisional.

The two product-guard regression tests bring the new acquisition test count to four; with the 16 frozen Phase 3 tests, 20 tests passed. The user-approved image and documentation transfers completed without retries. No fitting, original model modifications, split changes or official holdout scoring occurred.
