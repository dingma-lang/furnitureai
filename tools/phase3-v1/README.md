# Furniture AI Phase 3 — acquisition and fair-comparison preparation

**No new photographs or models have been downloaded. No real training or official holdout scoring was run.** The immutable preparation is `data/abo/experiments/phase3-preparation-v1/`. The source implementation is isolated here; completed weak-label/Phase 2A experiments and the verified Phase 2B evaluator are unchanged.

## What is ready

- `prepare.py`: complete image-count reconciliation, original/current cache and checkpoint verification, 4,423-candidate inspection, ranked 500-product acquisition, source URLs/main/additional roles, known-family exclusion and frozen internal development protocol.
- `acquire.py`: approval-gated downloader with no-network default, strict official source/redirect rules, current README/licence digest checks, all-object HEAD preflight, size limits, SHA-256/decode verification, bounded retries, durable conservative transfer ledger and interrupted-commit recovery.
- `runner.py`: no-network dry run by default, baseline cache construction from existing vectors, expanded image screening and frozen-encoder feature extraction after acquisition, fixed-development linear/cosine/MLP/centroid comparison, class-balanced product sampling, checkpoint resume, leaderboard, registry and family-bootstrap/per-category reports.
- Tests: 16 passing tests for selection, family/identity/hash isolation, label conflict, frozen-vector tampering, approval refusal, interrupted transfers, retries/limits, safe commit recovery, checkpoint preservation and reproducible experiment tie-breaking. Fake HTTP responses and synthetic classifier examples do not acquire real data or approve labels.

## Audit and acquisition proposal

The original count is exactly `275 - 6 + 19 = 288`. All six omitted originals were redundant known captures represented by retained image IDs. Files were not deleted, images were not relabelled, and product count stayed at 130. `expansion-audit.json` lists every removed ID, retained representative, added ID and all 45 screening/redundancy exclusions among 333 candidates. Original/current vector hashes, 27 registered checkpoints and historical preservation hashes were checked.

The proposed acquisition contains **500 listing IDs / 500 known-family groups / 1,484 image objects**, prioritising beds/desks and rebalancing remaining capacity conservatively:

| Class | New listing IDs |
|---|---:|
| Bed | 50 |
| Desk | 99 |
| Cabinet | 93 |
| Lamp | 77 |
| Sofa | 61 |
| Chair | 60 |
| Table | 60 |

484 products have three candidate images; 16 have two. Distinct image IDs/main/additional roles are not proof of independent captures. One item per known brand/model/family component is selected; no claim is made that all 500 are different physical designs or objects. Generic shared assets are excluded rather than used to merge unrelated products. All existing local product identities/known families and official reserved images are excluded. Ambiguous accessories, sets/components, category contradictions and insufficient exclusive links are recorded in `identity-exclusions.json`.

Exact source URLs are in `acquisition-plan.json`, all under:

`https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/small/<metadata path>`

Metadata paths, image IDs, original dimensions and main-image roles are checked using existing official Phase 1A metadata, including safely reading the existing listings archive without extraction. No remote request has been made in this phase. Source existence, current byte size, ETag, content type and actual image quality remain unverified until approved execution.

Existing official documentation identifies ABO as CC BY 4.0. Preserve Amazon.com and dataset-builder attribution from the original README, the licence link/text, original URLs, hashes and any changes. The downloader retrieves the two specifically listed official README/licence documents (at most 64 KiB each) and stops if they differ from the verified local copies; it does not silently accept changed licensing. Original SigLIP 2/OWLv2 weights stay local and unchanged; encoder adaptation/detection training is not enabled.

Cost and storage proposal:

- Images requested: **1,484**, hard ceiling **1,500**; no alternative/reserve downloads beyond the frozen plan.
- Estimated image transfer: **9,820,943 bytes**, using the historical mean of 624 Phase 1B small images. This is a point estimate, **not a remote size measurement**. A historical per-object p95 scenario totals **20,048,840 bytes**; it is not a confidence bound for the total.
- Per-object cap: **512,000 bytes**; at most **two attempts per object**. Maximum two-attempt image reservation **1,519,616,000 bytes**, plus **131,072 bytes** initial documentation reservation. The cumulative hard cap is **2,000,000,000 response-body bytes**, including retries/partials/documentation. Conservative reservations remain charged after a crash, even if observed bytes are lower.
- Estimated allocated image storage: approximately **12.16 MB**; new raw feature matrix contribution about **4.56 MB**. Image storage has a **2 GiB cap**; reserve up to another **2 GiB** for a training run/checkpoint history, plus the small preparation artifacts. Expected normal use is much lower; old weights are not copied.
- Initial no-retry request count: **2,970** (image HEAD+GET and two documentation GETs). Monetary network/provider fees have not been quoted; no paid API is configured.
- Image destination: `data/abo/image-pilots/phase3-expansion-v1/` (not yet created).

The proposal is subject to remote HEAD availability. All selected objects are checked before image GETs; missing/oversized/redirected/unexpected sources stop execution rather than silently substituting another identity. Automated screening may leave fewer than 500 eligible products. Metadata, captures, whole-object suitability and unknown physical aliases remain uncertain.

## Frozen fair comparison

`frozen-development-protocol.json` fixes **106 images / 45 products** (45 known groups) from a previously used TRAIN-derived fold. Its exact existing weak labels, row order, hashes and 768-dimensional vectors are frozen. There are 2 bed, 4 desk, 5 cabinet, 7 lamp, 8 sofa, 10 chair and 9 table products in this panel. The remaining **182 images / 85 products** form the unexpanded FIT baseline. New acquisitions may only add to FIT, never to development.

This panel was exposed during earlier adaptive experiments. It is **not an independent test** and the old 91.44% cannot be compared directly. Comparable baseline and expanded heads must be retrained from scratch on their allowed FIT sets with the same frozen panel, seeds, recipes, stopping policy and fixed encoder. Historical heads saw panel products and are therefore not used to initialise these comparisons. No final refit includes the frozen panel. Official validation/test decisions and vectors are not used for tuning; reserved identifiers/image hashes may be used only for leakage prevention.

Eight recipes (linear, cosine, small MLP, centroid) run at most 80 epochs each, patience 12, head batch 32, 75% CUDA allocator fraction, 30 minutes per invocation. FIT-only class/product weighting and one-view-per-product sampling reduce view-count imbalance. Model/optimizer progress is atomically checkpointed per epoch. Selection uses this fixed internal product macro-F1 with deterministic name tie-breaking, and is therefore optimistic. Reports include per-class support/precision/recall, confusion matrix and fixed-seed family bootstrap intervals. Bootstrap samples without all categories are counted and excluded; an interval is withheld if fewer than 100 valid resamples remain. It is not an independent real-world accuracy interval.

The baseline feature cache has already been constructed and checked bit-for-bit against the frozen panel at `data/abo/experiments/phase3-baseline-v1/features-v1/`. No classifier was fitted. An expanded cache can contain up to **1,666 FIT images plus the same 106 development images** before screening. Extra image features are sharded and resumable; old vectors are reused. New exact/perceptual duplicate suspects are quarantined, not identity-merged. Automated quality and fixed SigLIP prompt agreement provide uncalibrated weak evidence, never human review status.

## Commands safe to run now

From `/home/tristary/furniture-ai-ml`:

```bash
.venv/bin/python -B tools/phase3-v1/acquire.py
.venv/bin/python -B tools/phase3-v1/runner.py --mode expanded
.venv/bin/python -B -m unittest discover -s tools/phase3-v1 -p 'test*.py' -v
```

These perform no real downloads or fitting. `prepare.py` produces a fresh immutable preparation root and refuses to replace one; do not rerun it against v1. All runtime sources and preparation artifacts are hash-bound. If runtime code or inputs change, create a new preparation/cache version instead of rewriting frozen history.

## Commands only after explicit acquisition approval

No approval has been recorded. `approval-template.json` has `approved=false`. After the user approves the exact scope, record a separate versioned approval file with their approval, name, unchanged plan SHA-256 and limits. The tool cannot infer approval from a template or dry run.

```bash
.venv/bin/python -B tools/phase3-v1/acquire.py --execute \
  --approval-file data/abo/experiments/phase3-preparation-v1/<approved-file.json>
```

Do not run until approval exists. The destination must be absent or contain this exact plan's recoverable ledger. An existing unrelated destination is refused. No redirects, mirrors, original-resolution archives, model weights or substitute objects are permitted.

`transfer-ledger.json` records pessimistic precharges and observed bytes, each attempt, partial paths, object SHA-256/dimensions and provenance. A failed/interrupted attempt remains charged and its partial file retained. Resume uses the same plan/command, never an unbudgeted range transfer or overwrite. At most two attempts are allowed; exhausted attempts or conflicting final files require an audit. Incomplete commits with recorded hashes can be recovered. A crash can leave actual transferred bytes uncertain; the conservative charged total still upper-bounds application body consumption. Current licence/README mismatch stops before images.

## Training execution later

Prepared commands below are **not run as part of this preparation**:

```bash
# Fair baseline (cache already prepared):
.venv/bin/python -B tools/phase3-v1/runner.py --mode baseline --train

# After approved acquisition: screen/build cache, GPU smoke gate, then train from scratch.
nohup .venv/bin/python -B tools/phase3-v1/runner.py \
  --mode expanded --build-cache --train \
  >> data/abo/experiments/phase3-expanded-v1.log 2>&1 < /dev/null &

# Monitor:
tail -f data/abo/experiments/phase3-expanded-v1.log
cat data/abo/experiments/phase3-expanded-v1/leaderboard.json

# Graceful stop at the next committed epoch:
touch data/abo/experiments/phase3-expanded-v1/STOP

# Resume only after removing your own STOP control file:
rm data/abo/experiments/phase3-expanded-v1/STOP
.venv/bin/python -B tools/phase3-v1/runner.py --mode expanded --build-cache --train
```

SIGINT/SIGTERM during the queue also request a graceful stop. Interrupted feature extraction retains completed atomic vector shards. Cache/epoch/source incompatibilities fail rather than replacing completed results. Output roots are separate for baseline/expanded runs; old classifiers and verified evaluation artifacts are not modified. Each run includes a mandatory separate three-epoch smoke gate, per-recipe checkpoint/curve/result, leaderboard, selected checkpoint registry, per-class comparison report and completion record. No CPU training fallback, package installation or power-setting changes occur.
