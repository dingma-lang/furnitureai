# Autonomous provisional weak-label furniture training

This is separate from Phase 2A and the verified Phase 2B evaluator. It requires no human annotation to train. It never creates human review evidence, changes reviewer flags or consumes official validation/test features for tuning. Results are **provisional weak-label development metrics**, not human-verified accuracy.

Default output: `/home/tristary/furniture-ai-ml/data/abo/experiments/weak-training-v1/`.

The first run is complete: 288 images / 130 products / 129 known identity groups; eight configurations × three internal folds; selected cosine head, mean product macro-F1 91.44%, fold SD 12.49 percentage points. This differs from the old eight-run 83.43% protocol and is not a controlled improvement claim. See the output `report.md` and `provenance-report.json` for class support and limitations.

For a single autonomous command that prepares a fresh output if needed, runs its GPU smoke gate, processes/resumes the queue and generates the provenance report:

```bash
.venv/bin/python -B tools/weak-training-v1/run_queue.py
```

Use `--output data/abo/experiments/weak-training-v2` for a fresh dataset/code version and optionally `--source-inventory <new local versioned inventory>`. It forwards graceful stop signals to its active child and never proceeds to training without the smoke gate.

## Monitor and resume

From `/home/tristary/furniture-ai-ml`, always use the project interpreter:

```bash
# Monitor logs and leaderboard, without loading model features or scoring anything.
tail -f data/abo/experiments/weak-training-v1/training.log
.venv/bin/python -B tools/weak-training-v1/pipeline.py status

# Resume existing incomplete GPU work. Completed matching trials are verified and skipped.
.venv/bin/python -B tools/weak-training-v1/pipeline.py run

# Graceful stop: the running queue finishes/checkpoints its current epoch.
touch data/abo/experiments/weak-training-v1/STOP

# Resume after a requested stop: remove only this new control file, not checkpoints.
rm data/abo/experiments/weak-training-v1/STOP
.venv/bin/python -B tools/weak-training-v1/pipeline.py run

# Rebuild the provenance report from TRAIN-derived folds only; no official holdout scoring.
.venv/bin/python -B tools/weak-training-v1/report.py

# Verify that historical model, data, review and experiment artifacts are unchanged.
.venv/bin/python -B tools/weak-training-v1/pipeline.py verify
```

`SIGINT`/`SIGTERM` also request a graceful stop. An epoch is committed atomically before the next stop check. `SIGKILL`/power loss resumes from the last committed epoch. `progress.pt` stores model, optimizer, epoch, best state, history and patience; epoch-seeded sampling/dropout makes restart deterministic. A test compares interrupted/resumed training with uninterrupted training bit-for-bit on synthetic fixtures. Interrupted feature acquisition reuses completed per-image embedding shards. No inference/training falls back to CPU in production commands; the report uses CPU forwards of tiny heads on internal development embeddings only.

To run in the background from your GPU-visible Ubuntu terminal:

```bash
nohup .venv/bin/python -B tools/weak-training-v1/pipeline.py run \
  >> data/abo/experiments/weak-training-v1/training.log 2>&1 < /dev/null &
```

An OS file lock prevents concurrent runners in the same output directory. A completed queue will skip completed matching results; it does not retrain them. A modified code/config/data fingerprint stops incompatible resume instead of replacing historical outputs. Use a new versioned output root for changed code or data.

## Fresh versioned run

```bash
.venv/bin/python -B tools/weak-training-v1/pipeline.py prepare \
  --output data/abo/experiments/weak-training-v2
.venv/bin/python -B tools/weak-training-v1/pipeline.py smoke \
  --output data/abo/experiments/weak-training-v2
.venv/bin/python -B tools/weak-training-v1/pipeline.py run \
  --output data/abo/experiments/weak-training-v2
.venv/bin/python -B tools/weak-training-v1/report.py \
  --output data/abo/experiments/weak-training-v2
```

Prepare refuses to replace an existing manifest. Smoke must pass before run; the gate binds code and cache metadata. Every experiment root holds its own preservation snapshot. Reports may be regenerated inside the new output root; model and completed trial artifacts remain unchanged. No existing experiment directory is a permissible new output target.

`prepare --source-inventory <new-versioned-local-ABO-inventory.json>` accepts a new inventory under `data/abo/` with the original inventory schema: bound local image ID/path/SHA, official small-image source URL, product/class, train assignment, eligibility/rejection evidence and known group/family/capture fields. The original Phase 1D holdout inventory is **always** reserved in addition to the new inventory's holdouts. New images must be inside `data/abo/image-pilots/`, match their SHA-256 and have URLs/IDs matching the already-acquired official image metadata. There is no downloader. Do not modify Phase 1D to extend the data.

The separate implementation supports up to 5,000 manifest images rather than enforcing the original 275-image training limit. This bounded version allows at most 256 newly computed image embeddings per invocation's dataset. Increase bounds only in a new code/output version after considering resources; the original Phase 2A code stays unchanged. A synthetic 350-image grouping test verifies that the new pipeline does not impose 275.

## Data and labels

1. Inspect official local listing/image metadata and the existing local image inventory. Exclude source/category conflicts, hard prior rejections, shared generic assets, suspected texture/blur/mixed-scene risks and identities/families reserved for holdout. Validate file integrity, dimensions, variation, simple blur signal and original hashes.
2. Retain original metadata categories as provisional labels. Do not use human category corrections or review decisions. Record raw product types/title provenance, original eligibility, automated confidence and unresolved whole-object/capture status.
3. Reuse original SigLIP 2 train embeddings. Compute only missing local training-image embeddings and fixed positive/negative text embeddings from the already-saved, frozen local SigLIP 2 checkpoint. No weights or photographs are downloaded. Fixed prompts/thresholds are in `manifests-v1/policy.json`.
4. Strong model/negative-prompt contradictions exclude cached candidates; previously pending extra views require metadata/model agreement and a fixed cosine margin. Scores/margins are uncalibrated signals, not probabilities. They cannot establish whole-object visibility or physical identity. Known duplicate captures are retained once, not called independent.
5. Build transitive conservative product/group/family/same-brand-model connections before folds. These are leakage constraints, **not physical-identity merges**. Shared generic asset IDs are excluded rather than creating identity components. Holdout product/family, image/hash/pixel/capture overlaps stop the run. Unknown aliases can still exist. Internal fold checks prevent known product/group/hash/pixel/capture overlap; contradictory grouped labels stop rather than being auto-corrected.
6. Use deterministic three-fold class-stratified grouping from the training partition only. Never tune on the official validation/test partition. Fit class weights and centroid statistics using the current internal training fold only. Sample one image per product per epoch to limit view-count imbalance.

The new local set adds **19 views, zero product identities**. It retains 269 original cached vectors. Candidate pool 333; 34 model-contradiction exclusions and 11 known capture-redundancy exclusions. Final support is sofa 51 images/24 products, chair 68/29, table 58/27, bed 10/4, desk 29/11, cabinet 33/14, lamp 39/21. Four bed identities make category estimates fragile.

`manifests-v1/metadata-only-expansion.json` lists 4,423 additional nonlocal/nonreserved candidates: bed 202, desk 148, sofa 998, chair 1,530, table 837, cabinet 206, lamp 502. These are metadata candidates, not training images. Any substantial acquisition requires separate approval of exact official sources, CC BY 4.0 attribution, image counts and transfer/storage limits. No such acquisition was necessary for this batch.

## Models, outputs and limitations

The bounded queue compares three linear recipes, three small MLPs, a normalized cosine head and a product-balanced nearest-centroid baseline. Each trainable fold runs at most 80 epochs with patience 12. Winner selection uses mean internal product macro-F1, with deterministic name tie-breaking. Heads are fitted from scratch for development; old full-training classifiers cannot initialise those folds without leakage.

The winning recipe is refit on all training groups for the median selected epoch count, without a claimed independent score. A separate 20-epoch training-only warm-start companion reuses the previous final linear classifier; it is not ranked with development metrics because that previous classifier already saw all current training products. SigLIP encoder adaptation is disabled/deferred until unique-product expansion and lightweight results warrant it.

Artifacts:

- `manifests-v1/`: frozen candidates, exclusions, labels/provenance, thresholds, code/source hashes and metadata-only expansion inventory.
- `features-v1/`: combined train-only normalized vectors, accepted row metadata, group folds, screening evidence and resumable new-vector shards.
- `trials/<recipe>/fold-*/`: `best.pt`, `progress.pt`, `curve.json`, result/metrics/config/fingerprints.
- `final-selected/best.pt`: selected recipe fit on all new weak training data.
- `final-warm-start/best.pt`: separate previous-classifier continuation; no development score.
- `checkpoint-registry.json`: checkpoint paths/hashes and original checkpoint references.
- `leaderboard.csv`, `leaderboard.json`, `completion.json`: queue comparison, completion and selection.
- `report.md`, `provenance-report.json`: label source, uncertainty, class support, internal fold-only metrics, limitations and training provenance.
- `preservation-before.json`, `preservation-after.json`: historical artifact hashes and comparison.

Resources: existing RTX 4070 Laptop GPU, 75% allocator fraction, at most one newly extracted image at a time, head minibatch 32, 30-minute queue budget per invocation, 1 GiB new output budget. This run used about 23 MiB new allocated experiment storage. Existing models are loaded read-only with `local_files_only=True`; all pretrained parameters stay frozen. Neither OWLv2 nor SigLIP weights are saved again or modified. No package/environment/power changes were made.

Scores are optimistic model-selection summaries from unverified, repeatedly curated data. The 12.49-point fold SD is variability across folds, not a statistical confidence interval. Unknown aliases, incomplete capture independence and automated screening bias remain. No result is human-verified accuracy, room detection accuracy, retrieval performance or a fresh external benchmark. The original reviewed holdout and all Phase 2B gates remain available for a separately authorised independent evaluation.

Tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/weak-training-v1 -p 'test_*.py' -v
```

Fifteen tests pass, including product/family/capture/hash leakage checks, generic-stock nonmerging, dynamic row counts, screening uncertainty, shape/gradient checks, atomic interrupted writes, checkpoint resume equivalence and no production CPU fallback. Synthetic CPU tests do not train on real dataset features.
