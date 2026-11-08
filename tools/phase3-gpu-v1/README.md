# Phase 3 expanded GPU training — completed bounded stage

Project interpreter: `/home/tristary/furniture-ai-ml/.venv/bin/python`.
Output: `data/abo/experiments/phase3-gpu-v1/`. Completed old experiment directories and verified Phase 2B code are unchanged.

## Protocol correction and dataset

The earlier preparation runner used the frozen panel for early stopping and would rescreen all 1,484 acquired images. It was **not run** for this stage. The new isolated runner follows the latest instruction: enforce exactly the final audit-v3 allowlist, use training-only internal folds for all stopping/selection, then perform one controlled frozen-panel comparison after final checkpoint hashes are frozen.

FIT: **532 images / 281 listing IDs / 280 known leakage components**. This combines 182 original FIT images / 85 listings / 84 components with exactly 350 new eligible images / 196 listings. Known components are not proven independent physical identities. The 605 rejected, 304 quarantined and 225 unresolved images cannot enter optimization. Whole-object/capture identity and weak labels remain unverified.

Training image counts/products: sofa 80/40; chair 66/39; table 50/28; bed 54/29; desk 121/66; cabinet 59/30; lamp 102/49. Same-product/family/capture/exact-pixel tokens stay wholly in an internal fold. Original known metadata aliases are treated conservatively; different listing IDs are never assumed physically independent.

No official Phase 2B validation/test images or feature arrays are opened. Their paths receive metadata/stat-only preservation checks; their existing metadata identifiers/hashes are used only for isolation. The frozen panel matrix is absent from the training feature file and is not semantically loaded until the comparison stage. Its stored hash/stat is locked in advance. Original FIT vectors are read from only the first 182 rows of the old baseline cache through a read-only mmap.

All 350 eligible cached new embeddings were directly compared with fresh GPU encoder forwards, under the pinned model and image preprocessing, before reuse. Per-image source/cached/output hashes, model fingerprint, maximum numerical difference and reuse status are recorded. All 350 matched and were reused, together with the 182 original FIT vectors. No old cache/model file is changed.

## Experiment bounds and selection

Eight configurations: cosine (two learning-rate/regularization settings and an unweighted variant), linear (sqrt and inverse class weighting variants), and three small MLPs with dropout/regularization variations. Expanded model selection uses deterministic three-fold group splits. The fixed previous cosine recipe is refit as the baseline, with stopping epochs selected from two baseline-only internal folds; only two bed groups exist in the baseline FIT pool.

Maximum **30 GPU fits / 2,323 total training epochs**: one three-epoch smoke, 24 expanded folds, two baseline folds, two mandatory final refits and an optional expanded cosine-control refit. This run completed **29 fits / 806 epochs** because the selected model was the cosine control. Per-fold maximum 80 epochs, patience 12, minibatch 32, per-process CUDA allocator fraction 75%, per-invocation queue deadline 30 minutes, FIT product-based class weights and one view/product/epoch sampling.

All heads start from scratch. Historical heads already saw frozen-panel products and cannot be fair leakage-free baselines. The baseline therefore reuses the prior cosine **recipe**, not its trained weights. Deterministic seeds depend on run/recipe; this comparison is not a seed-matched causal ablation of dataset size. Epoch counts and class balance can also differ.

The GPU smoke gate binds code/cache hashes. A synthetic OOM test injects failure after a partial optimizer update: the epoch rolls back model/optimizer state, reduces batch size, and matches a reference run exactly. OOM at batch one propagates. Nonfinite model/gradient failures propagate. Each epoch commits model/optimizer/best-state/history atomically for deterministic recovery. Completed result/checkpoint hashes are verified and skipped, not refit.

## Results

Selected **cosine-reference** (lr 0.003, wd 0.001, inverse-square-root product-class weights). Internal product macro-F1 folds: 100.00%, 100.00%, 97.23%; mean **99.08%**, SD **1.60 percentage points**. Balanced accuracy folds: 100.00%, 100.00%, 96.98%; mean **98.99%**. These high internal numbers are model-selection agreement with screened weak labels, not verified accuracy.

After selection and final checkpoint hashes were frozen, a single controlled comparison used the identical 106 frozen images / 45 products, pinned vectors/preprocessing, class order, product-view aggregation and metric code:

- Refit prior cosine recipe: **79.13% product macro-F1**.
- Selected expanded cosine: **88.96% product macro-F1**.
- Difference: **+9.84 percentage points**; paired group-bootstrap 95% interval **+0.42 to +21.75 points**, 845 valid / 1,000 resamples (155 omitted for missing classes).

This supports higher weak-label agreement on the fixed panel, not human-verified real-world accuracy. The panel was exposed in historical work, and includes only two bed/four desk products. Bootstrap intervals condition on these models/examples and do not include training-seed or label uncertainty. SigLIP-based acceptance can favour SigLIP embedding agreement. The historic 91.44% used different data/folds and is not directly comparable to either this 99.08% internal result or the 88.96% fixed-panel comparison. No further optimization used the comparison results.

Detailed per-class confusion matrices, precision/recall/support, uncertainty, all per-fold metrics and interpretation are in `report.md` / `results-report.json`.

## Artifacts

- `manifests/dataset.json`: exactly 532 FIT rows, source/provisional labels, known groups and frozen internal fold assignments.
- `manifests/policy.json`, `configurations.json`, `input-lock.json`: reproducibility and strict source/allowlist bindings.
- `features-v1/fit-only.npy`, `metadata.json`, `validated-shards/`: train-only vectors and GPU-validated reuse provenance; no frozen panel vectors.
- `trials/<configuration>/fold-*/`: unique internal checkpoints, curves, metrics and resume state.
- `baseline-cv/`: baseline-only epoch selection.
- `final-baseline/best.pt`: previous cosine recipe refit without the panel.
- `final-selected/best.pt`: selected expanded recipe refit without the panel.
- `selection-frozen.json`, `comparison-checkpoints-frozen.json`: written before panel scoring.
- `comparison/intent.json`, per-arm committed logits and `result.json`: the one controlled comparison. Identical selected/control checkpoints reuse the same predictions.
- `leaderboard.csv`, `leaderboard.json`, `checkpoint-registry.json`, `report.md`, `results-report.json`, `completion.json`.
- `preservation-before.json`, `preservation-after.json`: 3,938 historical files checked; official image/feature files use stat-only checks rather than opening them.

The report records about 22.94 seconds of actual fitting and 11.90 seconds of GPU embedding validation; maximum observed **PyTorch allocated** VRAM was 1,542,137,344 bytes. This is not total driver or reserved memory. No OOM occurred in the real queue. No model download, installation, encoder fine-tuning, OWLv2 training or official holdout scoring occurred.

## Monitor, stop and resume

From the project root:

```bash
.venv/bin/python -B tools/phase3-gpu-v1/train.py status
cat data/abo/experiments/phase3-gpu-v1/leaderboard.csv
tail -f data/abo/experiments/phase3-gpu-v1/training.log
```

The completed queue is no longer training. To resume interrupted work safely:

```bash
.venv/bin/python -B tools/phase3-gpu-v1/run_all.py
```

Completed matching preparation, smoke and training stages are skipped. A completed comparison validates its binding and returns without scoring again; report regeneration reads committed results only. Do not delete or alter selection/comparison completion locks. Changed code/data requires a **new** versioned output root and cannot be resumed into historical artifacts.

Background execution:

```bash
nohup .venv/bin/python -B tools/phase3-gpu-v1/run_all.py \
  >> data/abo/experiments/phase3-gpu-v1/training.log 2>&1 < /dev/null &
```

Graceful stop/resume during active training:

```bash
touch data/abo/experiments/phase3-gpu-v1/STOP
# After the active epoch commits and the process stops, remove only your own control file:
rm data/abo/experiments/phase3-gpu-v1/STOP
.venv/bin/python -B tools/phase3-gpu-v1/run_all.py
```

SIGINT/SIGTERM are forwarded to the active child and stop at a committed epoch. An interrupted comparison resumes its fixed checkpoint predictions; it never reruns model selection. Do not run multiple orchestrators concurrently. Training itself has an exclusive file lock. No official held-out evaluation is invoked by these commands.

Verification:

```bash
.venv/bin/python -B tools/phase3-gpu-v1/train.py verify
.venv/bin/python -B -m unittest discover -s tools/phase3-gpu-v1 -p 'test*.py' -v
```

Thirteen tests passed: allowlist contamination, family/capture grouping, official/frozen isolation, label conflicts, FIT-only balancing, deterministic selection, no CPU fallback, bitwise interrupted-run recovery, skip/hash guards, OOM rollback equivalence, comparison gating and paired uncertainty. CPU fitting in tests uses synthetic fixtures only. Existing Phase 2B human-review and accuracy integrity gates were not modified.
