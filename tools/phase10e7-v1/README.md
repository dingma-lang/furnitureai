# Phase 10E.7 — single E5 optimization-query inference run

Completed results: `reports/phase10e7-v1/report.md`.
Private output: `data/open-images/experiments/phase10e6-training-query-cache-v1/`.

The completed execution order was CPU preparation, CPU safety tests, exactly one
bounded GPU launcher, then separate CPU analysis, contact-sheet generation,
actual assistant viewing/assessment, post-cache tests and preservation checks.
No training or optimizer object was created. Every candidate remains inactive.

**Do not run the launcher again.** `launch-once.json`, `executed-once.json`,
per-forward intents, immutable outputs and completion markers prohibit automatic
retry or replacement. A future additional inference run needs separate approval
and a new destination. The original source images and E5 checkpoint are read only.

Raw layout per `queries/NN.npz`:

- `query_ids`: all 3,600 original row-major60x60 IDs.
- `logits`, `scores`: 3,600x7, unchanged full-category coverage.
- `square_cxcywh`: actual normalized model boxes before clipping.
- `original_xyxy_normalized`, `original_xyxy_pixels`: exact inverse square-pad
  transform in FP64, retaining out-of-image coordinates for safe masking.

Scores are sigmoid stored logits, evaluated on CPU in FP64 and stored FP32.
The FP16 detector forward retains FP32 model parameters and FP32 raw output
arrays. Per-image JSON binds image/source group/hash, dimensions, checkpoint,
preprocessing tensor hashes and coordinate/precision definitions. `raw-complete`
was committed before CPU summaries or inspection ran.

Read-only final verification:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e7-v1/verify7.py check
```

Isolated CPU post-cache regression tests can be repeated without GPU work:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10e7-v1 -p 'test_analysis7.py' -v
```

`analysis/candidate-manifest.json` references inactive evidence observations and
disjoint unverified versus uncertain candidate IDs. It is not a training manifest
granting new negative labels. `analysis/visual-observations.json` records actual
assistant inspection with human-review and label-approval flags false. Current
source-bbox context, unknown background and unmatched detections are never
converted to Chair absence. Seating stools/bar stools remain protected as Chair;
ambiguous furniture and source-ignore regions remain ignored.

Only tools and small reports belong in Git. Raw queries, photographs, contact
sheets, detailed evidence manifests and model weights remain private/local.
