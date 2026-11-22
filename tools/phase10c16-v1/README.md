# Phase 10C.16 — CPU-only opt-in balancing

`balanced16.py` provides a default-off C8 objective wrapper, fixed source/region/signature BCE weights, an actual-parameter-gradient cap and an accumulation buffer helper. It has no model loader, GPU launcher or optimizer. Existing workers/defaults are untouched.

The source term averages over exact duplicate query signatures and verified region cells across the full known source group. All uncovered cells remain in the denominator. Existing positive/native focal and box losses retain their original normalization. `compose` requires measured unscaled gradients for all6C8 class-head tensors,including calibration,after the complete accumulation group;it cannot accept the C11 projection-only cache as a substitute.

For future integration, differentiate `(per_photo_branch_loss / actual_microbatches) * current_scale`, in that order. `Accumulator` sums scaled FP32 buffers and unscales once. The base is unchanged positive+native+box without the old added-source focal term. Only new source gradients are capped;then the existing global clip/AdamW protocol applies. Known positives with near-zero anchors stop the update;negative-only absolute budgets do not admit new dataset rows.

CPU analysis order: `audit16.py`, `spec16.py`, `synthetic16.py`, `detail16.py`, `browser16.py`, `verify16.py`. Each emits versioned immutable artifacts and refuses overwrites. Do not delete/reset audit directories to repeat. Run using `.venv/bin/python -B` and `CUDA_VISIBLE_DEVICES=''`. No invocation in this phase should create a CUDA context or call an encoder.

The report viewer can be served without inference:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c16-v1/browser16.py --serve
```

Open http://localhost:8778/. The server exposes cached audit metadata only,not photographs,weights or writable review/training endpoints. Browser tests use the installed Windows Chrome with GPU disabled and localhost readiness checks.

The future experimental configuration is disabled and compares against original-initialized C8 corrected Arm B:all12head tensors trainable,including shift/scale,without C11 calibration freezing or C13 regularization. The59-photo prerequisite GPU smoke is a proposal requiring separate authorization. No training launcher or new model checkpoint exists.
