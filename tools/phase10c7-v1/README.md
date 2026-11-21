# Phase 10C.7 — actual-query, zero-update GPU smoke

The approved smoke has run exactly24 original-OWLv2 GPU forwards on the frozen
24-photo training-only proposal. **Do not rerun or resume GPU execution:** the
forward budget is consumed. The launcher and worker use exclusive one-run markers
and durable per-forward reservations; existing files are never overwritten.

The model is original FP32 safetensors, all parameters frozen, eval mode, with
FP16 CUDA autocast matching Phase10B optimization. The pinned slow processor pads
only bottom/right to max(H,W), then resizes to960square. All3600queries and7logits
are saved per image. Original-coordinate boxes are inverse-transformed without
clipping, so padding/clipping cannot fabricate region containment.

Execution order was:

```bash
# Already completed; do not repeat these GPU preparation/execution commands.
.venv/bin/python -B tools/phase10c7-v1/preflight7.py
.venv/bin/python -B tools/phase10c7-v1/launch7.py

# Separate CPU-only analysis, already completed in cpu-analysis-v1.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c7-v1/analyze7.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c7-v1/signal7.py
```

Raw arrays and checksum receipts are durable before any matching/loss analysis.
If a later CPU replay is needed, use a newly versioned analysis destination and
unchanged raw caches. Never repeat inference to recover a CPU result.

CPU analysis applies the exact legacy Hungarian matching and focal normalization
to identical raw outputs, then the unchanged C.6 Chair-negative mask. It records
every added query index, source region, original/square coordinates, Chair score,
matched-positive assignment, and per-region coverage. Independent original-pixel
containment and protected-overlap checks passed. No confidence threshold or NMS
filters training queries. No detection accuracy/held-out scoring is performed.

There are292 additional terms across19photos/groups;24/41regions have coverage.
All17uncovered regions have no fully contained predicted query box. No region was
expanded and no box was clipped to force coverage. Positive targets, legacy masks
and non-Chair terms are unchanged; no known positive/protected overlap is targeted.

Signal strength is a separate caveat: added-query median Chair score is~0.000158,
maximum~0.0191, with none≥0.15. FP64 mean added focal contribution is~1.15e-7 per
photo. FP32 full-loss reduction often rounds away this tiny value, while the added
mask terms remain real. No backward pass was performed, so gradient strength or
AMP underflow behavior is unmeasured. A future controlled experiment may test
prevention of score inflation, not assume an immediate strong hard-negative loss.

One worker has a600second parent hard deadline,6.5GiB VRAM stop limit, an allocator
cap with driver-memory headroom, and sampled device-memory monitoring. It creates
no optimizer, gradient graph, checkpoint or delta loading. Original weights and
full in-memory state hashes match before/after. GPU runtime was15.67seconds;
peak allocated1.55GiB, reserved2.17GiB, sampled whole-device used3.33GiB.

Private outputs live in
`data/open-images/experiments/phase10c7-query-smoke-v1/`:

- `binding.json`: authorization, exact data/model/code hashes and environment.
- `forward-*.intent.json`:24 durable forward reservations.
- `queries-*.npz` and receipts:24 raw query files saved atomically.
- `raw-complete.json`: resource/forward count and before/after model fingerprints.
- `cpu-analysis-v1/`: per-image masks/losses/query-region attribution and coverage.

All earlier phases, source images, native labels, partitions and model files remain
unchanged. No training, backward passes, optimizer steps, downloads or validation/
reserved-set scoring occurred. The source evidence remains automatic and fallible;
it was never converted to human-reviewed labels. Phase10D remains unauthorized.

Forward-free CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tools/phase10c7-v1 -p 'test*.py' -v
```

The tests use only tensors, coordinates, the local processor and temporary files;
they do not call backward even on synthetic tensors. Public aggregate results are
under `reports/phase10c7-v1/`; all images/query files/detailed evidence stay local.
