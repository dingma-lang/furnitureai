# Phase 10D — one original-classifier-preserving box-only arm

Uses the approved C21 destination
`data/open-images/experiments/phase10c21-box-only-v1/` and fresh reports under
`reports/phase10d-v1/`. Historical phases are immutable dependencies.

Executed preparation sequence, with no automatic retries:

1. CPU-only `prepareD.py`: all original persisted tensor values, actual scope,
   original derived box bias, preprocessing and exact data/ordering binding.
2. CPU-only `verifyD.py`: meaningful scope, resource, persistence and unchanged
   metric tests. The launcher requires its passing binding receipt.
3. Exactly once `launchD.py`: hard 600-second worker budget; original-only local
   model loading; 4 zero-update smoke + 295 training + 4 fixed-query invariance
   + 19 exposed development forwards, at most 75 optimizer steps.
4. Separate CPU-only `analyzeD.py`: no Torch import; prediction persistence before
   unchanged metric calculation; immutable original/C20 caches reused.
5. CPU-only `verifyD.py --postrun`, `reportD.py`, then resource/provenance sealing.

These are **not commands to repeat a completed experiment**. A durable launch
token prevents another attempt, and outputs use no-replace atomic completion.
No existing files are removed to permit rerunning a generator.

Monitor an active authorized run:

```bash
tail -n 25 data/open-images/experiments/phase10c21-box-only-v1/gpu.log
```

Request a graceful stop only if desired:

```bash
touch data/open-images/experiments/phase10c21-box-only-v1/STOP
```

A STOP or failure is terminal for this authorization. Preserve all counters,
intents and checkpoints; do not automatically resume. The last epoch snapshot
contains six box weights, AdamW moments, cursor and Python/CPU/CUDA RNG. A future
separately authorized recovery must validate exact base/config/frozen/grid
fingerprints and account for previously consumed time/forwards/updates.

CPU regression checks can be repeated without starting training:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10d-v1 -p 'test_boxD.py' -v
```

`architecture.json` lists every actual trainable/frozen parameter and buffer.
`scopeD.py` adds protection for OWLv2's computed plain `box_bias`, absent from
`named_buffers()`, to the unchanged C21 scope and old box-only loss. It does not
alter model computations. Six-tensor checkpoints must not be loaded by the
historical twelve-head inference/adaptation loader. Production remains unchanged.
