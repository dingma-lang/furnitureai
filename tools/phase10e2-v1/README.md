# Phase10E.2 — one zero-update gradient verification

Authorized destination:
`data/open-images/experiments/phase10e-gradient-smoke-v1/`.
One run only:28 forwards,at most48 scalar-loss VJPs,180 seconds,6.5 GiB
allocated VRAM,128MiB new diagnostics. No optimizer,backward accumulation,
weight update,new loss variant,download,training or holdout scoring.

State A replays the exact first four E1 original-cache photos,measures actual
positive projection gradients,and proves/measures the MSE derivative. State B
loads the immutable existing C11 E5 checkpoint as a diagnostic reference and
replays the exact24 E1 training photos. This in-memory checkpoint switch is
intentional;before/after fingerprints must match within each state. Original
and reference files remain immutable.

One positive-loss VJP is measured for each of28 forwards. A distillation VJP is
measured only when its selected output derivative is nonzero. Empty-mask or
bit-identical-target MSE derivatives are exactly zero by the formula;these are
explicitly labelled analytic zeros,not fabricated executed GPU measurements.
The hard48-call guard remains authoritative;missing measurements are not filled.

FP16 replay uses the historical C11 `observe11.compare_array_sets(probe=True)`
tolerances fixed before execution:logits atol0.02/rtol0.001;normalized square
boxes atol/rtol1e-5. Shape,query layout,source hashes and processed pixel/token
hashes must match exactly. There is no query realignment or tolerance expansion.

`prepare2.py` verifies all59 teacher references and both checkpoint states on
CPU. `verify2.py` runs the CPU tests. `launch2.py` permits exactly one bounded
GPU attempt. `analyze2.py` consumes its immutable outputs on CPU. A later
separately approved training configuration,if justified,is always disabled.

CPU-only regression checks:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10e2-v1 -p 'test_gradient2.py' -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e2-v1/verify2.py --check
```

Do not rerun completed preparation,launch or analysis generators in place.
Outputs are atomic and no-replace. A STOP/failure is terminal;do not delete
receipts or automatically retry. Model weights,photos,raw query outputs and
gradient details remain local. Only code,configurations and compact reports
are committed to the experimental branch.
