# Completed Phase 5C head-only experiment

Artifacts and report: `data/open-images/experiments/phase5c-v1/`.
Exactly two approved fits have finished at 70 optimizer updates each. No more
optimization is authorized by this run. Frozen evaluation remains unused.

Read `report.md`, `final-verification.json`, `test-summary-final.json` and
`selected-checkpoint-registry.json` in the artifact directory. The selected
checkpoint is a head delta requiring the pinned original OWLv2 base, not a
standalone Hugging Face model directory. `compare.py` demonstrates loading the
base, calling `configure(model, 'heads', False)` and loading the head delta.

All commands use the workspace `.venv/bin/python -B` and local model files.

Monitoring (read only):

```bash
cat data/open-images/experiments/phase5c-v1/update-budget.json
cat data/open-images/experiments/phase5c-v1/heads-lr3e-5/complete.json
tail -n 15 data/open-images/experiments/phase5c-v1/heads-lr3e-5.console.log
```

Recovery for a genuinely incomplete authorized fit only was:

```bash
.venv/bin/python -B tools/phase5c-v1/worker.py --fit-id heads-lr1e-5 --resume
.venv/bin/python -B tools/phase5c-v1/worker.py --fit-id heads-lr3e-5 --resume
```

Completed fits skip without loading the GPU or making optimizer calls. Never
delete completion records, ledger events or receipts to force a rerun. A reserved
optimizer call without a committed matching receipt is deliberately blocked for
manual reconciliation rather than potentially exceeding the approved budget.
SIGINT/SIGTERM and a per-fit `STOP` file request a checkpointed graceful stop.
CUDA OOM is fatal and preserves the previous committed checkpoint; it does not
automatically change the approved batch or adaptation scope. FP16-only gradient
overflow supports bounded same-group, RNG-preserving loss-scale replay before an
optimizer call; persistent nonfinite gradients and all nonfinite losses remain
fatal. No extra optimizer calls are hidden by a GradScaler skip.

Tests (separate process per suite avoids old/new module-name collisions):

```bash
.venv/bin/python -B -m unittest discover -s tools/phase5-v1 -p 'test*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase5-metadata-v1 -p 'test*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase5-images-v1 -p 'test*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase5b-v1 -p 'test*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase5c-v1 -p 'test*.py' -v
```

All artifact creation entrypoints require fresh destinations. Verification
programs also preserve previous verification records rather than overwriting
them. Use the saved final audit for this completed run. No script accesses Mac
simulation sources or requests downloads; no official Phase 2B scoring is used.
