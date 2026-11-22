# Phase 10C.21 — CPU preparation only

This branch audits existing evidence and prepares one disabled experiment. It
contains no GPU runner or downloader. It neither continues C20 nor activates
Phase 10D. Historical implementations, datasets and checkpoints are inputs only.

Read `reports/phase10c21-v1/report.md` and `box-only-disabled.json` before requesting
execution. The new box scope adapter is **not** compatible with legacy workers
that expect twelve trainable detector-head tensors. A separately authorized
executor must use the six-tensor adapter, enforce the approved counters and save
predictions before separate CPU metric calculation. Do not run a C20 worker with
the new configuration.

CPU checks:

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10c21-v1 -p 'test_preparation21.py' -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c21-v1/verify21.py --check-seal
```

`verify21.py` without flags creates new write-once test/preservation artifacts and
therefore is a one-time build step. Repeating it in the completed directory stops
at an existing artifact; use the commands above for subsequent verification.
Never remove artifacts merely to rerun a generator.

Fresh preparation order was `common21`, `native21`, `teacher21`, `full_native21`,
`negative21`, `plates21`, actual inspection of both contact sheets, `review21`,
`contexts21`, `strategy21`, `evaluation21`, `acquisition21`. The explicit visual
observations in `review21` apply only to this exact immutable source/candidate
bundle. They are assistant observations, not reusable automatic approvals or
human reviews. New images/candidates require new evidence. An additional
metadata-only isolation registry v2 includes historical photograph hashes,
catalogue products and frozen references; it supersedes v1 for future intake.

Outputs are under `data/open-images/preparation/phase10c21-v1/` and
`reports/phase10c21-v1/`. Data, photographs, contact sheets, cached predictions,
raw rights records and model weights stay outside Git. The public SHA256SUMS-v2
binds reports; `artifact-seal-v2.json` binds code and a private hash manifest.
The first seal snapshot is retained, superseded after fixing its report-relative
path format; the final v2 checksum list uses repository-relative paths.
Pending hard links remain retained for recovery and are included in conservative
storage accounting.

The independent evaluation intake validates actual human completeness evidence,
individual rights files, exact identities and all-seven-category review. It does
not annotate images or create reviews. `verify_files=False` is only a synthetic
test facility, explicitly marked in its return value; actual intake must verify
files. No new independent images have qualified.

Rollback consists of stopping the future worker, preserving its receipts and
remaining on the original model. Production defaults were not changed. Existing
historical branches and artifacts remain usable without this directory. Do not
delete or replace checkpoints to roll back an experiment.
