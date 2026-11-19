# Phase 9: inference-only furniture precision experiments

The implementation is isolated from the application defaults and existing model,
dataset and experiment artifacts. Use the project `.venv/bin/python -B`. No
packages or weights are downloaded by these tools. Raw output is ignored under
`data/open-images/experiments/phase9-multiview-v1/`; the sanitised report is under
`reports/phase9-multiview-v1/`.

## Bounds and interfaces

The preflight protocol fixes15 exposed-development arms,18 previously exposed
photos, eight sequential views/model/photo, and at most288 development detector
passes. Actual model input sizes are768,960,1152; the two nonnative patch grids use
OWLv2 positional/box-bias interpolation. Four overlapping tiles and a horizontal
flip are computational augmentations, not independently captured views.

All parameters are frozen. The full36-photo final comparison is allowed only
after a checksummed selection and code manifest exist. Data loaders accept exact
nonfrozen rows; unknown/frozen membership fails before image opening. No official
ABO holdout or reserved frozen photograph is loaded.

Commands below are for the approved Phase9 run only, not generic installation or
permission for later experiments. Each source writer refuses incompatible or
existing artifacts. Do not delete outputs to make a rerun work.

```bash
# CPU checks; no models, downloads or inference.
.venv/bin/python -B -m unittest discover -s tools/phase9-v1 -p 'test_*.py' -v

# Resume exposure-stage detector cache; completed passes are hash-checked/skipped.
.venv/bin/python -B tools/phase9-v1/infer.py
.venv/bin/python -B tools/phase9-v1/siglip.py

# Analyze the fixed development recipes and lock a selection once.
.venv/bin/python -B tools/phase9-v1/analyze.py

# Only after selection is locked; at most one final comparison, no retuning.
.venv/bin/python -B tools/phase9-v1/infer.py --part final
# Run final SigLIP only if selection.json explicitly requires it.
.venv/bin/python -B tools/phase9-v1/analyze.py --part final
```

Per-pass records are atomically sealed with source/config/code hashes. A failed
or interrupted stage remains intact; missing stages can resume with the exact
same implementation. CUDA failure/OOM is explicit, without automatic CPU fallback
or resolution/threshold changes. Stop the foreground process with Ctrl-C, then
resume only the unchanged approved stage. A final stage cannot broaden its view
list beyond the locked recipe. GPU invocation bounds are30minutes,512MiB new
storage and75% allocator fraction with a6.5GiB allocated-memory stop target.

Precision/recall/AP compare published native partial-label annotations, not
independently verified room truth. Unmatched predictions become FPs only in an
FP-observable native scope. COCO completeness-unverified and filtered/NEX subsets
are not globally background. Classifier scores are relative, not calibrated
correctness probabilities. Box, class and disagreement diagnostics never alter
the original source labels or geometry.

`crossview.py` is an interface only until genuine same-scene/capture evidence and
object correspondences exist. It rejects cross-camera coordinate fusion without
verified registration. No new viewpoint generation is implemented.

Only source/tests and small sanitised protocol/metrics/report files are committed.
Photographs, model weights, galleries and raw annotations are not uploaded.
Return to the unchanged baseline with `git switch main` after committing work.
The baseline tag is `phase9-baseline-20261118`; ignored models/data require their
own independent backup, which Git does not supply.
