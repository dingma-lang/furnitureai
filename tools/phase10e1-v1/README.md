# Phase 10E.1 — disabled selective teacher distillation

CPU preparation only. No GPU run, downloads, new labels, model inference or
saved-model updates are authorized. The default workers stop before CUDA/model
construction/output-directory creation without a separate exact approval file.
The active code/data contract is `input-binding-v2.json`. The initial contract
and test receipts are retained as superseded snapshots after a report-generator
syntax correction; the objective and audited dataset did not change.

One method is implemented: mean squared **Chair-logit** consistency inside
previously source-supported visible negative regions. Both original-teacher and
student boxes must fit inside the same grant, avoid protected/ignored furniture
and matched positives, and have trustworthy image/prompt/grid correspondence.
Teacher Chair scores at or above the existing 0.15 operating threshold conflict
with the grant and are excluded. Six other categories remain unknown. Teacher
low confidence without independent source evidence never creates a negative.

The base objective stays identical to C11/C8 corrected classification plus
5×L1+2×GIoU. No new BCE, balancing, projection regularization or sampling variant
was added. Calibration remains frozen as in the closest C11 control; the two
classification projection tensors and six box tensors remain trainable. The
teacher uses immutable cached outputs and has no optimizer or gradients.

Existing teacher coverage is 59 images/56 known source groups. There are 278
admissible original-query targets on 15 images/groups. Only 19 of 35 admitted
regions have actual cached query coverage. This is not sufficient evidence of
broad seven-class confidence control. The coefficient is **unset**; training is
not ready for approval before real projection-gradient measurements.

Recheck CPU tests and preservation seal:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10e1-v1 -p 'test_distillationE.py' -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e1-v1/verifyE.py --check
```

Preparation order was `auditE.py`, `configE.py`, `prepareE.py`, CPU verification,
report generation and sealing. Generators are write-once. Do not delete or
overwrite outputs to rerun them. Reconstructed pixel/token hashes are today's
CPU hashes, not historical measurements. Initial GPU equality replay is required
before future training; a mismatch stops rather than silently rebuilding caches.

`launchE.py` supervises a separately approved future job with a hard deadline and
one durable launch token. `workerE.py` implements the zero-update gradient smoke
and single training arm; neither has run. Smoke would load the existing C11 E5
head checkpoint only as a diagnostic gradient reference after four original
replays, with no optimizer. Actual future training always starts from original.
No coefficient is automatically selected by the smoke or evaluator.

The training worker preserves deterministic order, unchanged matching/losses,
AMP finite-gradient checks, scoped optimizer, fixed coefficient, atomic epoch
checkpoints, RNG/cursor state, separate failure receipts and final raw predictions
before `evaluateE.py` CPU metrics. No intermediate epoch scoring or adaptive
selection is allowed. Failed/interrupted runs are terminal for their approval;
resume, cache repair and any new experiment require separate authorization.

Code and small reports go to Git. Teacher arrays, source photographs, processor
tensor fingerprints, training records and future weights stay local under data.
Existing Phase 2B review gates and every historical checkpoint remain unchanged.
