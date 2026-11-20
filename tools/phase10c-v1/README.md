Phase 10C contains CPU-only analysis and separately gated recovery preparation.
No GPU command was executed during this phase.

Use the project `.venv/bin/python`; all paths below are relative to the project.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c-v1 -p 'test_phase10c.py' -v
```

The completed seven-suite verification is recorded in
`reports/phase10c-v1/tests.json`. `verify.py` and `analyze.py` create exclusive
versioned outputs and intentionally refuse overwriting an existing closeout.
The historical Phase10B worker is unchanged and must not be restarted.

The exact unapproved inference scope is stored locally in
`data/open-images/preparation/phase10c-v1/inference-recovery-proposal.json`.
The neighboring `inference-authorization-template-v2.json` is explicitly
**not approved**. After
separate user authorization, record that authorization in a new versioned file,
retaining the exact proposal and current code hashes. Never alter the original
template or treat this README as approval. The CPU-tested command is:

```bash
timeout --signal=TERM --kill-after=30s 600s \
  .venv/bin/python -u -B tools/phase10c-v1/recover_validation.py infer \
  --authorization data/open-images/preparation/phase10c-v1/inference-authorization-approved-v1.json
```

This command currently refuses execution because no approved authorization file
exists. It authorizes at most 19 forwards of the saved LR3e-5 epoch-5 checkpoint
on exactly the old internal validation photographs, with zero optimizer calls.
It uses local model files only, FP32, at most 600 cumulative seconds, 6.5GiB
allocated VRAM and 128MiB new storage. The exclusive output destination is
`data/open-images/experiments/phase10c-recovery-v1/`.

Use `--resume` only with the same authorization and code. Successful receipts
are reused. An interrupted reserved forward or unreceipted raw tensor file
blocks automatic replay; inspect and request a revised recovery authorization
if needed. Do not delete the intent or any historical artifact. Touching `STOP`
inside this new destination requests a stop at the next image boundary; the
outer timeout also bounds a stuck inference. No automatic GPU retry occurs.

After all 19 raw records are complete, launch metrics in a separate CPU process:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B \
  tools/phase10c-v1/recover_validation.py score
```

Raw logits, square-coordinate boxes, restored boxes, maximum sigmoid scores
and category indices are atomically saved before NMS or metrics. The CPU scorer
verifies membership, source/model/code bindings, checksums, array types and
sizes; saves post-NMS predictions before scoring; and writes a new metrics file.
Matching, confidence floors, ties, AP sampling and partial-annotation scope are
regression-tested against the original implementation. This contains failures;
it is not proof that the underlying interpreter fault has been fixed.

`box_only.py` provides the tested six-tensor scope, strict optimizer factory and
positive-only box loss for a future separately authorized experiment. It does
not run a training queue. The proposed bounds and predeclared gates are in
`reports/phase10c-v1/box-only-config.json`. The old queue assumes 12 tensors and
must not consume these six-tensor checkpoints; integration of a new guarded
worker and its zero-update GPU smoke test is required before parameter updates.
