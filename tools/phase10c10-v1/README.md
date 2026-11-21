# Phase 10C.10: calibration attribution and disabled training preflight

Use `/home/tristary/furniture-ai-ml/.venv/bin/python -B`. This phase executes
CPU parameter/activation-availability analysis and synthetic tests only. It has
not run a real-photo detector forward, GPU training or new downloads.

The final candidate is **`reports/phase10c10-v1/experiment-disabled-v3.json`**.
`training_authorized`, source-mask activation and Phase10D authorization are
false. Earlier disabled revisions are retained: their experiment semantics are
identical; only implementation/test code pins were refreshed during CPU work.
These are not additional training variants.

Implemented components:

- `common10.py`: fresh C10 source contract and old C8/C9 seal verification.
- `audit10.py`: original/A/B tensor values, shapes, hashes, full parameter
  differences, optimizer participation/steps/moments and recorded gradient norms.
  Exact1538-scalar calibration table remains private in the preparation folder.
- `shared10.py`: distinguish frozen `owlv2.logit_scale` contrastive temperature
  from the two trainable detection-calibration Linear modules and frozen norms.
- `attribution10.py`: real-arithmetic LayerNorm bounds, no fabricated feature
  reconstruction; actual-feature CPU head counterfactuals prepared for later.
- `freeze10.py`: persistent eight-tensor scope, strict calibration `.grad is None`,
  original-byte checks, optimizer exclusion, full12-head checkpoint compatibility.
- `config10.py`, `prepare10.py`: one deterministic, disabled experiment. Same C8 B
  data/loss/order/optimizer, with only four calibration tensors removed.
- `launch10.py`, `future_worker10.py`: prepared but **not executed** GPU path.
  Launch authorization is checked before worker creation or CUDA. Every training,
  evaluation and probe transition uses `FrozenScope`, never legacy
  `configure_heads`, which would re-enable the four frozen tensors.
- `future_diagnostics10.py`: a separate CPU consumer requiring a completed
  approved run. Saves predictions before metrics; evaluates the same exposed19
  internal photos. Functional attribution requires genuine cached pre-head/text
  vectors, checksum/source/frozen-backbone proof and endpoint reconstruction.
- `test_preflight10.py`, `verify10.py`: CPU synthetic optimizer and source tests,
  including the complete75-step safety budget, and historical preservation.

Current phase preparation paths:

```
data/open-images/preparation/phase10c10-v1/
reports/phase10c10-v1/
```

The proposed training destination is
`data/open-images/experiments/phase10c10-calibration-frozen-v1/`.
It **does not exist** after preflight. No new checkpoint or training directory
was created. No launch command in this phase has approval to create it.

CPU test command:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c10-v1 -p 'test*.py' -v
```

The following guard check intentionally refuses execution, before CUDA or output
creation (it is not a training command that can proceed with this config):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c10-v1/launch10.py \
  --config reports/phase10c10-v1/experiment-disabled-v3.json
```

After separate explicit user approval, a **new versioned authorized config** and
approval record would be needed; never edit the disabled preflight artifacts.
The record must contain explicit user reference,75-update limit, canonical
`config_semantic_sha256` and exact file-byte `config_file_sha256`. No approval
record has been created. Flipping a flag alone is insufficient.

The prepared parent launcher enforces600seconds by process deadline. The worker
enforces75 updates /5epochs,467 total forwards,6.5GiB VRAM, source hashes,
nonfinite failure, immutable snapshots and a4GiB output guard. `STOP` requests
a safe stop. There is no automatic replay or extra variant. CUDA/AMP performance
is not validated by this CPU phase; all actual runtime checks remain conditional
on the separately approved initial probe.

Why467 prepared forwards:295 optimization inputs,96 unchanged training-source
probe inputs and76 FP32 diagnostic inputs (19 at0/1/3/5). The original reference
is rechecked at initialization in this standalone run; C8 B previously reused A's
reference and therefore had448 forwards. The19 initial forwards can also save
pre-head/text features without extra encoder calls. No features are available
now. Proposed cache storage is approximately211MB; all files remain private.

Checkpoints store all12 head tensors for compatibility with the original
`expanded.load_delta` inference interface. Training restore additionally checks
original calibration, frozen identities, eight named optimizer parameters,
hyperparameters, finite moment shapes and step cursors. Old adapted C8 heads
cannot masquerade as original C10 initialization. A future recovery needs a
remaining-budget audit; existing outputs are never deleted or replayed blindly.

Freezing these modules is a mechanistic ablation, not a demonstrated repair.
The projection remains able to inflate similar furniture logits. Existing
development labels are incomplete and cannot establish independent seven-class
80%precision/90%recall acceptance.
