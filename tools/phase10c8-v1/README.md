# Controlled Chair-negative A/B diagnostic

This phase authorizes exactly two arms, each five epochs / 75 AdamW updates,
from the immutable original OWLv2. The only intentional difference is the
source-supported Chair-negative mask. These are internal partial-label
development diagnostics, not independent furniture-recognition accuracy.

Use the existing project `.venv/bin/python -B`. No installations or downloads.

- `prepare8.py`: freeze fresh output, input hashes, two configurations and
  evaluation protocol before training. Requires the experimental branch.
- `launch8.py`: sequential workers with one-hour combined deadline. A must
  complete and pass its checkpoint checks before B starts. No automatic retry.
- `worker8.py`: frozen backbone, head-only optimizer, batch one / accumulation
  four, FP16 autocast with FP32 loss, explicit nonfinite checks, memory guards,
  source checks and unique snapshots after every optimizer update.
- `metrics8.py`: separate CPU process, unchanged recovery metrics. GPU-derived
  FP32 evaluation scores/boxes are saved before metrics. No model execution.
- `verify8.py`: CPU suites and historical byte verification.
- `summarize8.py`: CPU artifact analysis, checkpoint audit, CSVs and curves.

Current output is `data/open-images/experiments/phase10c8-ab-v2/`.
The v1 attempt failed during Python imports: zero forwards, zero updates.
It remains preserved. A standalone-import regression test covers that error.

Monitoring:

```bash
cd /home/tristary/furniture-ai-ml
tail -f data/open-images/experiments/phase10c8-ab-v2/A-control.log
tail -f data/open-images/experiments/phase10c8-ab-v2/B-corrected.log
nvidia-smi
```

Creating `data/open-images/experiments/phase10c8-ab-v2/STOP` requests a stop
at the next safety boundary. This does not authorize another training run.
Each applied update has an intent file, immutable checkpoint, optimizer/scaler
state, RNG state, cursor and SHA-256 receipt. Recovery evidence is saved at
every update. **No automatic replay** is permitted after an uncertain update,
nonfinite event, resource failure or integrity failure. The current launcher
deliberately refuses an existing queue. A resumption would need its own audited
remaining-budget authorization; completed arms must never be rerun.

Postprocessing of an already completed queue (run once into fresh result paths):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c8-v1/metrics8.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c8-v1/summarize8.py
```

Never replace production weights with these head deltas. Loading a diagnostic
checkpoint requires the exact original model, `expanded.configure_heads`, then
`expanded.load_delta(model, payload)`. Check its receipt and binding first.

All 59 optimization photographs / 182 positives are unchanged. Only the 19
existing training-derived development photographs can enter diagnostic scoring.
Stools used for seating and bar stools belong to Chair; footstools, ottomans
and step stools stay separate. Source evidence is assistant-inspected native
object-mask evidence, not fabricated human confirmation. Unknown image regions
and unsupported classes remain ignored. Training and gradient probes use only
the optimization partition; probes use `autograd.grad`, leave `.grad` empty
and verify unchanged weights. Control-arm added gradients are counterfactual
measurements, never optimizer contributions.

The private directory contains original photographs' IDs, raw predictions,
checkpoints, detailed masks and logs. Only small code and aggregate reports
are intended for GitHub. No dataset photographs or model binaries are committed.
