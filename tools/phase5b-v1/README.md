# Phase 5B Windows/WSL detection research

Use `.venv/bin/python -B` from `/home/tristary/furniture-ai-ml`. This stage ran original OWLv2 inference on **18 eligible development photographs**, plus three **no-optimizer** memory probes on one training photograph. No frozen-evaluation pixels, predictions or metrics were loaded. Opaque byte hashes of all acquired photos were verified as requested. No download, package change, production modification or parameter update occurred.

Artifacts: `data/open-images/experiments/phase5b-v1/`. Read `report.md` and open `error-index.html` or `original-development-v1/gallery.html`. The immutable dataset lock contains 27 train/18 development rows and only identity/hash references for 18 provisional frozen photos; all original 60/30/60 membership slots remain in the historical source manifests. Unsafe/unresolved examples remain excluded. The 60 frozen slots are never used for fitting, tuning, probing or benchmark metrics here.

## Actual development benchmark

Fixed original model and preprocessing, seven `a photo of a {category}` prompts, class-wise NMS .5. AP50/AP75 use a prespecified score floor .001 and global per-image cap 300; operating P/R use .15. Scores use the installed OWLv2 sigmoid/max-query postprocessor, not classifier scores or objectness multiplication. Class/image pairs without published coverage, or positive labels without native boxes, are unscored. Group/depiction regions are ignored for unmatched detections; this differs from official Open Images group-credit/hierarchy-expanded AP. All original boxes/MIDs are preserved. Metrics measure native annotation agreement, not independent real-world accuracy.

Completed inference can be verified/resumed without inference or overwriting results:

```bash
.venv/bin/python -B tools/phase5b-v1/baseline.py --resume
```

The completed dataset lock, binding and all stage checksums must match. Interrupted completed stages are reused; corrupt/incomplete stages stop safely rather than being overwritten. CUDA/model failures propagate without fake empty predictions or silent CPU fallback. Use a new versioned output root if code/input bindings change; preserve prior artifacts.

Tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase5b-v1 -p 'test_*.py' -v
```

The completed test log also includes existing Phase 5 suites. Python's `queue` module is not shadowed: the launcher is deliberately named `run_queue.py`.

## Future bounded training — SEPARATE APPROVAL REQUIRED

**Do not run these update commands until the user approves detector parameter updates.** Preparation itself is not training authorization. The runner refuses updates without the explicit flag. Initial recommended queue: two **head-only** configurations, learning rates 1e-5 and 3e-5, ten epochs maximum, batch one, accumulation four, at most **70 optimizer updates per fit / 140 total**, weight decay .01, mixed FP16, seed 20261105. Backbone, text encoder and objectness stay frozen. Actual train pool is 27 photos / 75 ordinary native boxes; development is 18 photos / 41 ordinary boxes. Full-model adaptation is not proposed.

Cumulative active-time bound is 1,800 seconds per fit across resume; the queue also enforces a 1,800-second worker wall-time target with up to 60 seconds of graceful-stop allowance. Two fits therefore have a documented maximum of roughly 62 minutes including stop grace, plus minimal orchestration. The allocator fraction is capped at .80 of GPU VRAM. Per-fit allocated checkpoint storage is capped at 2 GiB, approximately 4 GiB for the queue. No historical checkpoint is deleted to satisfy limits.

The focal classification loss masks unknown classes and ignored/padded regions; Hungarian assignment combines class score, L1 and generalized IoU. Native boxes are transformed into OWLv2's **bottom/right padded-square frame** at runtime, without modifying source boxes. Objectness background loss is disabled because these images lack exhaustive all-object labels. A small CPU regression suite and actual GPU backward probes validate components; a real optimizer-step smoke check remains necessary after approval. There are no adapted-model results yet.

Before approval, resolve or explicitly accept native Desk and Cabinetry scope/annotation granularity risks described in `report.md`. A micro-experiment can measure published-label fitting; it cannot establish better real-room accuracy with this tiny provisional pool. The original seven-class baseline remains unchanged regardless of future scope choices.

After separate approval, start from a **fresh** queue directory in the GPU-visible Ubuntu terminal:

```bash
cd /home/tristary/furniture-ai-ml
nohup .venv/bin/python -B tools/phase5b-v1/run_queue.py \
  --out-parent data/open-images/experiments/phase5b-heads-queue-v1 \
  --allow-parameter-updates \
  >> data/open-images/experiments/phase5b-heads-queue-v1.console.log 2>&1 &
```

Monitor:

```bash
tail -f data/open-images/experiments/phase5b-heads-queue-v1.console.log
```

Per-step immutable `update-*.json`, per-epoch validation JSON and committed checkpoint receipts provide progress. `leaderboard.json` is written on queue pause/completion. Best checkpoint selection uses development native macro AP50 only; no frozen predictions are requested.

Graceful stop after the current accumulation group and durable snapshot:

```bash
touch data/open-images/experiments/phase5b-heads-queue-v1/STOP
```

Resume, removing only the explicitly created transient control flag:

```bash
rm -f -- data/open-images/experiments/phase5b-heads-queue-v1/STOP
nohup .venv/bin/python -B tools/phase5b-v1/run_queue.py \
  --out-parent data/open-images/experiments/phase5b-heads-queue-v1 \
  --resume --allow-parameter-updates \
  >> data/open-images/experiments/phase5b-heads-queue-v1.console.log 2>&1 &
```

Completed fits are skipped only after binding/best-checkpoint hash verification. Snapshot files have unique names and SHA commit receipts; orphan/uncommitted files are preserved and ignored. Recovery restores delta weights, optimizer/scaler, exact epoch/group/update counters, cumulative time and Python/Torch/CUDA RNG. A damaged committed checkpoint stops; it is not silently replaced. A CUDA failure stops and leaves committed state; no automatic batch/label/precision change or CPU fallback occurs. Bindings protect the dataset, source code, base weights and all processor/model files. Resume cannot silently use changed packages/checkpoint inputs. To change configs or code, use a fresh experiment and record the change.

Snapshots contain **only trainable detector deltas plus recovery state**, not replacement pretrained directories. Reconstruction uses the original pinned local model, `adaptation.configure(model,'heads')`, then `adaptation.load_delta(model, checkpoint['weights'])`, after checking binding/base SHA and shapes. Never copy these into the original OWLv2 folder or production path.

Last-layer unfreezing and native rank-4 Q/V LoRA were probed successfully but are not enabled by the initial training config validator. Their similar activation footprint does not justify using them before the head-only/data-quality experiments. Any later expansion of update modes/bounds needs separate approval.
