# Furniture AI — remote, train-only automated experiment runner (v1)

**Stage:** candidate training / internal development only.  This package never runs the official Phase 2B holdout evaluation. Its development metrics are from folds of the *original training products*, and are **not unbiased production generalization estimates**. The ABO training labels remain provisional/weak.

## Supported target

- `/home/tristary/furniture-ai-ml` on Windows laptop Ubuntu WSL, PyTorch CUDA and verified Phase 2A dataset/cache.
- Uses existing, untouched `data/abo/experiments/phase2a-v1/experiment.py` strictly for verified **train-only** embedding loading.
- Expects 275 × 768 training embeddings over 130 products. Never reads validation/test feature matrices or reviewed Phase 2B CSVs.
- Checks known ID/family/decoded-image-hash overlaps against existing partition *metadata* (not holdout features/labels). Unknown physical aliases still need independent review.
- New outputs in `data/abo/experiments/auto-training-v1/` only. Original OWLv2, SigLIP2, Phase 2A checkpoints and Phase 2B reviews remain unchanged.

## What is trained

8 fixed linear/MLP head configurations, each on 3 product- and known-family-group-isolated internal folds. Max 65 epochs/fold, with patience 12. One training image per product per epoch. Per-fold validation product logits aggregate over available images. Validation here means **training-derived development fold**, not official proposed validation/test partitions.

Each fold checkpoints after every epoch; a stopped process restarts from the last committed epoch. The same configuration is never accidentally overwritten with different source/code fingerprints. Configuration failures are saved and excluded from automatic final selection. After all 8 succeed, the best mean development product macro-F1 configuration is refit using **all training** products, with number of epochs based on median best-fold epoch. That checkpoint is still **untested on official holdout**.

## Install from Mac (after downloading the zip)

**Local Mac terminal:**

```bash
scp ~/Downloads/furniture-auto-train-agent-v1.zip furniture-ai-rtx:
ssh furniture-ai-rtx
```

**Windows SSH prompt:**

```powershell
wsl -d Ubuntu
```

**Ubuntu prompt:**

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python - <<'PY'
from pathlib import Path
import zipfile
archive = Path('/mnt/c/Users/Ruoling Ma/furniture-auto-train-agent-v1.zip')
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        dest = Path(info.filename)
        if dest.is_absolute() or '..' in dest.parts or not info.filename.startswith('tools/auto-training-v1/'):
            raise ValueError('Unsafe archive entry: ' + info.filename)
        if Path(info.filename).exists():
            raise FileExistsError('Refusing overwrite: ' + info.filename)
    z.extractall('.')
print('Installed in tools/auto-training-v1 without overwriting existing files')
PY
```

## Test and review the plan (does not launch long training)

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B tools/auto-training-v1/test_auto_train.py
.venv/bin/python -B tools/auto-training-v1/auto_train.py check
.venv/bin/python -B tools/auto-training-v1/auto_train.py plan
```

Set standard offline environment and run a 3-epoch GPU smoke fold. **This does run a small TRAIN-only GPU experiment**, creating isolated smoke artifacts, but not the long batch:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -u -B tools/auto-training-v1/auto_train.py smoke
```

The long run is intentionally gated by an existing successful smoke whose source-code and feature fingerprints match. Inspect `data/abo/experiments/auto-training-v1/smoke-gate.json`.

## Background training AFTER explicit user approval

```bash
cd /home/tristary/furniture-ai-ml
mkdir -p data/abo/experiments/auto-training-v1
# Only if not already running:
tmux new-session -d -s furniture-training "bash -lc 'set -o pipefail; cd /home/tristary/furniture-ai-ml; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; .venv/bin/python -u -B tools/auto-training-v1/auto_train.py run 2>&1 | tee -a data/abo/experiments/auto-training-v1/training.log'"
```

Useful status/monitoring commands:

```bash
.venv/bin/python -B tools/auto-training-v1/auto_train.py status
cat data/abo/experiments/auto-training-v1/leaderboard.csv
tail -f data/abo/experiments/auto-training-v1/training.log
tmux ls
```

To gracefully stop between configurations:

```bash
.venv/bin/python -B tools/auto-training-v1/auto_train.py stop-after-current
```

To continue after shutdown/interruption:

```bash
.venv/bin/python -B tools/auto-training-v1/auto_train.py clear-stop
# Relaunch the exact same tmux command shown above, if no training agent is currently running.
```

A failure.json is never silently cleared or retried. Inspect failure and logs, resolve its cause, and deliberately restore/restart the affected configuration after backup. No arbitrary self-modifying/code-writing autonomous agent is included.

## Important limits and safety

- No Windows sleep prevention or automatic Windows boot/restart task. Closing Mac SSH is fine when running in tmux, but **Windows sleep/hibernation or WSL shutdown interrupts GPU work**. Restart via command when awake.
- `torch.cuda` is mandatory for GPU smoke/run; no CPU fallback. Uses a 65% PyTorch per-process CUDA allocation cap, not a guaranteed 65%-of-device utilization/temperature limit. Current thermal behavior is subject to Windows firmware/drivers; watch `nvidia-smi`.
- No new dependencies or pretrained weights are downloaded or installed by this package.
- Only 275 weakly labeled training images are available; no amount of repeating this tiny dataset can guarantee a robust detector or improved real-room generalization.
- Grouping detects *known* product/family, capture and exact-pixel links. An unresolved physical identity shared across partitions may still exist. Audited Phase 2B product identity and training overlap remain required for independent holdout claims.
- Fixed 8-config CV ranking is a **development search**. It should not be evaluated repeatedly on the official held-out test set to tune it.
- Fine-tunes only classification heads on frozen SigLIP 2 embeddings. Does **not** fine-tune SigLIP 2 encoder or OWLv2 detection. Detection needs verified boxes, and segmentation needs masks.
- The local/offline tests validate data isolation and serializers; a real CUDA smoke run **must** be performed on the Windows machine before long training.
