# C19 numerical guard and recovery preflight — CPU only

No recovery runner, model forward, training, CUDA initialization or download is authorized or implemented here. The active proposal is `reports/phase10c19-v1/recovery-disabled-v2.json`; it is disabled. The earlier proposal is retained unchanged as superseded preparation history.

`guard19.compose_checked` is the tested replacement for C18's extra subtractive guard. It calls the exact immutable C16/C17 composition, tracks the separately accumulated source directly, verifies the six-tensor25%parameter-gradient cap with a derived floating-point allowance, and preserves returned gradient tensors bit-for-bit. It never reconstructs the negative contribution as `composed - base`. Genuine cap or component corruption still raises. Original C16/C18 source files are not patched.

`telemetry19.FailureTelemetry` must wrap the whole future executor in `failure_scope`, including startup, forwards, updates and checkpoint writes. Update photo/source/epoch/group/microbatch/committed/next-update context before each operation. Supply live model fingerprint, optimizer/checkpoint availability and runtime/memory callbacks. `compose_with_telemetry(acc, telemetry)` writes candidate gradient norms, cap/tolerance/direction and component hashes atomically before checking the group. It writes failures separately and re-raises; unavailable measurements remain explicit. No photo pixels or checkpoint overwrites occur. Provider/storage failures are surfaced; they cannot be guaranteed recoverable from a broken filesystem.

`recovery19` validates checkpoint28, exact parameter order/moments/scaler/RNG/data cursor, and restores actual heads/moments into a shape-correct CPU fixture without a forward or step. CUDA state is structurally checked only; GPU RNG restoration remains a future prerequisite. No scheduler was used. Reset gradients before the next group and restore RNG LAST after all model/state loading. Do not redo committed updates or initial probes.

`audit19` compares all28matched saved C8/C18 checkpoints without inference. Small actual update differences do not establish performance improvement. Completing the fixed endpoint has only conditional diagnostic value. Whole-epoch2 source-vector reporting cannot be reconstructed from missing pre-stop vectors; resumed diagnostics must label segments honestly.

CPU regression command (new synthetic telemetry receipts only):

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10c19-v1 -p 'test*.py' -v
```

Report/audit generators refuse overwrites. Do not delete outputs to rerun them. A future separately approved recovery must use a fresh versioned destination, bound executor, original optimizer objective and exact updates29–75. The452cumulative-forward allowance explicitly includes four failed uncommitted forwards; it needs approval rather than claiming the old448-forward allowance remains sufficient.
