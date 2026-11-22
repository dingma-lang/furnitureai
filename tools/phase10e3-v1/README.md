Phase 10E.3 is CPU-only preparation. No training, detector inference, downloads,
environment changes or GPU recovery runs are authorized here. All prior code is
immutable, including the reproducible E2 `loss * 128` VJP implementation.

The opt-in `Capture(head, enabled=True)` gathers an ephemeral head snapshot from
one separately authorized normal forward. It verifies bit-identical normalized
projection and final logits by repeating the pinned head expressions in the same
autocast context. One additional head-only similarity contraction is counted
separately in the recovery proposal. No photographs or activations are saved.

`observed_logits(..., enabled=True)` supplies the *exact observed* forward logits
with an FP32 widened adjoint. It is a diagnostic custom autograd operation, not
a replacement forward or an automatically installed training loss. It preserves
the MSE, masks, query indices, parameter scope and per-photo normalization. It
does not reproduce FP16 backward rounding, underflow or overflow. This distinction
must accompany any reported projection norms; historical E2 FP16 norms must not
be mixed with the repaired positive/KD norm ratios.

For the observed dense projection d, saved norm n, denominator h=n+1e-6,
actual autocast-contraction query q, and frozen final scale s:

    dc = dz * s
    du = dc @ q
    dn = -sum(du*d) / h**2
    dd = du/h + dn*d/n       (norm subgradient zero when n=0)
    dW = dd.T @ x_cast; db = sum(dd)

All adjoint arithmetic, contractions and parameter gradients above are FP32;
no manual loss scaling is necessary. The actual rounded n/h/q/d/s are retained,
rather than recomputing a different FP32 forward. Division and norm derivatives
are evaluated at those observed nodes. Frozen calibration affects dz*s but never
receives gradients. Shared x/query and box outputs are not differentiable inputs
to this measurement operation. `projection-input-unrouted` telemetry describes
the hypothetical input sensitivity and never sends it into the shared backbone.

CPU tests use the actual installed Transformers classification head with tiny
synthetic inputs, independently constructed FP64 autograd references at observed
nodes, and deliberate FP16 overflow examples. These are not actual CUDA tests,
nor reproductions of the missing E2 intermediate. No numerical tolerance is
relaxed to accept nonfinite gradients. Full FP32 detector inference is unnecessary
for this projection-only diagnostic because all shared representations are frozen;
this does not validate subsequent legacy mixed-precision training.

`Trace` is disabled by default. Optional tensor/node hooks observe rather than
modify gradients. Statistics test original-dtype finiteness before widening
finite values for reductions. Events and exceptions are written atomically.
The earliest observed nonfinite hook is not necessarily the kernel that first
created it. `components3.py` persists the positive component before KD, records
unit loss/deltas/terminal derivatives/source identity/masks, and fails closed.
Partial output cannot be overwritten or automatically retried. The eventual
approved launcher must additionally enforce the proposal's *global* resource
caps, operation-intent counters and watchdog; per-record atomic limits are only
a secondary guard, not a substitute for that launcher.

Run CPU checks with the unchanged project environment:

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e3-v1/verify3.py --tests
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e3-v1/recovery3.py --dry-run
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e3-v1/verify3.py --check

The recovery proposal remains disabled and its coefficient remains null. There
is deliberately no executable GPU launcher in this CPU-only phase. One separately
approved recovery may bind this tested component to the bounded E2-style launcher;
it must not reuse/overwrite the failed E2 directory. Training requires its own
approval and a pinned, CPU-tested precision integration; diagnostic success alone
cannot authorize or certify the existing FP16 training path.
