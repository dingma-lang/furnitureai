# Phase 10E.3 — distillation numerics and failure telemetry

CPU preparation completed on 2026-11-22. **No GPU inference, GPU backward, model
training, checkpoint updates, downloads or environment changes occurred.** The
recovery proposal is disabled; the distillation coefficient remains unset.

The repaired measurement method is validated on synthetic CPU examples. A finite
nonzero adapted-student CUDA projection gradient is **not yet established**.

The E2 failure is confirmed at the returned scaled VJP, not at a particular kernel.
Its first active term had teacher/student Chair logits -15.7265625/-9.546875,
unit MSE 38.18853759765625, unscaled terminal derivative 12.359375, and scaled
derivative 1582.0. These were finite. At least one requested upstream gradient
was nonfinite. The missing intermediate cannot be recovered from saved logits.
Mixed-precision overflow remains a hypothesis; the synthetic reproduction here
demonstrates a mechanism, not the exact original failure.

| Stage | Verified implementation / precision |
|---|---|
| Original teacher | Historical FP16 CUDA forward; stored NPZ arrays FP32 |
| Master model parameters | Original418 tensors F32; all12 C11 head tensors FP32, verified CPU-side |
| Detector forward | E2 explicitly uses CUDA autocast FP16, not BF16 |
| Classification projection and cosine contraction | CUDA autocast policy permits FP16 linear/einsum; FP32 master weights are cast for these operations |
| Norm and division | Vector norm has CUDA FP32 optional-dtype policy; division follows actual input promotion. Historical intermediate dtypes were not saved |
| Frozen calibration | Shift/scale linear operations are low-precision policy operations; ELU/add/multiply inherit operands |
| Returned logits | Head explicitly casts masked-query result to FP32; E2 `.float()` does not widen earlier graph nodes |
| MSE | FP32 subtract, square and selected-term mean |
| E2 VJP | Manual `loss*128`, finite test, then returned gradient division by128; no GradScaler |
| E1 future training | Separate real GradScaler initialized128; remains disabled and unmodified |

The legacy path runs backward from FP32 loss through the final logit cast,
low-precision calibration arithmetic and contraction, normalization/casts, and
the projection's low-precision adjoint before reaching FP32 master gradients.
The precise dtype at each historical backward intermediate is not a saved fact.
The installed Transformers head, PyTorch autocast policy/header, generated norm
backward declarations, GradScaler implementation and compiled dispatch tables
were inspected and hashed in `dtype-audit.json`. Actual CPU-head forward/backward
hook events are included there and explicitly labelled CPU. No CUDA BF16 or
full-FP32 detector trial was performed.

Scale128 was an instrumentation multiplier matching the historical initial AMP
scale, not a distillation coefficient. It can protect tiny gradients against
underflow but can also overflow a low-precision adjoint. Dividing an infinite
gradient afterwards cannot restore it. GradScaler's optimizer-based inf checks
were not present in E2's `autograd.grad` call; simply adding a scaler or lowering
the scale would not establish a safe path.

The opt-in repair in `stable3.py` captures the live head inputs, rounded dense
projection, saved norm/denominator, contraction query and frozen calibration scale.
It requires bit-identical reconstruction of the original normalized projection
and final logits. A custom diagnostic VJP evaluates the same mathematical chain
rule at these observed values entirely in FP32, with scale1. It never sends a
half-precision adjoint through the upstream head. This is **not bit-equivalent
to rounded FP16 backward**, nor a new FP32 forward; both positive and unit-KD
measurements must use this same path for meaningful norm/direction comparisons.
Legacy E2 positive norms cannot be combined with repaired KD norms.

The output MSE, forward logits, query ordering, masks and per-photo mean are
unchanged. Frozen calibration/shared representations and box branches receive
no diagnostic classification gradients. Head tensor identity/version/scope is
checked so a stale snapshot cannot be used after parameter changes. This path
is disabled by default and is not installed into production or training.

Optional `Trace` hooks record original dtype/finiteness before finite-value
FP64 statistical reductions. They include forward values, typed returned VJP
slots, node/tensor adjoints, observed projection-input/output sensitivities,
scale/unscale status, source identity, mask count and exceptions/stacks. The
earliest *observed* nonfinite stage is distinguished from an originating kernel.
No nonfinite replacement, clipping, adaptive scaling, fallback or retry is added.
`components3.py` saves a successful positive component atomically **before** KD;
it retains partial evidence and writes a minimal failure receipt on exceptions.
Private photographs and large intermediate arrays are never serialized.

**96 final CPU tests passed** (38 new;20 E1 mask/cache;21 C11 scope;
17 E2 legacy/failure). The earlier95-test passing report is retained. Tests cover
small/large deltas, empty/exact-zero masks, zero norm, FP16/BF16 conversion,
finite terminal versus infinite upstream gradients, scaled/unscaled results,
gradient routing, stale scope, exceptions/hooks, atomic persistence and guards.
Nine independent FP64 autograd comparisons at observed nodes passed, with maximum
relative gradient L2 error **1.4e-07**. These are synthetic CPU observations,
not measured CUDA behavior. Earlier reference-test failures identified a rounded
norm-reference mismatch; the reference now models the saved norm correctly,
without relaxing the numerical tolerance.

All59 teacher caches were revalidated. Recomputed original masks retain exactly
**278 terms across15 source groups**,35 admitted KD regions from41 source grants,
and56 total training source groups. Existing C11 metadata remains267 terms on15
groups; only2 covered Chair scores exceeded0.15. No masks, source annotations,
inactive grants, categories or label evidence were changed. The same8 trainable
tensors/1,577,988 parameters and4 frozen calibration tensors remain specified.
9,973 inventoried historical files and
C8/C11/C13/C17–C21/D/E1/E2 seals passed preservation checks. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`. Full checkpoint/input hashes are in
`objective-preservation.json`; tensor dtypes are in `checkpoint-dtypes.json`.

The proposed one-shot recovery requires separate approval:

- One fresh original-cache correspondence forward on the failed photo, reusing
  the four sealed E2 original replays instead of repeating them all. Original
  agreement must yield exactly zero KD; no original VJP is needed.
- The same24 adapted-student optimization photographs in fixed order, using the
  immutable C11 checkpoint. Fresh features are necessary because E2 did not save
  them. Compare outputs against historical caches before derivatives.
- At most **25 detector forwards**, **25 head-only verification contractions**,
  **48 VJPs** (39 expected from existing coverage), **240 seconds**, **6.5 GiB
  allocated VRAM**, and **128 MiB** new output. Zero optimizer updates.
- Fresh destination `data/open-images/experiments/phase10e3-recovery-v1/`;
  exact source IDs and file hashes are in `recovery-disabled.json`.
- FP32 adjoints for both components; record precision, disable TF32 for these
  contractions, enforce watchdog/operation intents/global disk caps in the
  separately approved launcher, and stop without retry on any failure.

The proposal validator and dry run make zero GPU calls. This phase deliberately
contains no executable GPU launcher. The tested measurement components can be
bound to the bounded launcher after explicit recovery authorization.

The direct next sequence is: one bounded recovery; training-only norms/cosines
and source concentration assessment; one defensible fixed coefficient if the
evidence supports it; then one separately approved75-update original-initialized
training run and fixed comparison with original/C11 adapted baselines. The
existing E2 analytical non-domination bound can be reported, but it is not an
approved coefficient, an AdamW update bound or a performance guarantee.

A successful widened diagnostic does not certify the old FP16 training backward.
Before the75-update run, its precise numerical integration must be pinned and
CPU-tested; no silent substitution or automatic switch is permitted. This is a
concrete remaining training blocker, not permission for more variants.

No progress toward the immediate ≥60% precision / recall above68.85% target is
claimed from gradients. That target needs a comparable detection comparison;
the exposed19-photo partially annotated panel cannot establish the longer-term
80% precision/90% recall requirement on independent fully annotated seven-class
rooms. The latter still requires the separate independent evaluation evidence.

Rollback is simply to leave the new opt-in modules disabled and use the existing
immutable branch/checkpoints. No historical implementation needs reverting and
no artifact should be deleted. Small code and reports only are intended for Git;
all private diagnostics, photographs and weights remain local.
