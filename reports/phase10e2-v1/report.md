# Phase10E.2 — Stopped selective-distillation gradient verification

**Result D: inconclusive. The sole GPU attempt stopped on a nonfinite scaled gradient. Do not train;the distillation coefficient remains unset.** No retry,precision change,additional forward,optimizer,weight update,download,deployment or holdout scoring was performed.

## Actual budget and completion

Executed **6 forwards**, not the authorized maximum28:four original replays and two existing-C11 forwards. There were **7 VJP attempts**, six finite returns and one failed nonzero KD VJP. Five exact-zero KD derivatives were identified analytically without executing a VJP. Five positive parameter-gradient measurements were persisted;the sixth positive VJP returned finite but its norm was not persisted before the subsequent failure.

Worker time at failure:15.895s;supervised process wall time:17.726s. Peak allocated VRAM:1695397376 bytes (1.579GiB);reserved:2325741568 bytes. All stayed within180seconds/6.5GiBallocated/128MiBnewoutput. Durable operation intents and failure telemetry preserve the consumed counts. No further GPU execution is authorized by this completed attempt.

## Teacher and query replay

All59 original teacher identities,source groups,cache hashes and model/tokenizer/preprocessing contracts passed CPU preflight. Live source/pixel/token hashes,original-image registration,seven prompts/class order and60×60 row-major query layout were checked. No query realignment was performed.

All six completed forwards were **bit-identical** to their corresponding historical caches for logits and boxes,including the final raw forward recovered on CPU. Maximum observed replay error was zero. The predeclared historical FP16 tolerances were logit atol0.02/rtol0.001 and box atol/rtol1e-5;they did not need to be relaxed. FP16 forward values were stored as FP32 arrays;they are not independent FP32-forward references.

## State A: original initialization

| Photograph | Admissible terms | Positive projection L2 | KD projection L2 |
|---|---:|---:|---:|
| coco2017:000000002270 | 0 | 32.358654 | 0.000000 |
| coco2017:000000002347 | 1 | 22.714390 | 0.000000 |
| coco2017:000000030347 | 7 | 36.317802 | 0.000000 |
| coco2017:000000037367 | 2 | 34.455970 | 0.000000 |

Across the four photographs,ten admitted terms in three photographs had identical teacher/student logits. Distillation loss and derivative were exactly zero,as the MSE formula requires. Their zero parameter derivatives are explicitly analytic,not claimed as extra executed VJPs. Actual positive model-projection VJPs were finite/nonzero on all four photographs. This verifies original-start agreement and learnability,not regularization effectiveness or improved detection.

## State B: existing adapted C11 reference

Reference:`data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt`;SHA-256 `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f`. Loading this known snapshot in memory is the approved diagnostic state switch,not training or a learned model update. Two projection and six box tensors were gradient-enabled;calibration,shared representations,objectness,buffers and derived box bias remained frozen.

The first adapted photograph `coco2017:000000002270` completed with positive projection norm **1.194594**. It had no admitted KD term,so its KD derivative was analytically zero. That is the only completed adapted source-group measurement;there is no successful nonzero adapted KD projection-gradient norm or cosine.

The second adapted photograph `coco2017:000000002347` (source group `flickr-account:68258082@n06`) replayed exactly. Its single admitted Chair-negative term had teacher logit [-15.7265625],student logit [-9.546875],teacher score [1.4792467463325065e-07] and student score [7.14190139989638e-05]. The source-supported term was still below0.15;it was not an established high-confidence false-positive detection.

The recomputed finite unit-MSE loss is **38.18853759765625**. Its finite terminal-logit derivative is **[12.359375]**,or **[1582.0]** after scale128. These closed-form output derivatives are not recovered projection gradients. The actual model VJP then returned a nonfinite requested gradient and triggered the stop.

The prior positive VJP for this photograph returned finite,but its norm was not saved before the failing KD calculation. It remains unavailable. Raw outputs were saved atomically before loss/gradient work,so both the exact replay and finite scalar loss can be checked without new inference.

## Confirmed findings versus hypotheses

- Confirmed:finite,exactly replayed model outputs reached the scaled unit-distillation VJP;at least one requested returned gradient was nonfinite.
- Confirmed:the finite terminal-logit derivative is below FP16 maximum65504. The saved evidence does not identify which deeper projection-backward tensor/intermediate first overflowed.
- Likely explanation:mixed-precision backward overflow amplified by fixed scale128. This is a hypothesis supported by the failure boundary,not proof that removing scaling or switching precision would solve it. Neither alternative was executed.
- Missing:finite nonzero KD projection norms,directions,per-source ratios and any defensible fixed coefficient. No values were substituted from old experiments or invented.

## Protections,limitations and readiness

Completed positive VJPs had no box-output/box-parameter route and no accumulated parameter.grad. The failing KD VJP stopped before a full finite/routing summary,so no successful nonzero KD routing measurement is claimed. By construction the term uses only Chair logits and detached,source-supported masks;matched positives,ignored/ambiguous/native furniture regions and other six categories are not copied. All ten original selected terms were protected correctly and no completed record had a positive/KD overlap.

The intended24-photo source-group analysis could not be completed. One adapted source group has a persisted positive measurement and zero KD;the failing second source group has saved outputs but missing parameter norms. Sixteen planned source-covered regions already lacked original query coverage in E1,and this narrow Chair-only objective cannot establish seven-category confidence preservation. Neither score lowering nor zero original-start loss establishes better precision/recall.

**Coefficient:unset. Training readiness:D — inconclusive.** The first nonzero adapted KD gradient was numerically invalid,so actual parameter-gradient strength/conflict cannot be compared reliably. Choosing a fixed coefficient or preparing an enabled training arm would invent the required evidence. The existing E1 training configuration stays disabled and no new training configuration is proposed from this failed test.

**Recommended next action:**prepare a CPU-tested repair of gradient-measurement scaling and immediate per-target finite/dtype/magnitude telemetry,then seek separate approval for any further zero-update GPU attempt. Preserve the existing objective,photographs and checkpoints. Do not retry now,change precision automatically,train,or tune against the19-photo development panel.

## Model preservation and tests

Within State A,the complete before/after fingerprints match. Within State B,the failure-time complete fingerprint matches the pre-gradient adapted reference. Original and C11 checkpoint files remain unchanged. This separately verifies both model states;it does not incorrectly compare original weights with the intentionally loaded adapted heads. No optimizer was constructed and no new weight checkpoint was written.

50 relevant CPU regression tests plus 8 failure-specific integrity tests passed after stopping. Historical preservation checks cover9,973 inventory files and the C8/C11/C13/C17–C21/10D/10E.1 seals. No reserved or independent evaluation photographs were scored;no data,packages,production models or Mac application files changed.

Private artifacts:`data/open-images/experiments/phase10e-gradient-smoke-v1/`. Authoritative failure findings:`failure-analysis.json`;readiness:`decision.json`;completed-only gradient summary:`measurements.json`. The latter contains no recovered failed projection norm;the failure analysis explicitly distinguishes unobserved routes from successful completed checks. Source photos,raw query outputs and large model files stay local.
