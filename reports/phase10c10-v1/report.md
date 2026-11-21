# Phase 10C.10 — logit-calibration attribution and disabled training preflight

**Both learned detection-calibration modules changed during C8, including all
four weight/bias tensors. They received75 optimizer steps and have named
classification-gradient norm records. Their actual per-query contribution to
the confidence drift cannot be recovered from the saved artifacts.**

The scalar biases alone are too small to explain the observed change: a
conservative real-arithmetic counterfactual bound is below **.054 logits**,
versus the corrected model's mean Chair increase **+5.4225 logits**. The
feature-dependent calibration weights could matter substantially, but the
classification projection remains another viable source of broad inflation.
The95.45% common-mode statistic is not a95.45% calibration attribution.

One calibration-frozen ablation is implemented and CPU-preflighted, **not
authorized or executed**. No GPU training/inference, real-photo model forwards,
new downloads, production changes or Phase10D activation occurred. CPU tests
use isolated synthetic modules/tensors; they do not train the furniture models.

## Preserved inputs and original scoring implementation

Started at `0b8b801fe026e1471a103abb5ddf2218636e7e88` on the C9 branch and created
`experiment/phase10c10-calibration-preflight`. All C8/C9 public/private artifact
seals, pinned model/processor files and78 eligible image hashes were verified.
Original OWLv2 file SHA-256 remains:

`e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.

Read-only C8 endpoints:

- A: `data/open-images/experiments/phase10c8-ab-v2/A-control/checkpoints/update-075-65391d9c26e1.pt`
  SHA-256 `791cc41cacd4c8fad1a3ea8d6a822323c01743f3bb379b519abc94da64a4a44e`.
- B: `data/open-images/experiments/phase10c8-ab-v2/B-corrected/checkpoints/update-075-9164284d514b.pt`
  SHA-256 `986d511ae2c80ced12a4a8542bdd300af0477230246a4df1ef22ffdf8519a930`.

Audit uses the installed Transformers implementation, not another detector's
design. File paths/hashes are in the private input contract and C9 source audit.
For a960×960 processed photograph there are60×60=3,600 patch queries:

1. Frozen vision encoder outputs pass through vision post-LayerNorm. Each patch
   is multiplied elementwise by the broadcast CLS feature, then through the
   final frozen768-dimensional `layer_norm`.
2. Frozen text encoder/`owlv2.text_projection` produce the seven prompt vectors,
   normalized in `Owlv2Model` and again in the detection head with epsilon1e-6.
3. `class_head.dense0` projects image features768→512, followed by per-query
   L2 normalization with epsilon1e-6.
4. Normalized image/text dot products get an image-dependent shift and positive
   scale. For frozen pre-head feature f:

```
cos[q,c] = dot(normalize(dense0(f[q])), normalize(text[c]))
shift[q] = Linear_shift(f[q])
scale[q] = ELU(Linear_scale(f[q])) + 1
logit[q,c] = (cos[q,c] + shift[q]) * scale[q]
score[q] = sigmoid(max_c logit[q,c])
```

These are **two Linear modules, four tensors,1,538 scalars**, not two global
scalar parameters. Shift/scale are shared across the seven classes for each
patch, but vary with its image feature. Freezing only their biases would not
freeze the feature-dependent calibration weights.

There is also **`owlv2.logit_scale`**, shape scalar, value4.605170249938965. It is
exponentiated for auxiliary pooled CLIP contrastive logits, approximately100×.
It was already frozen and **does not set the detection-head logits/final score**.
Likewise the frozen pooled `owlv2.visual_projection` is not the trained patch
classification projection. The intervention targets `class_head.*`, not that
global temperature or pooled projection.

`objectness_head` computes a separate score from detached patch features, but
the executed C8 loss and postprocessing never use it for query selection,
confidence multiplication or object-presence loss. Category scoring uses all
patch queries before floor/NMS. The final scores are not validated probabilities.
No objectness-gating experiment is part of the proposed change.

## Exact parameter differences and optimizer evidence

Weights/vector biases below are represented by L2 norm; scalar biases show their
signed values. Full precision min/max/mean/std/norm/shape and tensor hashes for
every original/A/B head tensor are in `checkpoint-parameter-audit.json`.
`parameter-differences.csv` covers all12 head tensors. A private1538-row
`exact-calibration-values.csv` preserves every original/A/B calibration scalar
and signed difference without uploading model weights.

| Exact parameter path | Shape | Original scalar or L2 | A scalar or L2 | B scalar or L2 |
|---|---|---:|---:|---:|
| `class_head.logit_shift.weight` | [1,768] | .618349832384 | .620384329266 | .620362031987 |
| `class_head.logit_shift.bias` | [1] | -4.582312583923 | -4.581947803497 | -4.581951141357 |
| `class_head.logit_scale.weight` | [1,768] | 4.653565013496 | 4.649894476358 | 4.649923974120 |
| `class_head.logit_scale.bias` | [1] | .112281218171 | .111988954246 | .111992180347 |
| `class_head.dense0.weight` | [512,768] | 9.701100049825 | 9.700365610341 | 9.700385729483 |
| `class_head.dense0.bias` | [512] | .100537719024 | .102162839925 | .102147925874 |

Absolute vector change is `||trained-original||₂`, not the difference of the
two tensor norms. Relative change divides that vector-change norm by original
norm. Scalar signed changes are explicitly shown here:

| Parameter | A change | B change | Relative A / B |
|---|---:|---:|---:|
| Shift weight, delta L2 | .007835703176 | .007793976657 | 1.267196% /1.260448% |
| Shift bias, signed | +.000364780426 | +.000361442566 | .007961% /.007888% |
| Scale weight, delta L2 | .006202653434 | .006163470864 | .133288% /.132446% |
| Scale bias, signed | -.000292263925 | -.000289037824 | .260296% /.257423% |
| Projection weight, delta L2 | .145890496332 | .145208931521 | 1.503855% /1.496830% |
| Projection bias, delta L2 | .005888556543 | .005840462968 | 5.857062% /5.809226% |

Both Phase10B and C8 call `expanded.configure_heads`: all parameters become
frozen except the class head and box head, **12 tensors /1,579,526 scalars**.
The C8 serialized weight dictionary and optimizer share the exact ordered
`model.named_parameters` construction. Param IDs0–11, tensor shapes, nonzero
Adam first moments, changed values and **step75** counters verify that every
head tensor, including all four calibration tensors, participated in updates.
These were not unused optimizer entries.

Named gradient norms were recorded at epochs0/1/3/5 for all six class-head
tensors, over the same24 optimization photographs. They are unscaled, per-photo
classification-only and added-negative norm probes. Mean **used-classification**
norms in B, initialization→epoch5:

| Tensor | Initial | Epoch5 |
|---|---:|---:|
| `dense0.weight` | 39.477102 | 2.828411 |
| `dense0.bias` | 1.438334 | .101829 |
| `logit_shift.weight` | 115.087886 | 7.976513 |
| `logit_shift.bias` | 4.198446 | .281897 |
| `logit_scale.weight` | 1.043245 | .022499 |
| `logit_scale.bias` | .037918 | .000810 |

These prove an active gradient path, not a percentage contribution to logits or
the clipped AdamW update. Signed full-minibatch gradient tensors were not saved.
Training-step logs contain aggregate class/total-head norms. The six box tensors
have optimizer/moment/update evidence but no per-box-tensor gradient probe logs.
Adaptive optimizer state and clipping make raw gradient-norm ranking unsuitable
as functional attribution.

The vision/text encoders, text projection, global contrastive temperature,
objectness and normalization parameters were excluded from the optimizer. The
same original-initialized non-head frozen hash was recorded before/after both
arms: `224b9eeaffa0b9086a31b7d5a3cc4ac62467d4dcdabe4211dbfa8502755f9e31`.
Their A/B equality follows this preserved scope/hash evidence; separate full
trained-backbone tensor copies were not serialized. See
`shared-parameter-audit.json` for original shapes/values and this distinction.

## What can and cannot be attributed now

The saved original/A/B inputs and forward precision match. C9 measured all seven
logits increasing at every one of68,400 development queries; the corrected mean
Chair change is +5.422518 and median category-common change +4.222805. The
common component accounts for95.4468% of squared total logit change.

The relevant local NPZ audit found:

- 152 files: logits, square boxes, final boxes, scores, labels.
- 192 files: logits and square boxes only.
- 24 C7 files: logits and square/original normalized/pixel boxes.

None contains pre-classification f, original text vectors, normalized projected
features or actual per-query shift/raw-scale/activated-scale outputs. Snapshot
weights/optimizer states do not supply those per-image activations. SigLIP
embeddings from other project stages are a different encoder and cannot replace
OWLv2 features. Nothing was reconstructed or fabricated to fill the gap.

| Candidate source | Current evidence | Attribution conclusion |
|---|---|---|
| A. Shift module | Weight/bias updated75times; large named gradient norms | Plausible contributor; actual per-query shift change unavailable |
| B. Scale module | Weight/bias updated75times; bias decreases; feature-dependent activated scale unknown | Plausible contributor; a lower positive scale can lift negative logits, so parameter sign alone is insufficient |
| C. Dense projection + normalization | Weight/bias changed; normalized directions can rotate toward multiple similar prompts | Viable competing source; small relative weight change does not bound logit drift tightly without feature/projection norms |
| D. Other feature/norm parameters | Frozen hashes and identical preprocessing/prompt/eval paths | No learned parameter-change contribution from those frozen modules; direct activation equality is not measured. Normalization after changed dense0 can still change and belongs to C |

### A useful bias-only bound without invented activations

The frozen final LayerNorm gives `f=gamma*u+beta`, with sum(u)=0 and
`||u||₂<=sqrt(768)` in real arithmetic. For a scalar linear form `w*f+b`, its
range is bounded by center `w*beta+b` and radius
`sqrt(768)*||center(w*gamma)||₂`. Normalized cosine magnitude is at most one;
`ELU(x)+1` is1-Lipschitz. This yields conservative counterfactual envelopes:

| Envelope | A | B |
|---|---:|---:|
| Maximum raw shift change from whole module delta | .209255 | .208225 |
| Maximum raw/activated scale change from module delta | .165274 | .164310 |
| Maximum logit change from **biases only**, other components fixed | .053428 | .052926 |
| Loose maximum logit change from whole calibration delta, projection fixed | 30.629838 | 30.475577 |

The original activated-scale upper envelope is128.560514. These are universal
analytical envelopes, **not actual activations, observed contributions, additive
percentages or a formal floating-point rounding-error certificate**. Bias-only
changes cannot by themselves produce the observed mean +5.42 Chair change under
this model. The much looser whole-calibration bound does not rule out substantial
feature-dependent effects, but cannot establish that they occurred or dominated.

No defensible percentage split among shift, scale and projection is available
now. In particular,95.45% shared drift does not mean95.45% came from shift/scale.
No shift-only, scale-only or projection-only real-photo counterfactual was run.

## Is freezing the two modules sufficient?

**It is technically meaningful and a justified small ablation, not a proven
repair or a sufficient constraint on confidence.** Both modules actually moved,
were optimized and offer class-shared routes for changing positive scores.
Freezing them removes those four update paths. The remaining dense projection
can still rotate normalized image embeddings toward multiple furniture prompts
and recreate a common logit increase with pretrained calibration fixed. A
synthetic cosine test demonstrates that possibility; it is not evidence about
actual unseen room features.

The underlying sparse-negative objective remains unchanged: only matched class
positives, explicit native negatives and the41 source regions are supervised.
Unknown regions stay ignored. C8 never applied an added Chair-negative term
with score≥.50. Calibration freezing does not create missing hard negatives,
repair native Chair/stool completeness or provide seven-class real-room labels.

An important implementation trap was found: **the legacy C8 worker calls
`configure_heads` again at every training/probe/evaluation transition**, which
would re-enable all12 tensors. Merely applying the C9 freeze helper once and
reusing that worker would fail the intended experiment.

The isolated C10 worker instead installs `FrozenScope` once and never calls the
legacy re-enabler. Every mode transition and pre/post-update check enforces:

- Exactly eight named trainable tensors; all remaining parameters frozen.
- All four calibration gradients are **None**, stricter than zero tensors.
- Calibration parameters excluded from every AdamW group/state, including any
  possibility of weight-decay updates or duplicate group entries.
- Original calibration bytes unchanged; full frozen-state hashes rechecked.
- Exact named optimizer order, hyperparameters, finite gradients and step counts.

Accidental re-enabling fails immediately rather than being silently repaired.
The global norm clip remains1 over the permitted parameters. Removing four
gradients can change clipping of the remaining gradients: that is a consequence
of freezing, not a separately tuned coefficient. No other training rule changes.

## One disabled experiment and prepared diagnostics

Final immutable config: **`reports/phase10c10-v1/experiment-disabled-v3.json`**.
File SHA-256:
`6f1bd6ab6443d0a22d5655bf58b30f4fe28cc29939223936f04637dfc34f8333`.

Earlier disabled revisions remain intact; all three have identical experiment
semantics. Only implementation/test pins were refreshed. They are not extra
training variants, and none was executed.

| Item | Fixed proposal |
|---|---|
| Initialization | Immutable original OWLv2, not an adapted C8 checkpoint |
| Trainable classification tensors | `class_head.dense0.weight`, `.bias` |
| Trainable localization tensors | `box_head.dense0/1/2.weight`, `.bias` |
| Trainable total | Eight tensors /1,577,988 scalars |
| Newly frozen | `class_head.logit_shift.weight`, `.bias`, `class_head.logit_scale.weight`, `.bias` |
| Already frozen | All other model parameters, including vision/text, norms, objectness |
| Data | Same59 optimization photos /56 known groups /182 positives /41 existing negative regions; no new labels |
| Optimizer | AdamW LR1e-5, betas.9/.999, eps1e-8, weight decay.01 |
| Sampling/precision | Seed6106; same C8 order; batch1/accum4, last group3; FP16 autocast/FP32 loss; GradScaler128 |
| Loss | Same C8 B coefficients, Hungarian matching, corrected masks and ignores |
| Maximum | One configuration, five epochs /75 updates,600seconds,6.5GiB VRAM |
| Selection | Fixed final-epoch diagnostic endpoint; no adaptive tuning |

`training_authorized`, source activation and Phase10D authorization are all
false. The future destination
`data/open-images/experiments/phase10c10-calibration-frozen-v1/` **does not exist**.
The launcher checks a separately approved new config, explicit approval reference
and matching semantic/file hashes before creating a worker or initializing CUDA.
There is no flag-only bypass and no approval record was fabricated.

The prepared standalone run allows467 forwards:295 optimization inputs,96
unchanged training-source probe inputs and76 FP32 diagnostic inputs at0/1/3/5.
C8 B used448 because it reused A's original reference. The19 initial reference
forwards are explicitly included here; collecting their pre-head/text outputs
does not add further encoder forwards or alter the learning objective.

Prepared diagnostics preserve:

- Same-index, seven-class shared logit drift and Chair distributions.
- Native-scored Chair FP at.15/.50/.80, with unknowns separate.
- P/R at.15; AP50/AP75; TP/FP/FN and precision at the original overall recall
  .688525. Post-hoc PR envelopes remain descriptive, not production thresholds.
- Sofa/Bed recall, AP75/localization, duplicates/fragmentation and uncertainty.
- Named gradients for every head tensor, None for calibration; parameter hashes
  and full checkpoint compatibility.
- Actual training-query negative counts/scores, responsible region IDs and
  protected overlap checks, using the same C8 objective implementation.

Raw prediction artifacts are saved atomically before separate CPU metrics. The
parent process enforces a hard600second deadline; the worker monitors allocated,
reserved and observed device memory plus source integrity, nonfinite events,
update/forward limits and unique checkpoint writes. No automatic retry, new
variant or budget expansion is implemented. A4GiB private-output guard bounds
snapshots/features/logs; expected data is roughly1.7GiB. The19 feature files are
bounded at12MB each, approximately211MB total.

For a later authorized run, actual initial pre-head/text caches can permit
**functional** attribution on CPU: reconstruct original/A/B/candidate endpoint
logits within declared numerical tolerance, then evaluate all eight combinations
of original/trained projection, shift and scale. Six-order Shapley averages
allocate their interactions and must sum to endpoint delta. If provenance or
endpoint reconstruction fails, attribution stops; no zero/inferred features or
automatic new inference is permitted. Such head-swap attribution is conditional
on fixed features, not causal attribution of the training dynamics. No real
feature counterfactuals were computed in this phase.

Full12-head checkpoint payloads remain compatible with the historical inference
loader. Training restore separately requires original calibration and eight
named optimizer entries, with matching cursor, frozen identity, moment shapes
and settings. Old C8 adapted heads cannot masquerade as original initialization.
Checkpoint recovery evidence is retained; interrupted/uncertain updates are not
replayed automatically.

## CPU tests, readiness and approval boundary

**119 CPU tests passed**:40 C10 tests,16 C8 regressions,31 source-mask tests and32
legacy preflight tests. Covered persistent frozen identities/gradients, optimizer
exclusion, original initialization, installed class-head formula compatibility,
checkpoint serialization/restore/legacy inference loading, loss masks, source
isolation, deterministic config/order, disabled-before-CUDA execution, missing
activation rejection and counterfactual sum closure. A complete75-step synthetic
CPU optimizer test kept calibration byte-identical and gradients absent.

Two initial test fixtures were corrected (identical synthetic frozen-backbone
initialization for restore, and named optimizer groups matching Torch's API).
Their logs and disabled drafts are preserved. No real model/GPU experiment
failed or was retried. No production or historical code was patched.

Post-preflight checks verified9,973 historical model/data files,160 prior training
snapshots,24 older reports,27 implementation files and all C8/C9 sealed artifacts.
Original OWLv2, Phase5C, SigLIP2 checkpoints and all data/holdout partitions remain
unchanged. CUDA was not initialized. Actual CUDA/AMP/runtime behavior of the new
worker has **not** been exercised; it remains conditional on a separately
authorized initial probe. Earlier C8 memory usage is context, not a C10 result.

**Reason to proceed after separate approval:** a cheap, isolated test of four
genuinely optimized shared-calibration tensors, with unchanged loss/data and
strong scope checks. **Reason not to treat it as a solution:** exact layer
attribution is unavailable, projection inflation remains possible, negatives
remain mostly easy/sparse, and the19-photo panel has incomplete labels and prior
exposure. Do not increase budgets, combine corrections or deploy on that basis.

Proposed diagnostic evidence criteria remain predeclared: at least20% fewer
native-scored Chair FPs at.15, Chair/aggregate recall drop at most3 percentage
points and matched-recall precision drop at most2 points; disclose Sofa/Bed
regressions. Reject the candidate or stop on nonfinite/resource/scope/source/
checkpoint failure. If it fails to materially improve comparable-recall
precision, do not extend or launch another cycle automatically.

The independent seven-class real-room protocol remains separately frozen and
unconsumed (`reports/phase10c9-v1/independent-evaluation-protocol.json`): unseen
source groups, complete category/stool annotations, QA, ignores, fixed operating
points and uncertainty. No acceptance photograph was acquired or evaluated.
The80% precision /90% recall target remains a future independent requirement.

**Next approval required:** exactly one calibration-frozen run under the final
v3 specification (five epochs/75 updates, ten minutes,6.5GiB), including only its
declared existing-photo diagnostics. This report grants no training permission.
Phase10D remains unauthorized.

Supporting artifacts: `checkpoint-parameter-audit.json`,
`parameter-differences.csv`, `shared-parameter-audit.json`,
`attribution-limits.json`, `diagnostic-contract-v3.json`, `readiness-v3.json`,
`tests-final.json`, `preservation.json`. Commands and recovery constraints are
in `tools/phase10c10-v1/README.md`.
