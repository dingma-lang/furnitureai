# Phase 10E.4 — stable distillation gradient recovery

One authorized zero-update GPU recovery completed. **No training,optimizer,
weight update,download,development scoring or reserved-set scoring occurred.**
Every active source group yielded finite nonzero classification-projection
gradients. One proposed fixed coefficient is **7.063459015854303e-06**.
The75-update candidate remains disabled and requires separate approval.

| Resource | Actual | Approved maximum |
|---|---:|---:|
| Detector forwards,all completed | 16 | 25 |
| Head-verification contractions,all completed | 16 | 25 |
| VJPs,all completed | 31 | 48 |
| Supervised wall time | 26.700s | 240s |
| Worker elapsed,including CPU overhead | 26.048s | 240s |
| Peak allocated GPU memory | 1.5790GiB | 6.5GiB |
| Peak reserved GPU memory | 2.1660GiB | Reported separately |
| Optimizer steps / learned weight updates | 0 / 0 | 0 / 0 |

All59 teacher/image identities,source groups,processor tensor hashes,original
checkpoint and adapted checkpoint were verified before execution. 86
CPU tests passed before GPU use. We used one original correspondence photograph
and all15 saved-C11 photographs with nonzero selective coverage,representing15
source groups and267 terms. The other9 permitted photographs were skipped because
their immutable historical outputs had zero coverage. No IDs were substituted,
partitions changed or masks relaxed. The original59-photo teacher masks still
have278 terms across15 groups;the adapted count267 reflects existing query-box
and matching differences,not a changed annotation policy.

The one fresh original forward and all15 adapted forwards were **bit-identical**
to their historical logits/boxes. The measured forward is still FP16 autocast;
the numerical repair changes the adjoint,not these forward values. Internal head
reconstruction was exact. Original/student MSE and its closed-form terminal
derivative were exactly zero at the original initialization. No original VJP was
executed or falsely counted;the four E2 original replays were reused as sealed
evidence. Teacher targets were never rewritten.

Live CUDA dtype telemetry confirms FP32 shared head input,FP16 raw projection,
FP32 saved norm/normalized image/query features,FP16 contraction inputs/cosine
and calibration scale,and FP32 returned logits. Positive and KD adjoint stages
and parameter gradients were FP32,unscaled(scale1),with TF32 disabled for these
contractions. There was no GradScaler or manual128 multiplier in this run.
All 330 saved measurement telemetry events
were finite. The original E2 exact failing operation remains unknown: changing
both adjoint precision and scale does not isolate which original operation
overflowed or prove that lowering128 alone would have sufficed.

The31 VJPs comprise15 positive,15 selective-KD,and one native box-loss VJP. Both
classification components route to the two actual projection parameters;their
box-output/box-parameter gradients remain absent. The native box probe reached
all six box parameters with finite gradients and did not reach the projection.
Known protected/positive overlap violations:0. All uncertain/other-class output
terms remain ignored by KD. Scalar MSE and terminal gradients are reported
separately from these actual parameter-gradient measurements.

| Actual parameter-gradient measure | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Positive projection L2 | 0.0165098 | 1.60656 | 29.347 |
| Unit KD projection L2 | 1016.81 | 1349.83 | 3470.34 |
| Unit KD / positive ratio | 47.3066 | 864.376 | 70786.8 |
| Positive/KD cosine | -0.899903 | -0.713929 | -0.409432 |

| COCO image ID | Terms | Positive L2 | Unit KD L2 | Unit ratio | Cosine | Weighted ratio |
|---|---:|---:|---:|---:|---:|---:|
| 000000002347 | 1 | 0.050162 | 3470.34 | 69183 | -0.7347 | 0.48867 |
| 000000030347 | 6 | 1.0556 | 1150.41 | 1089.81 | -0.8782 | 0.0076979 |
| 000000037367 | 2 | 0.55106 | 1230.5 | 2232.95 | -0.6551 | 0.015772 |
| 000000074638 | 26 | 1.6066 | 1388.67 | 864.376 | -0.7139 | 0.0061055 |
| 000000108375 | 11 | 2.34 | 1016.81 | 434.526 | -0.8965 | 0.0030693 |
| 000000114729 | 7 | 0.042584 | 1228.02 | 28837.6 | -0.8322 | 0.20369 |
| 000000165742 | 49 | 8.1126 | 1604.33 | 197.756 | -0.5028 | 0.0013968 |
| 000000192834 | 33 | 5.3761 | 1398.93 | 260.211 | -0.4806 | 0.001838 |
| 000000195507 | 14 | 29.347 | 1388.31 | 47.3066 | -0.4629 | 0.00033415 |
| 000000263212 | 1 | 13.937 | 1247.46 | 89.5076 | -0.8999 | 0.00063223 |
| 000000294698 | 27 | 10.918 | 1349.83 | 123.638 | -0.4119 | 0.00087331 |
| 000000373101 | 62 | 0.74982 | 1300.93 | 1735 | -0.8428 | 0.012255 |
| 000000477263 | 11 | 2.2472 | 1596.86 | 710.585 | -0.4094 | 0.0050192 |
| 000000543194 | 3 | 0.33423 | 1368.03 | 4093.02 | -0.8657 | 0.028911 |
| 000000580827 | 14 | 0.01651 | 1168.68 | 70786.8 | -0.6696 | 0.5 |

All15 KD directions oppose positive classification gradients,so an unweighted
penalty is not justified. All15 KD gradients have positive cosine with the past
C11 projection displacement(range0.1065–0.1438);
descent would push against that displacement. This is consistent with restoring
pretrained behavior but **does not demonstrate less seven-category confidence
inflation**. Other six classes have no direct teacher constraint.

Source groups each contribute one measured photograph. The largest contributes
15.84% of the sum of individual KD norms;
the top three contribute30.45%.
These are norm shares,not fractions of a vector sum. The cosine between actual
source-summed positive/KD gradients is-0.66754.

The single coefficient uses only these training measurements:

    lambda = 0.5 * min(min_photo(||g_positive|| / ||g_unit_KD||),
                       ||sum_source_mean(g_positive)|| / ||sum_source_mean(g_unit_KD)||)

No coefficient sweep,development labels or development scores were used. With
lambda=7.063459015854303e-06,weighted per-photo KD/positive ratios range
0.000334148–0.5(median0.00610548).
The minimum measured positive-descent fraction in ordinary Euclidean gradient
geometry is0.640974;the analytic worst-case
bound is0.5. This is not an AdamW update bound,not valid for every future training
state,and not a guarantee of precision or recall. The weighted source-summed KD
norm is only0.2253% of the positive sum:the
weakest-positive examples set a conservative,possibly globally weak penalty.
At original agreement,the penalty starts with exactly zero gradient and grows
only as eligible student/teacher logits diverge.

Only2 of267 selected adapted Chair scores exceed
0.15. Most terms anchor already-low scores. Known positives are protected,but
shared parameter effects may still suppress useful Chair scores outside these
regions. These15 source groups and source-supported evidence are insufficient
to establish general real-room performance or universal annotation completeness.

Both in-memory stages retained identical421-tensor fingerprints before/after:
original `b969b61e49c5171754ed9b5e67c9491df55622049898654db6d84e505f553a45`,adapted `920751da07c624fdc6d5996b085f2afe32de8aaa338714d575c85293f6e34906`.
The authorized transition loaded existing C11 heads between stages;it was not
an optimizer update. Original OWLv2 SHA-256 remains `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`;
C11 SHA-256 remains `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f`. All9,973 inventoried historical
files and C8/C11/C13/C17–C21/D/E1/E2/E3 seals passed. `main` is unchanged.

Post-run checks passed85 tests plus6 new future-gate checks. One
E3 preparation-only assertion that the recovery directory must not exist is
explicitly inapplicable after authorized creation;it passed pre-GPU,and its
source was preserved. Every numerical,mask,routing and persistence check remains
active. Diagnostics,positive components,raw predictions and state hashes were
saved atomically;see `resource-accounting.json` and the final artifact seal.

The proposed training implementation is pinned in `precision4.py:combined` and
`future4.py`. It keeps the same FP16 forward and selective FP32 objective,uses
the recovered FP32 observed-node adjoint for **all** classification terms,and
the native box backward at scale1. It uses no GradScaler or legacy128 monitor.
Synthetic combined-loss CPU tests verify all8 parameter routes and unchanged
loss/gradient behavior at coefficient0 on the same numerical path. Actual CUDA
positive/KD components and one native box VJP passed. A full combined GPU VJP
has intentionally **not** been run;it is a mandatory zero-update check within
the future authorized queue,before optimizer construction. Finite behavior over
all future parameter states cannot be promised.

This is explicitly a different numerical backward path from C11's rounded FP16
adjoints/GradScaler128. Comparing against C11 is useful context,not a pure
loss-only ablation. No existing training worker or production defaults changed.

One disabled configuration is ready for separate approval:original OWLv2,
59 photographs/182 positives,2 projection+6 box tensors,original frozen
calibration/shared features,seed6106,AdamW LR1e-5,batch1/accumulation4(last3),
5 epochs/75 updates,600 seconds,6.5GiB,373 forwards(59 correspondence+295
training+19 fixed diagnostic inference),354 head-verification contractions,
2GiB output ceiling,and lambda=7.063459015854303e-06. Checkpoints are versioned
under `data/open-images/experiments/phase10e-distillation-fp32-v1`;that directory does not exist and no training was
launched. Loss/gradient/integrity/resource failures stop without retry.

The future19-photo comparison must use identical FP32 inference,threshold0.15,
annotations/mapping/matching and AP50/AP75 definitions for original and existing
adapted baselines. Report matched-recall precision,TP/FP/FN,Chair errors,Sofa/Bed
regressions,localization/duplicates,shared logit drift and disagreement. Save
predictions before separate CPU metrics. Do not score frozen/official material
or tune the fixed coefficient from this exposed panel. The ≥60%precision/recall
above68.85% development objective and eventual independent80%precision/90%recall
have not been advanced by this gradient-only diagnostic.

Recommendation:one separately authorized75-update diagnostic is technically
prepared,with cautious expectations because the global penalty is small and
coverage is mostly easy negatives. No further diagnostic variant or training
cycle is authorized. Rollback is to leave the new entry point disabled and use
the preserved original branch/checkpoints;delete or overwrite nothing.
