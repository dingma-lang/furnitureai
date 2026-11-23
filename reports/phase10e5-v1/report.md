# Phase 10E.5 — first selective teacher-distillation training

**Result C — Ineffective for the immediate precision/recall objective.** The one
authorized experiment completed safely,but severe confidence inflation and
scored false positives persist. No further experiment or deployment occurred.

| Model | Precision@0.15 | Recall@0.15 | TP/FP/FN | AP50 | AP75 | Precision at ≥original recall |
|---|---:|---:|---:|---:|---:|---:|
| original | 62.69% | 68.85% | 42/25/19 | 74.99% | 55.28% | 68.85% |
| C8 | 6.86% | 95.08% | 58/788/3 | 75.25% | 55.21% | 80.77% |
| C11 | 8.76% | 90.16% | 55/573/6 | 76.76% | 60.58% | 56.76% |
| E5 | 8.76% | 90.16% | 55/573/6 | 76.76% | 60.58% | 56.76% |

The last column is the unchanged PR-envelope maximum precision at recall at
least42/61(68.85%),not an adopted threshold. Original's envelope precision is
68.85%(42TP/19FP),compared with E5's56.76%(42TP/32FP). E5 therefore cannot
recover ≥60%precision while retaining original recall anywhere on this saved
curve. No threshold,coefficient or hyperparameter was changed using these data.

E5's threshold counts are identical to C11;AP50 differs by only about
-0.00098 percentage points,and AP75 is identical. This is no meaningful observed
improvement over that historical model. AP/localization gains versus original
do not compensate for25→573 scored FP. The original-model precision target
was lost despite recall improving68.85%→90.16%.

| Category | Original TP/FP/FN | E5 TP/FP/FN | E5 precision | E5 recall | Original→E5 AP75 |
|---|---:|---:|---:|---:|---:|
| chair | 24/19/15 | 35/486/4 | 6.72% | 89.74% | 31.75% → 39.95% |
| sofa | 11/1/2 | 12/26/1 | 31.58% | 92.31% | 69.31% → 74.64% |
| bed | 7/5/2 | 8/61/1 | 11.59% | 88.89% | 64.78% → 67.16% |

Chair contributes486/573(84.82%)of E5's scored FP. Chair misses fell15→4,
but Chair FP rose19→486. Sofa and Bed gain one TP each,with26 and61 scored FP.
Sofa/Bed AP75 improves. On the *same* GT matches,Sofa mean IoU improves by
+0.014888,Chair by+0.013584,and Bed changes
-0.000902. The tiny shared-Bed decrease is not a major regression.
Comparing all matched-box means without accounting for newly matched objects
would confound localization with differing match difficulty.

Scored duplicate FP:original1,E52.
Fragmentation candidates increase4→44,mixed-object candidates15→61,and category
disagreement candidates14→29. These geometric heuristics are not human-confirmed
failure labels. The FP explosion is not explained by scored duplicates alone.

The native completeness policy leaves1513
post-NMS predictions unscored at0.15,including849 Chair predictions. They are
not confirmed errors or negatives. Per-category `unknown_unmatched` in the
historical metric payload uses the0.001 AP floor;the separate `at_015` field
reports operating-point unknowns. Ground truth supports39 Chair,13 Sofa and9 Bed
instances across19 exposed,partially annotated photographs. Table,Desk,Cabinet
and Lamp lack supported positive scoring here. Native Chair/Stool semantics
also retain the historical difference from the project's Chair-includes-seating-
stools policy. Nothing was silently relabelled or treated as verified background.

Broad inflation persists on identical68,400 query locations. Mean seven-class
logit drift is+3.010410,versus C11's
+3.011035. E5 differs from C11 by only
-0.000625in mean raw logit.

| Category | Mean E5 logit change vs original |
|---|---:|
| sofa | +4.200500 |
| chair | +4.246541 |
| table | +2.318330 |
| bed | +3.835936 |
| desk | +2.605568 |
| cabinet | +1.864930 |
| lamp | +2.001061 |

Raw Chair scores≥0.15 rise82→2,713(C11:2,714);scores≥0.5 rise3→440.
E5's raw category argmax disagrees with original at41,763/68,400 query locations.
These are aligned patch queries,not independently matched physical objects.
Score/logit quantiles for all seven categories are saved in `results.json`.

The safety gate verified all59 original-cache correspondences,then the required
three representative full combined-loss checks,covering99 nonzero-coverage
Chair-distillation queries across3 source groups and all3 positive categories.
The checks used one original-initialized student,original teacher caches,and
the approved FP16-forward/FP32 observed-node classification adjoint,with native
box backward and scale1. No GradScaler or manual128 scaling was used. Full/base/
KD gradient decomposition,positive projection route,all six box gradients,
known-positive/ignored protections,frozen scope and coefficient application
passed. At original agreement KD loss and gradient are *zero*,as expected;the
gate's nonzero coverage must not be misreported as a nonzero initial KD gradient.
Actual nonzero adapted KD propagation had already been recovered in E4 and was
then monitored during this training trajectory.

No optimizer existed during the gate,and no parameter changed. The first update
intent follows the passing gate marker. The59 original checks and gate consumed
36.718s of the same600s queue. There were no automatic
retries,alternate precision paths or extra models.

Actual execution:5 epochs,75 verified AdamW steps,295 training presentations,
373 detector forwards(59original checks+295training+19final FP32 inference),
354 head-verification contractions,383 explicit diagnostic VJPs,
and295 training backward calls. Wall time was193.726s;
worker elapsed191.978s;peak allocated VRAM
1.8858GiB(reserved
2.1680GiB),within6.5GiB.
The pinned data,seed6106,order,batch1/accumulation4(last3),LR1e-5,AdamW
betas0.9/0.999,eps1e-8,weight decay0.01,clip norm1 and coefficient
7.063459015854303e-06were preserved. No backbone/calibration updates occurred.

| Epoch | Positive loss | Box loss | Unit KD | Weighted KD | Median KD/positive grad norm | Maximum ratio | KD term occurrences |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.355980 | 0.462742 | 0.131187 | 0.00000093 | 4.18524e-05 | 0.000596248 | 276 |
| 2 | 0.095428 | 0.462663 | 0.938815 | 0.00000663 | 0.000575192 | 0.037107 | 274 |
| 3 | 0.048558 | 0.449766 | 2.038486 | 0.00001440 | 0.00198535 | 0.168284 | 270 |
| 4 | 0.046650 | 0.426679 | 2.515221 | 0.00001777 | 0.00286206 | 0.268827 | 269 |
| 5 | 0.033013 | 0.423580 | 3.090153 | 0.00002183 | 0.00364705 | 0.445817 | 266 |

Losses are per-image diagnostic means,not norms of the accumulated optimizer
gradient. Every epoch retained15 source groups with KD coverage. Term counts
are occurrences during an evolving epoch,not final-checkpoint coverage snapshots;
the initial teacher mask had278 terms. The fixed containment/protection policy
was unchanged as predicted boxes and positive matches evolved.

There were73 actual nonzero KD VJPs,and all73 conflicted with positive-gradient
directions. No ratio exceeded0.5 and no KD-dominance flag fired. Epoch5 active-
presentation median weighted KD/positive norm ratio was0.003647(0.365%),with
maximum0.445817. Weighted KD loss was only about0.051%of mean classification
loss at epoch5. The term was mathematically active,but its observed global
effect was negligible:the coefficient was too weak in aggregate within this
limited,mostly-easy-negative coverage. That does not show that increasing it
would be safe or sufficient;verified-positive conflicts and uncovered queries
remain substantial limitations.

The projection changed from original(weight L2 displacement
0.14572461,bias
0.00616913),while
remaining extremely close to C11(weight difference L2
7.3460226e-05).
All five new head/optimizer/RNG checkpoints passed CPU loading,hash,parameter-
scope,coefficient and optimizer-counter checks. All4 calibration tensors stayed
bit-identical,and full frozen-state hashes match before/after. Historical
original OWLv2,SigLIP2,classifiers and detector checkpoints remain unchanged;
9,973 inventoried historical files and prior experiment seals passed.

Final checkpoint:`data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt`

SHA-256:`416620a947a96579bdd20c85e7adeb8b60f65c8983e73d3b3943a58ceabef4f0`

All19 prediction arrays and their source/hash manifest were saved atomically
before the isolated Torch-free CPU metrics process. Original,C8 and C11 metrics
were recomputed from saved outputs and exactly matched their historical headlines.
84 CPU tests passed before training and
84 afterward. Additional integrity
checks verified actual forward partition membership,exactly five presentations
per training photograph,no updates before the gate,constant coefficient/scaling,
and immutable checkpoint contents. No frozen/reserved images were scored.

This is not a clean causal distillation-only ablation against C11:the historical
control also differs in numerical adjoints and GradScaler use. C8 additionally
trained calibration parameters. The19-photo panel is exposed and partially
annotated,and three-category native weak-label agreement cannot establish
independent seven-category accuracy. No progress toward the simultaneous
≥60%precision/improved-recall milestone,or eventual independent80%precision/
90%recall requirement,is demonstrated.

**One next direction:** improve *training-only trusted hard-negative coverage*
at actual high-confidence Chair-confusion locations,using source-supported or
human-confirmed evidence and protecting positive Chairs/seating stools and all
uncertain regions,before another training run. Do not mine the19-photo evaluation
panel into training labels. No additional acquisition,annotation activation,
inference,coefficient sweep or training was performed or authorized here.

The candidate remains isolated and undeployed. Rollback means retaining the
immutable original/previous checkpoints and leaving production defaults alone;
no historical artifact should be deleted or overwritten.
