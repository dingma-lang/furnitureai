Furniture AI Phase 10C — CPU-only validation recovery and precision diagnosis

Prepared on the user-session date 2026-11-19. No GPU inference, training,
downloads, environment changes, dataset reassignment or production replacement
were performed. The experimental branch is
`experiment/phase10c-validation-precision`, based on Phase10B commit
`25dff1f5bc937c1a6466a31059ea359392c9e314`.

The LR3e-5 epoch-5 weights survived, but their predictions did not. Final metrics
cannot be recovered CPU-only. Both fits retain 75 committed optimizer updates;
neither needs to be retrained to recover validation. An inference-only command
and an explicitly unapproved authorization template are prepared separately.

The saved fatal trace places the abort in the Python main thread, evaluating
the generator inside `tools/phase9-v1/fusion.py:valid`, called by metric event
matching after validation inference. The autograd thread had no Python frame.
The log contains no associated CUDA OOM, nonfinite-loss failure or update-budget
failure. This establishes where execution stopped, not what corrupted execution.

In [CPython 3.14.4's interpreter source](https://raw.githubusercontent.com/python/cpython/v3.14.4/Python/bytecodes.c),
the fatal message is emitted when the interpreter executes the CACHE data opcode
as an instruction. It is an interpreter invariant failure, not an ordinary
invalid-box exception. It does not identify a stale disk cache, model fault,
CUDA fault or particular native extension as the cause. No project-local core
dump was found. Current disassembly has no jump targeting CACHE; it cannot
reconstruct the aborted process's memory.

The project interpreter is Python3.14.4 with GIL enabled. JIT support is compiled
in but was disabled in the inspection process; crash-time JIT state was not
saved. PyTorch is 2.14.1+cu132. Current trace/profile hooks are absent, and no
explicit torch.compile, code.replace or eval-frame hook was found in the
Phase9/10A/10B project code. Extension modules appearing in a fatal log do not
prove their responsibility. Four isolated CPU probes passed: 500,000 historical
validator calls with and without a Torch import, and 21 historical category/AP
passes per metric probe with and without Torch. CUDA remained uninitialized.
These probes do not reproduce the full post-GPU process state and do not prove
the problem resolved. Interpreter specialization, an extension interaction or
memory corruption remain hypotheses. No environment or hardware change is
recommended from this evidence alone.

A superficially similar [WSL issue](https://github.com/microsoft/WSL/issues/41019)
was also checked: its author ultimately corrected the initial kernel diagnosis
to a hardware problem. It is not evidence that this project's kernel or hardware
caused the abort. No WSL, BIOS, kernel, Python or package settings were changed.

The worker builds predictions in a local dictionary, calls metrics, and saves
only aggregate metrics afterward. Nine validation records exist: LR1e-5 E1–E5
and LR3e-5 E1–E4. Neither those records, the reference file nor the weights-only
checkpoint payload contains per-image boxes/logits or event arrays. Global PR
curves retain pooled scored TP/FP thresholds, but lose category and image
identity. There is no epoch-5 prediction cache to score in a CPU subprocess.
The exact saved LR3e-5 E5 checkpoint is:

`data/open-images/experiments/phase10b-heads-v1/heads-original-lr3e-5/checkpoints/update-0075-ca40fb61879a4f838a6e254227a700d5.pt`

SHA-256: `3c8774892d6ba85ddb01f735fa9391790e4200259dd302176ab429a863d6eb2f`.
It remains immutable and unscored at epoch5. E1 below is the earlier selected
available LR3e-5 checkpoint; E4 is the last completed validation, not E5.

All following results are descriptive agreement with native partial annotations
on the same already-exposed 19 training-derived internal photographs and 61
boxes. They are not human-verified accuracy, an independent benchmark or a
seven-class result. Source groups are isolated from the new optimization pool,
but Phase5C previously trained on three of these photographs. No official,
reserved or frozen evaluation material was scored.

| Model | Precision @0.15 | Recall | AP50 | AP75 | TP/FP/FN |
|---|---:|---:|---:|---:|---|
| Original OWLv2 | 62.69% | 68.85% | 74.99% | 55.28% | 42/25/19 |
| Phase5C | 57.53% | 68.85% | 74.51% | 58.51% | 42/31/19 |
| LR1e-5 E5 | 6.83% | 90.16% | 74.17% | 60.23% | 55/750/6 |
| LR3e-5 E1 | 7.52% | 90.16% | 74.92% | 60.39% | 55/676/6 |
| LR3e-5 E4 | 4.28% | 91.80% | 58.26% | 36.60% | 56/1252/5 |
| LR3e-5 E5 | unavailable | unavailable | unavailable | unavailable | requires approved inference |

| Category | Original TP/FP/FN | Phase5C | LR1e-5 E5 | LR3e-5 E1 |
|---|---|---|---|---|
| Sofa | 11/1/2 | 11/2/2 | 10/21/3 | 11/25/2 |
| Chair | 24/19/15 | 24/24/15 | 37/641/2 | 36/571/3 |
| Bed | 7/5/2 | 7/5/2 | 8/88/1 | 8/80/1 |

Table, Desk, Cabinet and Lamp have no active scoring targets. Empty metrics for
them must not be interpreted as perfect performance or confirmed negatives.
For LR1e-5 E5, Chair contributes 641/750 (85.5%) of scored false positives.
Sofa loses one true positive despite the aggregate recall increase. Bed/Chair
matched IoU also decline in some comparisons; the AP75 aggregate alone does
not prove every category localized better. Per-category AP and matched IoU are
retained in `analysis-summary.json` and the original comparison report.

Standard duplicate FPs at IoU0.5 are only 1,3,4,3 respectively for original,
Phase5C, LR1e-5 E5 and LR3e-5 E1. They cannot explain hundreds of additional FPs.
The corresponding mixed-object heuristic counts are 15,39,73,70; fragmented-GT
flags are 4,3,46,44; category-disagreement flags are 14,11,31,27. These measures
count different entities and must not be added together or called verified
errors. Without saved coordinates, specific failed photographs and overlap
distributions cannot be reconstructed. High-scoring fragments that do not
satisfy duplicate IoU may still matter, but that requires new authorized cached
inference and annotation review to establish.

The pooled scored-event median confidence shifts from 0.01045 original to
0.01850 Phase5C, 0.19726 LR1e-5 E5 and 0.17293 LR3e-5 E1. These quantiles include
only scored TP/FP events above0.001, not all proposals, ignored objects or
unknown unmatched predictions. Per-category confidence distributions are not
recoverable from the saved aggregates and were not invented. CPU weight-only
inspection confirms both class and box heads changed. In LR1e-5 E5 the class
projection changes 1.50% in relative L2 and logit-shift weights 1.23%; LR3e-5 E5
changes them 2.53% and 2.08%. Parameter changes do not establish output-score
causality. The installed class head applies an image-dependent shift and scale
to normalized image/text similarity, so small weight changes can affect many
proposal scores. The observed shift and fixed-threshold failure support a
calibration problem; LR3e-5 E2–E4 also lose ranking/localization quality, so its
regression is not merely calibration.

The same saved PR definitions were used throughout: seven fixed class prompts,
unchanged processor, one full960 FP32 validation forward, max sigmoid class per
query, score floor0.001, class-aware NMS at strict IoU>0.5, cap300, operating
threshold0.15, greedy same-category matching at0.5/0.75, ignored-region handling
and source-specific FP observability. AP remains the existing101-point ranked
interpolation with legacy tie ordering; displayed PR thresholds group equal
scores so operating points are attainable. AP is a macro average of supported
classes, while pooled precision/recall are micro counts. This distinction and
the AP/PR tie convention were preserved. CPU equivalence tests cover native
scope, cached exposed predictions, ties, duplicate/unknown handling, NMS and
full internal aggregate calculations. No metric definition was silently fixed.

| Model | Best precision at original recall≥68.85% | Threshold | Best recall at precision≥80% | Best precision at recall≥90% |
|---|---:|---:|---:|---:|
| Original | 68.85% | 0.167356 | 62.30% | unreachable above score floor |
| Phase5C | 57.53% | 0.153062 | 59.02% | unreachable above score floor |
| LR1e-5 E5 | 73.68% | 0.873266 | 62.30% | 7.07% |
| LR3e-5 E1 | 59.15% | 0.792401 | 67.21% | 15.90% |

These are post-hoc descriptive curve envelopes, not adopted thresholds or a
new independently validated operating policy. The fair threshold-swept original
reference is 68.85% precision, not its lower62.69% fixed-threshold precision.
LR1e-5 has a modest4.83-point matched-recall advantage, but gains no recall over
the original at80% precision. None of the saved curves meets80% precision and90%
recall together. LR3e-5 E4 reaches only31.58% best precision at the original
recall. Threshold adjustment alone is not a convincing solution to the goal.
The old ranking gate did not require improvement over original matched-recall
precision; its LR3e-5 E1 `investigate` status is therefore not evidence of gain.
Historical rankings and decisions remain unchanged.

The optimization pool contains59 photographs/56 known groups/182 positive
boxes: Sofa40 in34 photographs, Chair114 in42, Bed28 in25. Internal validation
has19 photographs/19 groups: Sofa13 boxes in9 photographs, Chair39 in11, Bed9
in6. These per-class image counts overlap. Native FP-observable internal scope
is only1 Sofa,4 Chair and5 Bed photographs; COCO unmatched objects remain
unknown, LVIS NEX is respected, and ignored native instances disable unsafe
exhaustiveness. No four-class taxonomy uncertainty was promoted into training.

The classification term supervises matched positive queries and only explicit
safe native negatives. There are only three negative-category training
photographs: two Sofa and one Bed, with no Chair negatives. Unmatched queries
in positive-only images receive no Chair-negative loss. A synthetic CPU
regression test confirms the positive-only case raises one matched class logit
under descent, with no opposing unmatched-class gradients. This is a plausible
mechanism for score inflation, not an ablation proving causation. Mean
sample-weighted total loss drops0.7656→0.4519 atLR1e-5 and0.6783→0.3919 atLR3e-5.
The historical run did not log separate focal/L1/GIoU contributions; they cannot
be recovered from total loss or final weights. The new box-only loss explicitly
returns its components for future logging.

Some apparent FPs could be missing or semantically mismatched annotations even
where native completeness permits scoring. The existing masks also leave many
unmatched predictions unscored, potentially making observed precision more
optimistic. Neither direction can be quantified from these aggregates. No
annotations were changed, and no unknown object became background. With only19
groups, skewed61 boxes and many explored checkpoints, credible independent
generalization or group-bootstrap uncertainty cannot be established from the
stored aggregates. A future once-only independent evaluation needs separately
approved unexposed real-room data and complete native/category-aware boxes;
exposed historical panels are unsuitable as unrestricted tuning sets.

The isolated `box_only.py` implementation freezes every module except the six
`box_head.dense{0,1,2}.{weight,bias}` tensors:1,184,260 parameters. Class,
objectness, backbone and normalization parameters remain frozen/eval. It rejects
foreign or duplicate optimizer tensors and missing/nonfinite box gradients.
Matching retains the previous Hungarian class/5*L1/2*GIoU cost with class logits
detached. Optimization uses only5*L1+2*GIoU on active native positive boxes;
there is no classification, objectness or unmatched-background loss. Tests use
the installed real OWLv2 box and class head implementations with synthetic
features, verify gradients only in the box head, unchanged frozen logits and
all unchanged weights before any optimizer step. No real encoder was run.
This adapter is CPU-ready, not GPU-validated. Its six-tensor state is deliberately
incompatible with the old twelve-tensor queue; that queue must not be resumed.

The proposed next fit is one original-initialized box-only run atLR1e-5,
AdamW weight decay0.01, seed20261118, FP32, batch1/accumulation4, five epochs,
maximum75 updates,30 minutes including smoke,6.5GiB allocated VRAM stop target,
and2GiB new experiment storage. It reuses exactly the59/19 split and182/61 boxes.
Estimated memory is roughly2–4GiB and runtime5–15 minutes, extrapolated from
Phase10B; the GPU zero-update smoke must measure this before any updates. A
new guarded worker must integrate the adapter with reserve/commit recovery,
frozen fingerprints, pre-metric prediction caching and isolated CPU metrics.
No such training execution is authorized by this phase.

Success is predeclared in `box-only-config.json`: matched precision≥71.85%
at recall≥68.85%, AP75≥57.28%, fixed0.15 precision≥62.69% and recall≥68.85%,
Sofa/Chair/Bed TP at least11/24/7 and no per-category AP75 loss>2 points.
Frozen parameter/logit drift, unsafe gradient/assignment, nonfinite values,
resource violations or failed primary/category gates reject the experiment.
Duplicate/fragmentation regressions trigger investigation rather than silent
acceptance. This is a mechanism-isolation experiment, not an expectation that
box adaptation alone can reach80/90 with incomplete three-class supervision.
No automatic threshold adoption or further fit follows it.

All159 CPU tests pass across seven relevant suites, including16 new tests.
The final preservation check rehashed9,973 historical dataset/model files,
160 Phase10B snapshots, all24 sealed comparison artifacts,27 historical
implementation files and all78 eligible source images. Source IDs, image hashes
and known groups remain disjoint across optimization/internal validation.
Original OWLv2, Phase5C, SigLIP2 and the furniture classifier hashes match their
saved records. No prior results, annotation overlays or frozen partitions were
modified. The Mac simulation project was not accessed. Git backs up only the
small code and aggregate report artifacts; an independent off-workstation
backup of ignored private datasets/weights remains unverified.

The immediate approval request is **inference only** for the exact saved
LR3e-5 E5 checkpoint on the same19 internal photographs: maximum19 forwards,
zero updates,600 cumulative seconds,6.5GiB allocated VRAM and128MiB new output
under `data/open-images/experiments/phase10c-recovery-v1/`. The exact proposal and
unapproved template are under `data/open-images/preparation/phase10c-v1/`.
The command and CPU-scoring step are in `tools/phase10c-v1/README.md`. Successful
raw caches are immutable and reused; uncertain forward boundaries block replay.
Recovery saves raw tensors before metric code, but cannot guarantee avoidance
of the unidentified interpreter fault. No GPU command has been run.

Box-only GPU smoke/training is a **separate** authorization, limited to the
one-fit proposal above after recovery review. Existing panels remain exposed
diagnostics; no reserved scoring, downloads, full-backbone updates, deployment
or production threshold change is included. Rollback is to continue using the
immutable original detector; all new code, manifests and eventual outputs use
separate versioned paths. Nothing needs to be deleted or overwritten.
