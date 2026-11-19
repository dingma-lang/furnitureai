# Phase9 — inference-only furniture precision experiments

## Backup and scope

Starting commit **e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6**, verified equal to GitHub main before changes. Experimental branch **experiment/phase9-multiview-detection**; local baseline tag **phase9-baseline-20261118**. Production detector defaults and main were not changed. Git backs up tracked code only. The original OWLv2 (619,918,824 bytes), SigLIP2 (1,500,800,904 bytes), expanded classifier and selected Phase5C delta were directly SHA-256 verified before/after. No independent off-workstation backup of these ignored weights or source datasets was verified. No large model copies were made. Raw photos, licensed datasets, model weights and unsanitised experiment artifacts are excluded from Git.

## Fixed experiment protocol

Fifteen predeclared arms, using only **18 previously exposed development photographs /41 native boxes** for selection. Both base and selected Phase5C models remain frozen. CUDA smoke used the first exposed photo; its16 cached passes were reused. Eight passes/image/model: full-frame768/960/1152, horizontal-flip960, four overlapping62.5%-extent tiles960. These are transformations of one photograph, **not new physical viewpoints or independent captures**. Upsampling creates no new source detail. Positional-embedding/box-bias interpolation gives actual48/60/72 patch grids; recorded pixel tensors and hashes verify genuine model-resolution changes. No vertical flips.

Confidence.15, class-aware NMS IoU.5, AP floor.001, max300 per-view detections, one-to-one same-native-category matching IoU.5/.75 and101-point AP. Historical original/Phase5C baselines were exactly reproduced at operating counts/AP level. All original predictions remain intact; new per-view/fused predictions are separate. Custom confidence-weighted averaging uses complete-link same-class IoU>=.55 and at most one member/view; it is not claimed to be an external WBF implementation. Scores remain uncalibrated. Ambiguous cluster assignments are flagged rather than chained. Adjacent-object tests are included; unknown physical duplicates remain possible.

The tile method excludes internal-boundary fragments, preserves full-frame boxes, and adds only unmatched interior proposals with score>=.25 and area<=20% of the image. Scale/flip consensus requires at least two views. Original-versus-adapted agreement is correlated model agreement, not independent verification.

The optional frozen SigLIP cosine head checks only crops with minimum32-pixel side, aspect<=4, away from photo edges. Unsuitable/uncertain crops remain advisory, not forced categories. A different category with relative softmax margin>=.10 may abstain from that proposal; detector labels are never replaced. The head/encoder/preprocessor fingerprints are pinned. No encoder/head/detector updates occurred.

Selection is fixed-confidence precision improvement >=3pp over adapted baseline, loss of at most one matched GT object, AP75 loss<=2pp, and no matched-object loss for tiny-support Sofa/Chair/Lamp. Rank by precision, thenAP75/F1. Full PR curves and precision envelopes at historical reference recalls are supplementary diagnostics, not additional threshold searches. **Selected: adapted-flip_agreement.** Qualified improvement on the exposed protocol: **True**. If none qualifies, the baseline is retained and no robust winner is declared. Qualification failures are recorded in the immutable selection receipt.

## Exposed development results

| Method | P % | R % | F1 % | AP50 % | AP75 % | TP/FP/FN | Duplicate FP | Seconds/photo | Peak GiB |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| adapted-baseline | 30.21 | 70.73 | 42.34 | 79.56 | 53.60 | 29/67/12 | 3 | 0.378 | 2.420 |
| adapted-flip_agreement | 35.44 | 68.29 | 46.67 | 79.99 | 54.91 | 28/51/13 | 4 | 0.752 | 2.420 |
| adapted-scale_agreement | 44.26 | 65.85 | 52.94 | 77.57 | 53.49 | 27/34/14 | 3 | 1.222 | 3.117 |
| adapted-scale_nms | 20.55 | 73.17 | 32.09 | 80.63 | 50.56 | 30/116/11 | 7 | 1.175 | 3.117 |
| adapted-scale_weighted | 34.94 | 70.73 | 46.77 | 79.13 | 55.03 | 29/54/12 | 5 | 1.223 | 3.117 |
| adapted-siglip_consistency | 30.11 | 68.29 | 41.79 | 77.98 | 53.51 | 28/65/13 | 3 | 0.975 | 2.420 |
| adapted-tiled_guarded | 29.90 | 70.73 | 42.03 | 79.42 | 53.59 | 29/68/12 | 3 | 1.731 | 2.420 |
| crossmodel_agreement | 27.27 | 73.17 | 39.74 | 74.69 | 51.87 | 30/80/11 | 2 | 0.814 | 2.420 |
| original-baseline | 21.71 | 68.29 | 32.94 | 72.16 | 51.91 | 28/101/13 | 2 | 0.438 | 1.842 |
| original-flip_agreement | 22.50 | 65.85 | 33.54 | 70.67 | 53.27 | 27/93/14 | 3 | 0.800 | 1.842 |
| original-scale_agreement | 23.64 | 63.41 | 34.44 | 70.71 | 46.54 | 26/84/15 | 0 | 1.255 | 3.116 |
| original-scale_nms | 17.07 | 68.29 | 27.32 | 71.99 | 46.11 | 28/136/13 | 0 | 1.219 | 3.116 |
| original-scale_weighted | 21.49 | 63.41 | 32.10 | 70.75 | 54.05 | 26/95/15 | 1 | 1.256 | 3.116 |
| original-siglip_consistency | 21.88 | 68.29 | 33.14 | 72.19 | 51.94 | 28/100/13 | 2 | 0.912 | 1.842 |
| original-tiled_guarded | 21.71 | 68.29 | 32.94 | 72.16 | 51.91 | 28/101/13 | 2 | 1.746 | 1.842 |

| Category | Adapted baseline TP/FP/FN | Selected TP/FP/FN | Baseline AP75 % | Selected AP75 % |
|---|---|---|---:|---:|
| sofa | 2/4/0 | 2/4/0 | 50.50 | 50.50 |
| chair | 2/0/0 | 2/0/0 | 100.00 | 100.00 |
| table | 7/27/1 | 7/20/1 | 36.80 | 23.10 |
| bed | 7/9/0 | 7/6/0 | 58.81 | 81.19 |
| desk | 2/4/7 | 2/4/7 | 12.58 | 13.73 |
| cabinet | 6/21/4 | 5/15/5 | 43.42 | 43.60 |
| lamp | 3/2/0 | 3/2/0 | 73.07 | 72.28 |

Matched-recall envelope for selected method at recall>=29/41: **{'fp': 52, 'precision': 0.35802469135802467, 'recall': 0.7073170731707317, 'threshold': 0.1478632539510727, 'tp': 29}**. This envelope describes the observed PR curve; it was not used to retune the final operating threshold. Final confidence remains.15. AP50 is not operating precision.

Desk/Cabinet sensitivity (excluded categories, same predictions): adapted baseline **{'ap50': 0.9646864686468646, 'ap75': 0.6383498349834984, 'precision': 0.3333333333333333, 'recall': 0.9545454545454546}**; selected **{'ap50': 0.9742574257425742, 'ap75': 0.654125412541254, 'precision': 0.39622641509433965, 'recall': 0.9545454545454546}**. These are diagnostic aggregations, not repaired labels or replacement selection protocols. Per-category JSON contains AP50/AP75, P/R/F1, matched IoU, duplicates, FP-observable images and unknown ignored counts. Mixed-object/fragmentation counts are geometric candidates, not human-confirmed errors.

Runtime is the sum of measured required pass/verification stages plus measured cache-read/fusion overhead per photograph, excluding model loading, scoring, stage writes and galleries. Shared caches were used across ablations; this is not15 independently rerun end-to-end timing benchmarks. Cache-reading/fusion overhead is also recorded separately in each metric JSON. Model load records and smoke cold-start costs are retained. Peak VRAM means PyTorch allocated memory, not total/reserved GPU memory. Batch one, sequential models,75% allocator fraction,6.5GiB stop target and30-minute invocation bounds were enforced. CUDA/model errors are fatal; no CPU fallback or recipe weakening. Caches are immutable/checksummed and resumable. Actual detector passes: **396**.

## One locked comparison on36 fresh-development photographs

No recorded prior model use was found in inspected local model-evaluation evidence; the18 exposed IDs/hashes are disjoint. Unrecorded external use cannot be ruled out. Exact36 membership, native126 boxes, source groups and hashes were locked as metadata before selection; no fresh pixels entered tuning. Selected configuration and non-test code hashes were sealed before any final pixels were opened. Original baseline, adapted baseline and the one selected recipe were compared once. **No configuration/threshold was retuned from these results.**

| Method | P % | R % | F1 % | AP50 % | AP75 % | TP/FP/FN | Duplicate FP | Seconds/photo | Peak GiB |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| adapted-baseline | 50.72 | 84.13 | 63.28 | 78.39 | 61.69 | 106/103/20 | 6 | 0.369 | 1.842 |
| adapted-flip_agreement | 59.64 | 78.57 | 67.81 | 78.17 | 63.28 | 99/67/27 | 3 | 0.722 | 1.842 |
| original-baseline | 53.33 | 76.19 | 62.75 | 78.78 | 63.68 | 96/84/30 | 2 | 0.339 | 2.420 |

**The fresh-set precision/AP are not directly comparable to the18-photo values.** Source taxonomies and annotation completeness differ. COCO completeness-unverified and LVIS NEX/filtered subsets do not turn unmatched predictions into FPs. Exact native coverage with all source targets retained is required for unmatched-FP observability; source scopes and exclusions remain immutable. Retained annotated positives still contribute recall. Ignored/unresolved objects and other unlabelled furniture remain unknown. Results therefore measure **conditional native-source annotation agreement**, not exhaustive real-room precision. Cabinet has no fresh positives; no seven-class independent performance claim is possible. The fresh set is now exposed by this one comparison and must not be called untouched for future searches.

## Visual evidence

Local gallery: `galleries/exposed/index.html`, all18 exposed photographs, blue native GT, green source matches, red source-relative FPs, orange unscored proposals. Panels include original/adapted baselines, multi-scale NMS, guarded tiles, weighted fusion and the selected/cross-model method. Duplicates and mixed-object candidates are labelled; missed source annotations are counted. Creator/source/license evidence is retained in the gallery attribution JSON. Plausible appearance is not proof of correctness; native Cabinetry run/front and Desk/Table ambiguities remain. No photo/gallery is pushed to Git.

Full per-method predictions and case diagnostics: `evaluations/exposed/`; one final comparison: `evaluations/final/`. PR curves/CSVs: `curves/exposed/`, `curves/final/`. JSON events identify new/lost matches and source FPs; localization means use matched objects and can have survivor bias. Raw per-view source coordinates, crop transforms, tensor sizes, inference times and memory are under `passes/`.

## Genuine multi-photograph views

No independently verified grouped same-room/capture set was identified in the local authorised room data. Different creators/photos or matching product IDs do not prove physical identity/capture independence. The existing ABO capture/product evidence does not establish registered real-room views. `crossview.py` therefore implements only a guarded interface: require same-scene provenance, reject duplicate captures, retain predictions in their own image coordinates, and require independent object correspondences before object-level consistency. Cross-camera box fusion without verified registration is deliberately rejected. No unseen viewpoint was generated, no external image was downloaded, and no genuine-view accuracy is claimed.

## Recommendation and conditional next training proposal

Read exposed and final results together: an exposed-set qualification is not proof of broad improvement. Do not promote any method into the application's default. Prefer lower false positives only when recall, Bed/Table localization and rare-category support are maintained; distinguish Cabinet-driven source agreement and unmatched/unknown annotation gaps. Keep SigLIP as advisory unless its matched comparison qualifies, and retain safe abstention for mixed/partial crops. A method adding boxes or agreeing with itself is not automatically stronger evidence.

If further training is approved later, propose a source-native, partial-label **head-only** study using the prepared120-photo training pool and training-only grouped inner validation, original and Phase5C immutable references, at most two configs, five epochs/150 optimizer updates each (300 total), batch1/accumulation4, conditional FP16, soft6.5GiB VRAM target and one-hour wall cap. Exact target IDs, Cabinet unit semantics and LVIS/COCO completeness masks must be locked/tested first. No full-backbone/LoRA updates or new weights/data. The36-photo panel cannot be repeatedly optimised after this comparison; reserve a genuinely untouched, separately approved final evaluation. Cabinet still has zero fresh native positives, so a broad seven-class claim needs better real Cabinet evidence. This is a proposal only; **no training was performed or authorised by the implementation**.

## Verification, preservation and rollback

64 tests passed. Historical stat snapshot and strong critical dataset/registry/model hashes passed; frozen photographs received stat-only preservation, no byte reads, decoding or scoring. The previous33 inactive LVIS labels remain inactive. Original model/configuration/source files and historical predictions were not overwritten. There were no installations, dataset/model downloads, production changes or Mac project modifications. New disk allocation is recorded in `completion.json`;512MiB phase budget includes raw artifacts, galleries, source and sanitised report.

Only new Phase9 source/tests, `.gitignore` report allowlisting, and the small sanitised report/protocol/summary/test files are committed. The data directory remains ignored. Git commit and push receipts are reported in chat after completion; no force-push or merge into main.

After the experimental commit is saved, rollback code with `git switch main`. The original production detector has not changed; there is no weight rollback or dataset deletion to perform. The baseline tag is a code reference, not a model/data backup. Keep ignored experiment artifacts for provenance.


## Post-comparison interpretation and targeted visual review

**No tested recipe demonstrated a reliable general improvement at comparable recall. Keep the original production default unchanged.** The immutable selection receipt's `robust_improvement_on_exposed_protocol=true` means only that horizontal-flip agreement passed the predeclared limited aggregate gates; it is not evidence of robust generalisation. Those gates did not sufficiently protect category-level localisation.

On the exposed panel, adapted flip agreement reduced FP67 to51 and improved precision30.21% to35.44%, but lost one Cabinet match and reduced Table AP75 from36.80% to23.10%. Bed AP75 improved58.81% to81.19%. The mixed-object geometry flag increased16 to17, and duplicate FPs increased3 to4. Fewer predictions did not solve object-unit ambiguity.

The single locked fresh comparison reduced FP103 to67 and improved conditional precision50.72% to59.64%, but recall fell84.13% to78.57%: five additional missed Chairs, one Desk and one Table. At the same fixed.15 threshold, six-supported-class AP50 fell78.39% to78.17%; AP75 increased61.69% to63.28%. Cabinet had zero fresh positive annotations. These are source-native, partial-annotation results, not verified exhaustive accuracy.

As a descriptive matched-recall check using the already saved fresh PR curves, reaching106/126 recall gave a best observed precision envelope of52.22% for adapted baseline (97FP) versus34.53% for flip agreement (201FP). Those points occur at different recorded thresholds; they were not deployed, selected or used to retune anything. The experimental operating threshold remains.15. This check argues against a dependable precision gain at baseline recall. Full curves, including ties, remain available locally.

Weighted scale fusion is a secondary research candidate: exposed precision34.94%, unchanged70.73% recall, FP54 and AP75 55.03%. Table AP75 improved36.80% to68.32%, but Bed declined58.81% to54.46%, Cabinet43.42% to28.34%, and Lamp73.07% to66.34%. It was not selected and was not evaluated on the fresh panel; do not substitute it after seeing fresh results. Naive multiscale NMS added false positives; guarded tiling and SigLIP filtering showed no useful measured improvement. Original/adapted agreement did not improve precision over adapted baseline.

Targeted visual inspection used only three existing exposed-panel gallery derivatives, without creating human-review flags or changing annotations:

- `galleries/exposed/ad6910a9fd91a322.jpg`: tilted bedroom. The adapted baseline's highest-ranked matched Bed proposal includes much of the wall/bookshelf. Flip agreement moves the matched box onto the bed; the native matched IoU improves0.517 to0.914. Duplicate bed proposals and many uncertain small lamp-like proposals remain.
- `galleries/exposed/ee7a7243c3badd23.jpg`: crowded dining interior. Multiscale NMS produces a visibly denser collection of overlapping furniture proposals. Flip agreement changes one matched Table box from IoU0.810 to0.713, crossing below.75 despite retaining its IoU.5 match. Several Desk/Table labels and mixed-object boxes remain unresolved; missing native labels do not prove false positives.
- `galleries/exposed/9bc242fe94eb6537.jpg`: kitchen cabinetry. The native annotation treats the lower cabinetry run as one unit. Adapted and fused predictions also contain individual doors/drawers and larger mixed regions. A previously matched upper Cabinet proposal is lost under flip agreement. Plausible storage appearance does not resolve the source's object-unit definition.

These observations support better native annotation-unit handling and crop abstention research, not automatic relabelling or production promotion. No fresh or frozen photograph was opened for this visual gallery review.

All64 tests passed; 12,979 historical file stat records, four critical model hashes and three critical dataset/registry hashes passed preservation checks. No frozen image bytes were read. The 396 detector passes peaked at3.117GiB allocated VRAM. Timing tables sum measured view/verification work plus cache-read/fusion overhead; they exclude model loading and are not independent end-to-end production latency measurements. No parameter updates, downloads or installations occurred.

The next possible training study remains a separate approval request: at most two head-only configurations on the prepared120-photo training pool, training-only grouped validation, maximum five epochs/150 updates each,300 total updates, one-hour wall bound, batch1/accumulation4 and6.5GiB allocated-memory stop target. Lock source-native loss masks, exact IDs, group separation and Cabinet units before approval. Do not reuse the now-exposed36-photo comparison for tuning; keep reserved frozen material untouched. No new training is executed by this report.
