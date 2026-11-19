# Phase 10A — backup and expanded-data training preflight

Prepared on `experiment/phase10a-expanded-preflight`, starting at Phase9 commit `275b26bd4bb630a5ca5bb02ec1d45c2f12b6fdf3`. Main and the annotated baseline tag still point to `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`. GitHub main was independently read at that same commit. The Phase9 branch is not present on the remote; pushing it failed because HTTPS credentials are unavailable. Neither GitHub CLI nor Git Credential Manager is installed. No tokens were requested or displayed. The README supplies optional user-run CLI installation, browser login and non-force push commands. No package was installed.

## Backup and integrity

The local ignored `backup-inventory.json` records paths, logical/allocated sizes and SHA256 for **9,973 files /10,490,220,990 bytes** under data and models, including320 checkpoint/tensor candidates. All9,973 files were rehashed after preparation and matched. Frozen files were hashed for backup integrity only; none were decoded or scored. All120 historical training photographs were decoded for integrity and confirmed to match their native dimensions with EXIF orientation1. No development/frozen photos were decoded. The source photographs and annotations are unchanged.

Critical checkpoints independently match their preserved hashes:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Original OWLv2 | 619,918,824 | e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7 |
| Phase5C head delta | 18,984,973 | d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6 |
| SigLIP2 | 1,500,800,904 | 612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b |
| Expanded classifier | 28,649 | 51097ebf696a7765d023c8c02a9e4ea14965e0da04877cdd65be4416b9c13f0d |

No independently verified copy of the ignored data/weights is available. A proposed destination is a mounted encrypted external drive at `/mnt/e/FurnitureAI-backups/phase10a-20261118/`, at least20GiB free. That drive is not mounted here. Approval must precede any copy. Use the local inventory and relative rsync file list, preserve successful partial transfers, never use delete, then verify all destination SHA256 values and a Git bundle. A local Git bundle can protect committed source, but is on the same workstation and is not an independent disaster backup. No private data was copied or uploaded.

## Actual annotation coverage

The prospective inventory reconciles exactly: **120 photographs /114 known source groups /373 boxes /297 instance masks**. Source photographs:20 historical Open Images,9 Phase6A Open Images,45 COCO,40 Phase7 LVIS and6 Phase8C LVIS. COCO/LVIS overlays are not counted as new photographs. Known source/creator components, photo IDs, encoded hashes, saved decoded hashes and historical development/frozen references remain isolated. No additional known conflicts were found; unknown physical-scene relationships remain unverified.

| Category | Prospective photos | Prospective boxes | Prepared photos | Prepared boxes | Prepared masks |
|---|---:|---:|---:|---:|---:|
| Sofa |43|53|43|53|51|
| Chair |53|153|53|153|117|
| Table |45|63|0|0|0|
| Bed |32|38|31|37|19|
| Desk |39|41|0|0|0|
| Cabinet |3|7|0|0|0|
| Lamp |13|18|0|0|0|

Category photo counts overlap. **78 unique photographs /75 source groups /243 boxes /187 masks** form the conservative, still-provisional preparation pool.42 prospective photos have no supported positive after filtering.130 boxes remain inactive:63 Table,41 Desk,18 Lamp,7 Cabinet and one Open Images Bed marked IsInside. These are derived dispositions only, not changes to source annotations, reviewer flags or historical eligibility. All33 previously inactive LVIS overlay annotations remain inactive.

Table/dining-table/desk units, cabinet fronts versus whole cabinets, and lamps versus other lighting units were already explicitly unresolved in the source-readiness policy. They cannot silently become seven broad project-class targets. Native IDs, coordinates, polygons, ignored regions and scope flags are retained. Truncation alone does not establish an incorrect label:78 retained boxes have source/border truncation flags and remain explicitly provisional. Mixed-object/fragmentation concerns are not resolved by this offline audit or by the mere presence of a mask.

## Rights and source completeness

All120 saved photograph-specific CC BY2.0 grants were replayed offline from checksum-verified evidence: exact Flickr photo, creator account, specific licence and attribution. Native annotation attribution remains separate, CC BY4.0. This does not revalidate current ownership, privacy/property releases or every possible commercial right. There were no live creator requests or new dataset/model downloads.

COCO absent categories, LVIS federated missing/nonexhaustive categories and filtered/unresolved instances never become global background. Unknown scope is masked. Explicit negatives are restricted to exactly bound safe native Sofa/Chair/Bed categories: only three prepared photographs provide such negative evidence, all in optimization (two Sofa negatives, one Bed negative). No explicit Chair negatives are available. No global objectness/background loss or instance-mask head is introduced.

## Prepared implementation

The historical Phase5C worker is not suitable unchanged: it hardcodes27 training/18 development photos,70 updates/10epochs and a class-level coverage loss which can turn unmatched instances into negatives. Historical code is untouched. New `tools/phase10a-v1/` contains a checked dataset loader, native target policy, source-group split, square-padding box transforms, strict12-tensor head loading, unknown-safe classification/L1/GIoU loss, native metric scopes and durable recovery-budget scaffolding.

All1,579,526 class/box-head parameters can be trainable; backbone and objectness stay frozen and in eval mode. The actual preserved head delta was loaded into matching CPU test heads without constructing an encoder or running real-image model inference. The pinned local image processor was tested on synthetic images of varying dimensions: bottom/right square padding,960x960 resize,16-pixel patches and existing normalization. The input contract hashes every original model/tokenizer/processor file, both configs, the prepared manifest and implementation. Original model assets were not changed.

Source-native evaluation conversion preserves unknown FP scopes. The19-photo internal validation set has FP-observable native support in only1 Sofa image,4 Chair images and5 Bed images; these counts overlap. Most COCO unmatched detections remain unknown. This small, uneven coverage cannot support an exhaustive real-room precision claim.

## Two conditional experiment configurations

Both proposals start from immutable original OWLv2, with identical seed20261118, prompts/class order,960 preprocessing, AdamW weight decay.01 and head scope. The only configuration difference is LR1e-5 versus3e-5. Immutable original and Phase5C adapted checkpoints are comparison references.

- Source-grouped training-only split: **59 optimization photos /56 groups /182 boxes**, and **19 inner-development photos /19 groups /61 boxes**.
- Optimization positives:40 Sofa,114 Chair,28 Bed boxes. Internal validation:13 Sofa,39 Chair,9 Bed boxes.
- Five epochs maximum, batch1, accumulation4 (normalize the final short group by its actual size).
-150 updates/config ceiling;300 queue ceiling. This exact59-photo plan normally performs75 updates/config,150 total.
- One-hour queue wall limit including resumed execution;30minutes/config;6.5GiB allocated-memory stop target. OOM/nonfinite failures stop rather than silently alter scope.
- Conditional verified FP16, finite-gradient checks, clipping1, no backbone/LoRA adaptation.

Historical Phase5C runs used about218seconds for70 updates and peaked at1.93GiB allocated VRAM. For this proposal, estimate **5–15minutes/config and2–4GiB allocated VRAM**, not measured guarantees. No GPU forward/backward smoke test is authorized in Phase10A. A future worker must validate these bounds before updates.

Recovery scaffolding durably reserves update budget before a call, records checksum-bound committed checkpoints, and blocks uncertain interrupted boundaries, changed bindings or corrupt snapshots. Tests use temporary CPU fixtures only. **A reviewed expanded GPU execution worker and separate explicit approval are still required.** The Phase10A authorization gate always refuses training; do not invoke the old Phase5C worker with these files.

## Selection and future independent evaluation

Use only the new training-derived inner development group for configuration selection. Primary goal: precision at reference-model matched recalls, with full PR curves, fixed.15 P/R, AP50/AP75, duplicates, matched IoU, fragmentation/mixed-object heuristics, per-category regressions, timing and memory. A recall level the candidate cannot reach fails qualification. Predeclared category-localization/recall gates supplement aggregates; group-level uncertainty and scope coverage must accompany results.

The18- and36-photo panels are exposed and are not unrestricted tuning pools. After selection, seal config/code/checkpoint/threshold and request separate permission for one reserved-set comparison. First audit that set's native seven-class coverage and licensing without changing membership. Missing Cabinet units and exhaustive FP coverage may still prevent a seven-class conclusion. No reserved scoring is performed or approved here.

## Verification and next authorization

**121 tests passed:**32 Phase10A,18 Phase5B,12 Phase5C,25 Phase8A and34 Phase9. Tests include native mapping, source/creator isolation, hash consistency, withheld access rejection, partial/NEX gradients, ignored regions, variable dimensions, actual local preprocessing, actual delta keys/shapes, frozen scope, non-overwrite behavior,150/300 update caps, time bounds, corrupt checkpoints and uncertain-resume rejection. No actual model training or new model inference occurred.

Recommendation: do not approve a broad seven-class precision-improvement run yet. Resolve Table/Desk/Cabinet/Lamp supervision and obtain adequate FP-observable validation support first. Alternatively, separately approve the explicitly limited Sofa/Chair/Bed head-only diagnostic above, without claiming it addresses the four unsupported categories. Neither model promotion nor reserved scoring is included. External backup destination/copy and any CLI installation need separate consent.

Rollback after saving work: `git switch main` restores baseline code, or `git switch experiment/phase9-multiview-detection` restores Phase9. No model rollback or dataset deletion is needed. Preserve all receipts; resume future training only from a checksum-bound committed cursor with matching data/config/code/processor/base, optimizer/scaler/RNG and cumulative budget. Never replay an uncertain optimizer boundary or delete historical budget records.
