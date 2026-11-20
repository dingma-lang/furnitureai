# Phase 10B — dataset recovery and continuous training preparation

Client date:2026-11-19, America/Los_Angeles. Branch `experiment/phase10b-recovery-training-prep` starts from `e12cb3c`. Phase9 and Phase10A branch tips were verified against GitHub; main remains `e14c4c2`. The attachment ends mid-sentence at “2. Full model”; its complete preceding steps1–7 were implemented. No training, model inference, downloads, cleanup, external backup transfer or default-model replacement occurred.

## Environment and preservation

Windows11 build26200, Ubuntu WSL, RTX4070 Laptop8188MiB (7948MiB free at inspection), NVIDIA595.79. Project `.venv/bin/python` is Python3.14.4; Torch2.14.1+cu132 imports and reports CUDA13.2 available. Windows C: has about515GiB free; WSL reports925GiB logical free. These capacities are not additive: the expanding WSL disk remains backed by Windows storage. No packages, drivers or power settings were changed.

All9,973 historical data/model files matched their Phase10A byte hashes. Phase10A artifact seals and source hashes passed. Original OWLv2, Phase5C heads, SigLIP2, classifier, source annotations, existing partitions and separate33 inactive LVIS overlays remain unchanged. Reserved photos were not decoded/scored. No Mac application files were modified. Code Git backups do not independently back up ignored weights/data; that risk remains.

## Dataset lineage

Cohort rows below are representations and must not be summed into independent photographs. Existing COCO/LVIS overlays are not new images. Detailed per-file classification, counts and original per-category metadata are retained locally under `data/open-images/preparation/phase10b-v1/`.

| Dataset | Photographs / annotations | Training and evaluation use | Exclusions and future use | Allocated GiB |
|---|---|---|---|---:|
|ABO Phase1 metadata|0 photos; Listings/image metadata, not detection boxes|[]; Candidate selection only|Accessories, category conflicts, shared identities screened; Metadata for classification/retrieval expansion; no room-detection negatives|0.105|
|ABO original image pilot|624 photos; {"diagnostic_only": 24, "flagged": 74, "provisional_candidate": 461, "quarantined_product": 18, "rejected_for_whole_object_learning": 47}|["Phase2A", "auto-training-v1", "weak-training-v1"]; 402-feature cache:275 weak train,59 validation,68 test; weak training later288 views/130 products; official reviewed evaluator remains gated|{"component": 1, "detail": 32, "diagram": 16, "multipack": 1, "packaging": 1, "scope_mismatch": 1, "swatch": 3, "wrong_class": 4}; Frozen-feature catalogue classification/retrieval, not detector training|0.010|
|ABO Phase3 expansion|1484 photos; Weak product categories; no scene boxes/masks|["phase3-gpu-v1"]; 350 newly eligible vectors combined with182 baseline training images;106-image45-product comparison panel excluded from updates|{"provisionally_eligible": 350, "quarantined": 304, "rejected": 605, "unresolved": 225}; Useful weak catalogue supervision, with identity/filtering bias limitations|0.027|
|Open Images Phase5|118 photos; 63eligible:27train/18dev/18reserved|["Phase5B original baseline; Phase5C two head fits"]; 18development exposed;18reserved untouched|32rights slots not acquired;50 unresolved and5scene quarantines; Real-room native positives; partial scope and unit policy required; no absent-label negatives|0.059|
|Open Images Phase6A|41 photos; 52eligible unmasked boxes|["No new fitting; included in prospective expanded pool"]; 9prospective train/5freshdev now exposed|27quarantined; Real-room native positives; partial scope and unit policy required; no absent-label negatives|0.017|
|COCO Phase6D|72 photos; 362native boxes/masks;280eligible boxes/masks|["No detector fit yet"]; 45prospective train/10freshdev now exposed|16quarantined/1rejected; Real-room native positives; partial scope and unit policy required; no absent-label negatives|0.026|
|LVIS Phase7|66 photos; 81eligible core boxes/masks:63train/18dev|["No detector fit yet"]; 40prospective train/8freshdev now exposed|18quarantined;33existing-photo overlays stay inactive; Real-room native positives; partial scope and unit policy required; no absent-label negatives|0.015|
|LVIS Phase8C|27 photos; 99native;39provisional and60inactive|["No detector fit yet"]; 6prospective train/13freshdev now exposed|8quarantined; category/unit and visual-registration restrictions; Real-room native positives; partial scope and unit policy required; no absent-label negatives|0.016|
|Open Images metadata|0 photos; Native source categories/boxes/masks; exact class counts preserved in local audit reports|[]; Metadata and candidate auditing only|Rights-unverified photos not eligible; Reusable native annotation inventory; no photograph downloads authorized|0.201|
|COCO2017 metadata|0 photos; Native source categories/boxes/masks; exact class counts preserved in local audit reports|[]; Metadata and candidate auditing only|Rights-unverified photos not eligible; Reusable native annotation inventory; no photograph downloads authorized|0.733|
|LVIS metadata|0 photos; Native source categories/boxes/masks; exact class counts preserved in local audit reports|[]; Metadata and candidate auditing only|Rights-unverified photos not eligible; Reusable native annotation inventory; no photograph downloads authorized|0.427|
|Combined Phase10B eligibility|78 photos; 243boxes/187masks;130targets remain inactive|[]; 59optimization/19training-derived inner validation; not trained|42photos without supported positives; four broad categories unsupported; Ready for bounded three-category diagnostics after separate GPU approval|0.000|

ABO metadata:147,702 listings /145,615 item IDs /398,212 image-metadata IDs. COCO2017 metadata:118,287train images /860,001 annotations and5,000val images /36,781 annotations. LVIS metadata:100,170train images /1,270,141 instances and19,809val images /244,707 instances. These are source metadata totals, not acquired or rights-cleared photo counts. Exact furniture class counts, native IDs, flags and historical rights-audit exclusions are preserved in `metadata-source-counts.json`. Open Images metadata here remains the previously acquired validation/test source metadata; the full training split was not downloaded.

Licensing: ABO saved CC BY4.0 documentation; Open Images/COCO/LVIS photographs use preserved photograph-specific CC BY2.0 evidence, with native CC BY4.0 annotation attribution kept separate. Source copyright evidence is not a warranty of current ownership, privacy/property releases or other rights. No paused provider requests resumed. All source licensing ledgers remain local.

## Models and experiments

| Model family | Data/settings | Improvement | Regression / qualification | Retention |
|---|---|---|---|---|
|Original OWLv2|No local detector updates; Pinned original processor;960pixel baseline|Immutable production/reference detector|Exposed18:P21.71 R68.29 AP50 72.16 AP75 51.91; partial native labels|essential immutable default|
|SigLIP2 encoder|Frozen in all local classifier experiments; 224pixel preprocessing;768-dimensional normalized embeddings|Reusable catalogue/crop features|Not independently validated as a real-room verifier|essential pinned encoder|
|Phase2A classifier|275images/130products; {'classes': ['sofa', 'chair', 'table', 'bed', 'desk', 'cabinet', 'lamp'], 'feature_scope': 402, 'full_experiment_cpu_fallback': False, 'gpu_required': True, 'learning_rate': 0.001, 'main_epochs': 50, 'model_revision': '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2', 'model_weights_sha256': '612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b', 'optimizer': 'AdamW', 'proposal_not_finalized': True, 'seed': 20261103, 'smoke_epochs': 3, 'source_files': {'data/abo/curation/phase1c-v1/inventory-v2/inventory.json': 'e0a027ce58d69e6e8dff6c7e301b0ef89f6cc83d5ff00ebfd8cb9d9f1483ce68', 'data/abo/training-prep/phase1d-v1/proposal-v2/classification-weak-train.json': '5f75abc3ac31faf619cbf3b9c45d0c2b36e5670e6c613f3d5aadfa99648abe14', 'data/abo/training-prep/phase1d-v1/proposal-v2/eligibility-inventory.json': 'd3c8478385a6d0e93a3beed37b82b8a637de2b6895a5efff55293fbf5287b166', 'data/abo/training-prep/phase1d-v1/proposal-v2/split-proposal.json': 'c090eed097212f3e1bfd4f3999deda38a91269f09440d8c84b9b78a69610a3a4', 'data/abo/training-prep/phase1d-v1/proposal-v2/weak-classification-eligibility.json': '8cadeb5c5e307e792cacfb6adebb4149724d18692bb044af878dd08b35329bec'}, 'training_images': 275, 'training_products': 130, 'validation_checkpoint_selection': False, 'weight_decay': 0.0001}|Completed50epochs; training agreement improved|97.82%image/99.23%product are weak training metrics, not verified accuracy|historical comparator|
|auto-training-v1|275train views; {'configs': [{'kind': 'linear', 'lr': 0.001, 'name': 'linear-lr1e3-wd1e4', 'wd': 0.0001}, {'kind': 'linear', 'lr': 0.0003, 'name': 'linear-lr3e4-wd1e4', 'wd': 0.0001}, {'kind': 'linear', 'lr': 0.001, 'name': 'linear-lr1e3-wd1e2', 'wd': 0.01}, {'kind': 'linear', 'lr': 0.003, 'name': 'linear-lr3e3-wd1e3', 'wd': 0.001}, {'dropout': 0.2, 'hidden': 64, 'kind': 'mlp', 'lr': 0.001, 'name': 'mlp64-d20-lr1e3', 'wd': 0.0001}, {'dropout': 0.3, 'hidden': 128, 'kind': 'mlp', 'lr': 0.0005, 'name': 'mlp128-d30-lr5e4', 'wd': 0.001}, {'dropout': 0.4, 'hidden': 128, 'kind': 'mlp', 'lr': 0.001, 'name': 'mlp128-d40-lr1e3', 'wd': 0.01}, {'dropout': 0.4, 'hidden': 64, 'kind': 'mlp', 'lr': 0.0003, 'name': 'mlp64-d40-lr3e4', 'wd': 0.001}], 'fold_count': 3, 'max_epochs': 65, 'patience': 12, 'scope': 'TRAIN partition only; locked holdout not used', 'signature': {'code_sha256': 'd551d3e3cc63e568f173294cc1706047d9b91499682a29311f44932bcef10539', 'feature_sha256': 'b8dedef6c7d6fbd8e69949eee0821f45941a6ca0219917e57bd1c1b67a828c5b', 'model_processor_fingerprint': 'b5d0736ddf111580ada6d0bd9d207c423a2324d250962ddbbb6d7095229dee91'}, 'timestamp_utc': '2026-11-04T20:28:55.515390+00:00'}|Best internal macroF1 about83.43%|Adaptive training-derived weak-label selection; no official holdout accuracy|historical comparator, superseded for selected catalogue head|
|weak-training-v1|288images/130products; {'configs': [{'kind': 'linear', 'lr': 0.003, 'name': 'linear-reference', 'wd': 0.001}, {'kind': 'linear', 'lr': 0.01, 'name': 'linear-lr1e2', 'wd': 0.001}, {'kind': 'linear', 'lr': 0.003, 'name': 'linear-smooth05', 'smoothing': 0.05, 'wd': 0.01}, {'dropout': 0.2, 'hidden': 64, 'kind': 'mlp', 'lr': 0.001, 'name': 'mlp64-d20', 'wd': 0.001}, {'dropout': 0.4, 'hidden': 64, 'kind': 'mlp', 'lr': 0.003, 'name': 'mlp64-d40', 'wd': 0.01}, {'dropout': 0.3, 'hidden': 128, 'kind': 'mlp', 'lr': 0.001, 'name': 'mlp128-d30', 'wd': 0.01}, {'kind': 'cosine', 'lr': 0.003, 'name': 'cosine-head', 'wd': 0.001}, {'kind': 'centroid', 'name': 'nearest-centroid'}], 'policy': {'batch_size': 4, 'class_prompts': ['a photograph of a complete sofa', 'a photograph of a complete chair', 'a photograph of a complete table', 'a photograph of a complete bed', 'a photograph of a complete desk', 'a photograph of a complete cabinet', 'a photograph of a complete lamp', 'a close-up photograph of fabric or wood texture', 'a furniture dimension drawing or diagram', 'a photograph of a furniture part or accessory', 'a collage of multiple furniture products', 'a photograph of packaging', 'a blank image'], 'encoder_adaptation': False, 'extra_view_margin': 0.01, 'folds': 3, 'human_review_required': False, 'max_epochs': 80, 'max_local_images': 5000, 'max_new_image_features': 256, 'max_new_storage_bytes': 1073741824, 'max_queue_seconds': 1800, 'max_white_fraction': 0.99, 'min_gray_std': 8, 'min_laplacian_variance': 8, 'min_side': 96, 'patience': 12, 'seed': 20261104, 'strong_contradiction_margin': 0.06}, 'runtime_budget_is_per_invocation': True, 'scope': 'PROVISIONAL WEAK-LABEL DEVELOPMENT; not human-verified accuracy; adaptive internal model selection'}|Cosine mean product macroF1 91.44%,foldSD12.49pp|Adaptive training-derived weak-label selection; no official holdout accuracy|historical comparator, superseded for selected catalogue head|
|Phase3 expanded classifier|532images/281listing IDs/280known groups; {'config': {'balance': 'sqrt', 'kind': 'cosine', 'lr': 0.003, 'name': 'cosine-reference', 'wd': 0.001}, 'fold_balanced_accuracy': [1.0, 1.0, 0.9698412698412698], 'fold_epochs': [12, 6, 5], 'fold_macro_f1': [1.0, 1.0, 0.9722791828054986], 'mean_balanced_accuracy': 0.9899470899470899, 'mean_macro_f1': 0.9907597276018328, 'name': 'cosine-reference', 'sd_macro_f1': 0.016004621269401772}|Matched106-image45-product panel macroF1 79.13%→88.96%|Both models correctly classify 41/45 products; expansion gains both bed examples and one cabinet, loses two tables and one sofa. Macro-F1 improvement reflects class balance, not more total correct products. Single frozen-panel comparison after training-only selection; provisional weak-label agreement, not verified accuracy|selected catalogue classifier, retain baseline and final head|
|Phase5C heads-lr1e-5|27OpenImages train photographs; {'accumulation': 4, 'batch': 1, 'epochs': 10, 'learning_rate': 1e-05, 'updates': 70}|See native18-photo per-category comparison; selected3e-5 AP50 79.56 P30.21 R70.73 AP75 53.60|Table FP increase; Bed/Table localization regressions; Cabinetry annotation influence|historical adaptation comparator|
|Phase5C heads-lr3e-5|27OpenImages train photographs; {'accumulation': 4, 'batch': 1, 'epochs': 10, 'learning_rate': 3e-05, 'updates': 70}|See native18-photo per-category comparison; selected3e-5 AP50 79.56 P30.21 R70.73 AP75 53.60|Table FP increase; Bed/Table localization regressions; Cabinetry annotation influence|essential adapted comparator|
|Phase4/5/9 inference methods|No model updates; Phase4 roomcomparison; Phase5cropgate; Phase9fixed15arms8views/model/photo|Some source-relative FP reduction|No reliable precision gain at comparable recall;36-photo comparison now exposed|historical inference reports, no new weights|
|Phase10A/10B prepared heads|Prepared59optimization/19innerdev; LR1e-5/3e-5,batch1,acc4,≤5epochs,≤150updates/fit,≤300total,≤1hour,6.5GiB|No results: not trained|Four categories inactive; very limited FP-observable validation|pending separate authorization|

Actual historical detector comparison on18 exposed images /41 native objects, confidence.15 and native IoU.5/.75:

| Checkpoint | Precision % | Recall % | AP50 % | AP75 % | FP / FN |
|---|---:|---:|---:|---:|---|
|Original|21.71|68.29|72.16|51.91|101 /13|
|Phase5C LR1e-5|28.04|73.17|79.05|52.95|77 /11|
|Phase5C LR3e-5|30.21|70.73|79.56|53.60|67 /12|

Per-fit per-category precision/recall/AP/FP/FN/localization are retained in the imported canonical history and original Phase5C comparison JSON. Both head fits increased Table false positives and reduced Bed/Table AP75 relative to original, despite aggregate gains. Phase9 did not establish a reliable precision improvement at comparable recall. Its36-photo comparison is now exposed; none of these measurements is independently verified real-room accuracy. No result is claimed to meet80%precision and90%recall simultaneously.

The catalogue history includes the original50epoch Phase2A head, all eight initial automatic recipes, eight weak-label recipes, and Phase3 three-fold/final fitting (29GPU fits,806epochs). The raw archive indexes104 actual saved run/config/completion metadata records; it retains source paths/hashes and unflattened measurements. Seven canonical events seed the new append-only experiment history. Missing historical Git/data hashes remain null; metrics were not recomputed. Each event carries dataset version/hash, Git/checkpoint identity, settings, metrics/per-category values when available, resources, retain/investigate/reject decision, evaluation scope and next action. Classification F1/training agreement and detector AP remain separate metrics.

Other project-local simulator/inpainting/depth-labelled assets and reports are explicitly unresolved where detector provenance is missing. They are not silently declared superseded or safe to remove. All320 checkpoint/tensor candidates remain indexed through the existing backup inventory and new artifact classification.

## Space classification — proposal only

| Classification | Allocated GiB | Treatment |
|---|---:|---|
|essential|3.869|Pinned references, source photos/metadata/licensing, features, manifests and dependencies: retain.|
|historical|3.024|Mostly per-update recovery checkpoints: retain until selected/last-state dependencies and independent backups are verified.|
|reproducible_temporary|0.027|Derived galleries/curves/cache artifacts: only potentially regenerable with preserved inputs/code/attribution.|
|unresolved|2.871|Other model/assets or interrupted-state provenance: no cleanup assumption.|

Conditional cleanup-candidate upper bound: about3.05GiB (historical checkpoint candidates plus about27.8MiB derivatives). This is not an immediately safe deletion amount: selected/last committed checkpoints and unverified dependencies must be retained. There is ample current disk space, so no urgent cleanup is recommended. Nothing was deleted or moved.

## Annotation recovery and current eligibility

All130 inactive targets were inspected across11 contact sheets, with targeted enlarged views. Each has full context, a crop, unchanged native box, native polygon where available, category, scope and exclusion reason. Assistant triage:71partial/fragmented,21visually ambiguous,18taxonomy conflicts,17apparently complete candidates,3shared/multiple-unit ambiguities. All have human_reviewed=false and training_active=false.

The17 apparent recovery candidates occur in14photos:5Table,3Desk and9Lamp targets. This is a prioritised review queue, not17 verified or reactivated labels. Concrete failure cases include headboard/pillows labelled Bed, tabletop-only views, tiny background regions, wall fixtures labelled Lamp, a kitchen island versus dining table, and individual kitchen doors/fronts labelled Cabinet. Cabinet110/111 are filing/storage-unit candidates needing unit/occlusion checks; Cabinet120–123 are individual fronts, and124 is a clipped radio/storage unit. No whole-cabinet reference was verified or activated.

New eligibility preserves78unique photos /75known source groups /243boxes /187masks:

| Category | Prospective photos / boxes | Active preparation photos / boxes |
|---|---|---|
|Sofa|43 /53|43 /53|
|Chair|53 /153|53 /153|
|Table|45 /63|0 /0|
|Bed|32 /38|31 /37|
|Desk|39 /41|0 /0|
|Cabinet|3 /7|0 /0|
|Lamp|13 /18|0 /0|

Category photo counts overlap.42photos have no supported positive under the conservative policy. No absent, filtered, unresolved or LVIS nonexhaustive annotation is converted to generic background. Four unsupported categories remain masked. All prior source-group/rights controls are inherited from checksum-bound Phase10A. The separate33 LVIS overlays are unchanged and absent from this recovery queue.

## Optional review workflow

Run `.venv/bin/python -u -B tools/phase10b-v1/review_server.py --port 8786`, then open http://127.0.0.1:8786. The server is loopback-only, uses exact source membership, source-hash checks, origin/CSRF validation and bounded requests. Choose a target and decision, supply reviewer/evidence and explicitly confirm. A new uniquely versioned local JSON is saved; no previous review, source label or eligibility is overwritten. Submissions remain inactive pending a later eligibility audit. Completing all130 reviews is not a prerequisite for the prepared three-category diagnostics.

## Working diagnostic queue — implemented, not executed

The missing worker is implemented at `tools/phase10b-v1/worker.py`. Two fixed configurations use original OWLv2 initialization, LR1e-5 /3e-5, seed20261118, frozen eval-mode backbone/objectness, twelve head tensors (1,579,526 parameters), batch1 /accumulation4, at most5epochs /150updates each /300total, one hour queue /30minutes per fit, and6.5GiB allocated-VRAM stop target. With59optimization photos, the expected budget is75updates per fit (150total); the last accumulation group has3samples and is normalized accordingly.

Internal validation is19photos /19groups /61boxes, isolated from59optimization photos /56groups /182boxes. Only training-pool images can enter either role. Original and Phase5C comparator predictions will be created on those same internal examples after approval, without opening the exposed18/36 panels or reserved data. Native FP-observable validation coverage remains only1Sofa /4Chair /5Bed photographs. Explicit negative training evidence remains only3photos. These limits prevent broad real-room accuracy claims.

Before any optimizer call, a separately approved `--smoke-only` run must verify finite CUDA forward/backward, frozen/heads weight hashes, memory and a committed cursor0 checkpoint. The smoke makes zero optimizer calls and has not been executed here. The full queue requires its bound smoke receipt. Future smoke/resume commands are in the README.

The worker implements local-only model loading, strict data/code/model/processor/approval bindings, update reservations before optimizer calls, immutable full recovery snapshots, optimizer/scaler/RNG/epoch/group recovery, finite gradients, clipping, frozen-gradient checks, cumulative time precharging, disk/VRAM guards, sequential models, completed-fit skipping, graceful STOP/SIGINT/SIGTERM and explicit failure propagation. An uncertain optimizer boundary or corrupted snapshot blocks automatic replay. No OOM retries, adaptation expansion, CPU fallback, dataset substitution or deployment occur automatically. All snapshots remain versioned; plan roughly3GiB storage,8GiB hard queue cap.

Estimated runtime5–15minutes per configuration and2–4GiB allocated VRAM are extrapolations, not measurements of this worker. GPU/CUDA/AMP execution remains untested until separately approved smoke. The fixture-only rehearsal exercised2x75 simulated update commitments, interrupted recovery and59images x5epochs per fit, with zero model/optimizer calls.

Candidate ranking uses matched-recall precision against both immutable references, with category recall and AP75 guards; unreachable recall fails qualification. It records full PR curves, native AP50/AP75, per-category P/R, FP/FN, matched IoU, duplicate counts, geometric mixed/fragmentation/disagreement flags, latency and memory. These geometric flags are not human-confirmed errors. Results are typed provisional source-native internal agreement, never human-verified accuracy.

## Verification, continuous operation and next authorization

Final CPU checks: **143 tests passed**, including new worker/smoke-gate ordering, source membership, review integrity, partial metrics, GPU/disk/storage guard fixtures, update/time caps, corruption/uncertain-resume rejection, and the prior Phase10A/5B/5C/8A/9 suites. Syntax and source hashes are sealed. No actual GPU model forward/backward or optimizer call was run.

The reusable history and prepared two-fit queue provide a short development cycle after one bounded training approval. Optional annotation review can proceed independently, with recovered labels entering only future versioned datasets. Do not repeatedly retune on the exposed18/36 panels. Preserve reserved evaluation for a separately authorised single comparison after selection is frozen; do not modify its membership.

Exact next approval: authorise the zero-update GPU smoke, then the two prepared Sofa/Chair/Bed head-only diagnostics on the existing59/19 training-derived split, within5epochs /150updates per fit /300total /one hour /6.5GiB, no downloads, no external backup, no reserved scoring, no production replacement. Exact manifest/config/source hashes are in the ignored `training-approval-request-final-v4.json`; no approved authorization file exists. Training may not start from this preparation request alone.

After committing work, rollback code with `git switch experiment/phase10a-expanded-preflight` or `git switch main`. Model/default rollback is unnecessary. Preserve ignored artifacts and receipts. Future background execution, monitoring, STOP and exact-bound resume commands are documented in the README; they are not authorization to run now.

## Historical view-count reconciliation

The preserved Phase3 expansion audit explains the earlier275→288 training-view change exactly:275 -6 redundant known-capture representatives +19 admitted views =288. The six source files were retained; this was manifest-level representative selection, not deletion or six new physical identities. The333-candidate screening log retains each contradiction/quality exclusion and is indexed at `data/abo/experiments/phase3-preparation-v1/expansion-audit.json`.
