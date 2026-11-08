# Furniture AI: ABO training and evaluation plan

Prepared 2026-11-03. Project root: `/home/tristary/furniture-ai-ml`.

## 1. Decision and scope

Start with ABO furniture classification and product retrieval using frozen image features, then a small learned projection or limited image-encoder adaptation. Keep the original OWLv2 detector unchanged during this stage. Detection fine-tuning should follow only after reviewed bounding boxes or synthetic instance annotations exist; room instance segmentation needs a separate mask-producing model and mask evaluation.

This document is a proposed plan, not an implemented training pipeline. This inspection used `.venv/bin/python -B`; it did not train, run model inference, install packages, change the environment, download datasets or weights, or alter existing predictions/checkpoints. Public documentation was consulted online. Subsequent installation, downloads, substantial implementation, and training require separate authorization; approval of this document is not approval to execute those stages.

## 2. Verified workspace and code inventory

### Environment and baseline

- Interpreter: `/home/tristary/furniture-ai-ml/.venv/bin/python`, Python **3.14.4**. Use this explicit interpreter for every future project command.
- Installed package metadata: torch **2.14.1+cu132**, torchvision **0.29.1+cu132**, transformers **4.57.6**, numpy **2.5.4**, Pillow **13.0.0**, safetensors **0.8.0**, scipy **1.18.1**, faiss-cpu **1.15.1**, tokenizers **0.22.2**, sentencepiece **0.2.2**. Package metadata inspection does not establish native import or training compatibility.
- `peft`, `accelerate`, `pycocotools`, and `sam3` distributions were not installed. None is required to write this plan. Basic head training can use PyTorch without PEFT/Accelerate.
- Baseline directory: `models/owlv2-original-baseline-20261103/`, including processor, tokenizer, config, and weights. Manifest identifies `google/owlv2-base-patch16-ensemble`, upstream commit `cfd3195ba4ea9592eec887ded089f4c08eff231d`, original pretrained weights, and `training_performed: false`.
- `model.safetensors`: **619,918,824 bytes**, **154,966,792 parameters**, F32 tensors. Independently computed SHA-256: `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`, matching the baseline manifest.
- Saved OWLv2 preprocessing uses padding/normalization and **960 × 960** input. Preserve this for the primary original-versus-adapted OWLv2 comparison.
- Current session: approximately **14 GiB host RAM**, 12 GiB available, 4 GiB swap; filesystem reports about **943 GiB available**. These are session readings, not guaranteed physical laptop resources or a storage reservation. WSL-backed storage may also be limited by host free space.
- `nvidia-smi` reported GPU access blocked by the operating system. The RTX 4070 Laptop / approximately 8 GB VRAM is user-provided hardware context, not reverified here. Historical predictions record CUDA execution, but that does not establish current GPU availability. Resolve access and measure available VRAM before any approved training.

### Relevant implementation

| Area | Existing files and behavior | Implication for the next stage |
| --- | --- | --- |
| Detection | `src/furniture_bench/detection/owlv2.py`: lazy `Owlv2ForObjectDetection`/processor loading, local-only by default, configurable prompts, confidence and NMS, raw candidates and filter audit | Reuse inference adapter for local baseline and separately saved adapted checkpoints; it is inference code, not a trainer |
| Filtering | `postprocessing.py`: same-class IoU suppression, optional same-class containment and cross-class IoU suppression | Retain audit trail; test adjacent same-category objects before enabling stronger suppression |
| Image embedding | `embedding/siglip2.py`: `google/siglip2-base-patch16-224`, batched `get_image_features`, L2 normalization, inference mode | Retrieval backbone differs from OWLv2; trainable code must bypass inference mode and save its own processor/feature contract |
| Retrieval | `retrieval.py`: normalized cosine / FAISS IndexFlatIP; `product_catalog.py`: multiple gallery photos aggregated to distinct listing identities | Rebuild gallery vectors for each trained embedding model; never mix old gallery vectors with new query vectors |
| Native isolation | `faiss_worker.py`: persistent CPU worker using explicit multiprocessing `spawn` | Preserve isolation from PyTorch/OpenMP; main guard and picklable inputs remain necessary |
| Experiment runner | `pipeline.py`: one record per image, failure reporting, crops, timings, raw candidates, optional retrieval and masks; refuses an existing output directory | Reuse fresh-run safeguards, preserve error records, extend metadata rather than rewriting old predictions |
| Detection evaluation | `evaluation.py`: confidence-ordered one-to-one same-class matching at configurable IoU; precision/recall, category counts, per-pair IoU, ignored regions, failures | Useful fixed-operating-point evaluator; no COCO AP sweep, explicit duplicate summary, or mask metric implementation yet |
| Retrieval evaluation | `evaluation.py`: verified in-catalog R@5 given detection and end-to-end; `scripts/benchmark_abo_products.py`: catalog retrieval R@1/R@5/MRR and synthetic robustness transforms | Extend to R@10, mAP and open-set calibration; current benchmark script forces CPU and is not a training script |
| ABO preparation | `scripts/prepare_abo_subset.py`: bounded catalog-photo selection, product-type mapping, exact/heuristic near-duplicate rejection, gallery/query roles | Script downloads images and writes manifests: do not run during planning. It uses a limited shard selection and is not a full training split generator |
| Retrieval protocol | `scripts/prepare_retrieval_protocol.py`, `evaluate_frozen_retrieval.py`, `dataset_quality.py`, retrieval experiment modules | Existing identity-level development/held-out roles and household-session checks are useful patterns, but not a new 70/15/15 training protocol. Protocol preparation can rename/rewrite manifests; do not reuse against preserved data |
| Segmentation | `segmentation/sam31.py`: optional experimental SAM 3.1 bridge, text prompts and box-to-mask association; rectangular retrieval crops retained | Neither a segmentation trainer nor a verified installed model; ordinary box detection does not produce instance masks |
| Corrections/tests | `corrections.py`, `annotations.py`, `tests/test_benchmark.py`, detection/retrieval/dataset-quality tests | Correction records are not automatically training labels. Human review and split membership must be explicit |

Searches of project `src/`, `scripts/`, and `tests/` found no optimizer/backward/Trainer training loop. In installed Transformers 4.57.6, `Owlv2ForObjectDetection.forward` has no `labels` argument and returns logits, objectness and boxes without computing a detection loss. The separate image/text contrastive loss in `Owlv2Model` is not a bounding-box training objective. A generic Trainer call will therefore not supply detector fine-tuning automatically. This is confirmed in the [versioned implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/owlv2/modeling_owlv2.py).

### What the existing result establishes

`reports/gpu-owlv2-rtx4070-test01/predictions.json` and `reports/gpu-owlv2-rtx4070-evaluation-provisional01.json` cover **3 images**, **16 reference objects**, TP **12**, FP **11**, FN **4**, precision **0.5217**, recall **0.7500**, at IoU **0.5**. Detector confidence was **0.15**, same-class NMS IoU **0.5**, with default prompts for seven categories. Retrieval and segmentation were disabled.

All three annotation records are `assistant_provisional`, not human reviewed; one records predictions seen before annotation. These numbers are a historical smoke-test reference, not an independent accuracy estimate or a model-selection test set. Preserve them byte-for-byte and treat these images as development/regression material. New reviewed annotations must live in a new manifest and require new predictions because evaluation checks the manifest digest.

Only the OWLv2 baseline is present under project `models/`. This GPU workspace contains a small room pilot, not a prepared ABO training collection. Existing ABO scripts do not prove their referenced catalogs or a usable local SigLIP checkpoint are present. Check any external cache read-only later before proposing a download.

## 3. ABO supervision: what is and is not available

ABO has 147,702 listings, 398,212 unique catalog images, 586,584 spin images associated with 8,222 listings, and 7,953 products with 3D models. These are whole-dataset counts, not furniture counts. The official registry identifies CC BY 4.0 and the 2023 license update. Preserve attribution and source provenance. [Official registry](https://registry.opendata.aws/amazon-berkeley-objects/).

| ABO subset/annotation | Classification | Product retrieval | Object detection | Instance segmentation |
| --- | --- | --- | --- | --- |
| Listings: item IDs, product types, multilingual names, attributes, image links | Direct image-level supervision after category mapping and audit | Direct listing-identity positives across linked views; attributes support optional auxiliary losses | No per-instance box from a listing label | No per-instance pixel mask from metadata |
| Catalog images, original or max-256 version | Suitable for object-centric classification; lifestyle/multi-object images need review | Direct same-listing retrieval pairs with distinct captures | No general exhaustively annotated room boxes; derive labels only with additional annotation | No general room-instance masks |
| Spin/turntable images | Extra views with inherited listing label | Strong multi-view identity positives; avoid adjacent-frame leakage and weighting bias | Foreground boxes require masks/manual annotation; product identity alone is insufficient | Do not assume real spin photos carry dense masks |
| glTF/GLB models and textures | Inherit audited listing category for synthetic renders | Synthetic-to-real view matching using linked IDs | Render scenes with camera/instance IDs to generate boxes | Render visible instance-ID masks; these are synthetic labels |
| Material-benchmark renderings: RGB, object masks, camera parameters, normals/depth and material maps | Synthetic category supervision | Synthetic viewpoint pairs where identity mapping exists | Derive tight 2D boxes from provided foreground masks | Direct synthetic single-object foreground-mask supervision; not annotated real room instances |
| `abo-part-labels.tar` | Optional component reasoning | Not necessary initially | Not a substitute for 2D room boxes | 3D part labels are not ready-made 2D room instance masks; projection/rendering and semantic conversion would be needed |
| `abo-mvr.csv.xz` official retrieval experiment split | Optional external reference | Reproduce separately if all required assets/protocol are available | Not room-detection ground truth | Not room-segmentation ground truth |

The official site describes rendered views with geometry/masks and lists the downloadable subsets; archive-specific schemas and identity links must be checked before adopting a particular render package. Different rendering releases are not interchangeable. [Official dataset and downloads](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html). The authors also provide a separate 30-view reconstruction rendering release, distinct from the material benchmark. [Authors' repository](https://github.com/jazcollins/amazon-berkeley-objects).

Classification labels inherited from metadata are weak at image level: a sofa listing may show a room containing chairs/tables, a dimension drawing, fabric detail, or accessories. Keep these records distinguishable from reviewed single-product views. Never label the whole frame as a sofa bounding box or treat unlabelled secondary furniture as background.

For real room detection, collect licensed room photographs with all supported visible instances boxed, agreed category definitions, ignore regions and review evidence. For segmentation, add visible per-instance masks and stable instance IDs; semantic furniture masks do not distinguish two chairs. For end-to-end exact-product retrieval in rooms, also obtain independently verified room-object-to-listing links. Without identity verification, evaluate category detection or visual similarity, not exact manufacturer recognition.

Synthetic scenes can add boxes/masks cheaply, but require a renderer/compositor, multiple instances, occlusion, lighting, scale, room backgrounds and split-safe assets. Record visible versus amodal boxes/masks explicitly; primary evaluation should use visible extent. Synthetic-only performance cannot establish real-room improvement. Detector/SAM pseudo-labels may help training after review, but must remain tagged as pseudo-labels and never serve as independent test truth.

## 4. Initial furniture dataset design

Keep the current taxonomy: **sofa, chair, table, bed, desk, cabinet, lamp**. Treat SHELF→cabinet as an existing preparer policy to review, not a semantic fact. Version a `taxonomy-v1` mapping, source product-type values, exclusions, and table-versus-desk/cabinet-versus-shelf guidelines. Do not infer furniture solely from English titles. Review accessories, swatches, covers, miniatures, bundles, ambiguous types, and non-furniture household products. Preserve unknown/excluded reasons.

Proposed first acquisition, after approval: metadata plus a furniture-only max-256 catalog subset. Aim for **2,000–5,000 product groups**, **3–6 distinct useful views per product** (roughly 6,000–30,000 images). These are targets subject to metadata availability and deduplication, not claims about retained counts. Use all relevant listing shards rather than the current first-valid-ID shard sample. Cap products per class and views per identity; do not let long spin sequences dominate training. Classification can retain one-view products; retrieval evaluation requires independent gallery/query captures.

Pilot room-label target: roughly **200–500 training photos**, **50–100 validation photos**, **100–200 test photos** from distinct rooms/households, with a goal of at least **50 test instances per category**. These are modest starting targets, not a statistical power guarantee. Report actual category coverage and confidence intervals; acquire more test rooms if intervals are wide. Keep a fixed easy/hard breakdown for occlusion, clutter, size and lighting. Begin segmentation with a smaller carefully reviewed mask subset if annotation budget is limited.

## 5. Reproducible, leakage-resistant splits

### Product grouping before splitting

1. Build a source manifest joining listings, image IDs, spin IDs and mesh IDs. Merge repeated item IDs across domains/languages. Use `item_id` as the canonical starting identity, preserving domain records separately.
2. Construct connected product groups using shared item/image/spin/mesh IDs, exact image hashes, and reviewed product-family equivalence (brand/model/variant metadata when available). Group verified variants/aliases conservatively. Metadata similarity alone is not proof of identity; audit ambiguous links. Quarantine unresolved shared assets rather than assigning them across splits.
3. Audit cross-product near-duplicates with perceptual hashes and normalized/cropped comparisons; manually review candidate links. Current dHash/RGB heuristics are useful candidate generators, not proof that captures are independent. Shared capture clusters must remain together even within retrieval gallery/query role assignment.
4. Keep every view, crop, augmentation, render and synthetic derivative of a product group in its assigned learning split. Hash identities before augmentation. A new render of a held-out product is still held out.

### Deterministic assignment

Freeze a **70% train / 15% validation / 15% test** group allocation, seed **20261103**. Within each audited category stratum, sort group IDs by SHA-256 of `abo-furniture-v1|20261103|group_id`; assign the first floor(0.70n) to train, next floor(0.15n) to validation, remainder to test. Stratify conflicting/multi-category groups using a recorded deterministic primary category, but never split the component. Publish actual counts; rare classes may need additional data rather than unsafe per-image balancing.

Freeze the eligible dataset snapshot before this assignment; adding products later produces a new manifest version. Save explicit `group_id → split` assignments so membership never silently changes. Record component construction rules, source hashes, exclusions, mapping version, seed, algorithm and manifest SHA-256. Require **zero overlap** of product groups, source image IDs, exact hashes and derivative lineage across learning splits; resolve flagged near-duplicates before locking.

Previously examined ABO identities/protocols, if recovered from another workspace, should be development/regression data or excluded from the fresh test set. Absence of these catalogs here is not evidence that their results were never seen. This protocol prevents new training/evaluation leakage; unknown overlap with upstream web-scale pretraining cannot be ruled out and must be disclosed.

### Retrieval roles within held-out identities

Identity-disjoint learning does **not** mean a test query's true product is missing from its test gallery. For validation and test separately, reserve one or two distinct capture clusters per present product as gallery views, and a different cluster as query. Test gallery identities may be unseen during training: indexing their frozen-model vectors is permitted, but no gradients, projection fitting, or calibration may use those identities.

Use separate validation and test galleries. Fix the same gallery images, distractors, product counts and query lists for every model. Within each held-out split, deterministically reserve approximately **20% of eligible product groups** as absent-product queries and remove all their gallery views. Require present and absent coverage per class; report actual counts. If only one capture exists, use gallery-only or classification-only roles, not an invalid same-photo query pair. Additional distractors must have audited identities and be fixed before scoring.

### Room and synthetic splits

Split by household/room/capture session before selecting frames; also connect rooms that show the same known physical product identity. For a strict unseen-product test, link verified room objects to ABO groups and keep them consistent across learning splits. If unknown room identities prevent certifying product disjointness, mark that benchmark household-disjoint only and disclose the limitation.

Generate synthetic training scenes from training products and training backgrounds only; use separate held-out backgrounds, room templates and products for validation/test. Never combine products from different learning splits in one scene. All views of the same synthetic scene remain together. Group repeated room backgrounds and edited variants before splitting.

## 6. Training strategy for approximately 8 GB VRAM

All ranges below are engineering estimates requiring a later approved forward/backward memory probe. No fitting/probe was run here. Keep about **1–1.5 GiB VRAM headroom** for display/runtime variability; target peak allocation below roughly **6.5–7 GiB**. Load one model at a time and perform FAISS search on CPU. Do not run detection, embedding and segmentation training concurrently.

### Stage A: frozen baselines and inexpensive learning

1. Lock data and evaluation configuration. Run original OWLv2 on the new reviewed detection set into a new reference directory. Run frozen SigLIP 2 retrieval separately if compatible local weights exist or acquisition is later approved. Never call OWLv2 the original SigLIP retrieval model.
2. Cache frozen SigLIP 2 image vectors at native 224 resolution, batch 8 initially (reduce to 4/1 on memory pressure). Fit a linear classifier on train features with class-weighted cross-entropy, and optionally a small 256-dimensional retrieval projection with supervised contrastive or triplet loss. Seed all sampling; train-only feature statistics. Start with AdamW learning rate 1e-3, weight decay 1e-4, 10–30 epochs, validation patience 5. These are starting settings, not tuned results.
3. For projection training, sample multiple views per identity and several identities per class; avoid false negatives among known aliases. Normalize projected embeddings. Prefer supervised multi-positive loss over naive diagonal-only contrastive loss when a batch repeats an identity. Select classification checkpoints by macro-F1 and retrieval checkpoints by product-macro R@5, with deterministic tie-break R@1 then earlier epoch. Keep the objectives and artifacts separate.

This stage provides an inexpensive, interpretable improvement test. Frozen features + small head may need well below 1 GiB for fitting; feature extraction might occupy approximately 1–3 GiB depending on loaded model components and batch. OWLv2 detection remains a separate fixed baseline and cannot be claimed to improve from embedding-only training.

### Stage B: limited embedding adaptation

Only if Stage A supports a real validation gain, unfreeze the final image-encoder block/projection (freeze the text tower) or implement small rank-4/8 adapters on explicitly verified layers. Use 224 inputs, mixed precision, microbatch **4–8 images**, gradient accumulation **4–8** steps (effective image batch 16–64), and identity-balanced pairs. Start backbone LR **1e-5**, new head LR **1e-4**, weight decay **0.01**, 5–10% warmup, gradient clipping **1.0**, 3–10 epochs, early-stop patience **3**. Verify gradients only reach intended parameters.

Estimated peak **3–6 GiB** for limited adaptation; actual encoder loading and activation storage can exceed it. Accumulation increases optimizer batch size but does not create additional simultaneous contrastive negatives; use within-microbatch pairs or a documented train-only feature queue. Unfreeze more only after profiling. Augment with moderate crops/color/lighting/occlusion that preserve identity; do not train a color-sensitive SKU task with transformations that change the distinguishing product attribute.

### Stage C: supervised detection adaptation

Prerequisite: validated box labels, sufficient real room coverage, and a tested loss implementation. Begin with frozen OWLv2 vision/text towers and train detection heads; alternatively evaluate a compact purpose-built detector as a separately identified candidate. A closed-set detector must still be compared on the same seven-category room test and must not be presented as preserving open-vocabulary capability.

OWLv2 adaptation needs explicit bipartite matching, class loss, L1/GIoU box losses, background/unmatched handling, padded-query masking and an objectness-target policy. Validate resizing/padding/box normalization and empty-target images. Check the installed objectness path for detached features when deciding what can learn. Test known-box gradients and small-batch overfit only after training authorization. Do not repurpose `return_loss=True` on the image/text backbone as detection training. The [OWLv2 documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/owlv2) explains the detection architecture and matching-based training background.

Start microbatch **1**, accumulation **8–16**, mixed precision (BF16 if verified supported, otherwise FP16 with scaling), head LR **1e-4**, weight decay **0.01**, 5% warmup, gradient clip **1.0**, up to **10 epochs**, patience **3** on validation AP. Preserve native 960 preprocessing for the first comparable head-only attempt. Freeze backbone computation under no-grad when only heads train. Estimated **3–6 GiB** head-only; partial backbone adaptation at 960 may exceed 8 GiB. If necessary, propose 640/768-resolution positional interpolation only after checking model/processor behavior; evaluate both original and candidate under the same reduced-resolution condition as an additional comparison.

If head-only adaptation plateaus, consider last-block/adapters, checkpointing and LR **1e-5** for trainable backbone parameters. Budget **5–8+ GiB**, with OOM possible. Do not promise full end-to-end 960-pixel OWLv2 tuning on this laptop. Synthetic training should mix in reviewed real training rooms, with mixing selected on validation only, to address the catalog-to-room domain gap. Evaluate old-category retention and hard negatives to detect catastrophic specialization.

### Stage D: segmentation only when justified

Do not jointly fine-tune the optional SAM bridge and OWLv2 on 8 GB as the initial experiment. First evaluate a fixed, locally available mask model with box prompts or a compact instance-segmentation candidate, sequentially with detection. Adapter/small-head training at 512–640 and batch 1 may be feasible (rough estimate **4–8+ GiB**); exact architecture and implementation need approval and profiling. Masks from ABO renders are a useful synthetic warm start, not sufficient proof of real-room segmentation quality.

## 7. Python 3.14, storage and memory risks

### Compatibility

- Existing package versions and historical CUDA inference are evidence that this workspace has a working history, not proof that every training dependency works under Python 3.14. Check native imports, paired torch/torchvision operators, CUDA/driver access, dataloader workers and backward kernels later before committing to a trainer.
- `pyproject.toml` pins **Pillow >=10,<13**, but installed Pillow is **13.0.0**. Record this dependency mismatch; do not silently reinstall the project or downgrade packages. The declared Python minimum >=3.9 also does not represent the full source requirements: some code uses newer syntax/features.
- Optional `requirements-odometry.txt` pins NumPy 1.26.4/Open3D 0.18.0 for a separate worker environment. Do not apply it to this Python 3.14 model environment. Old pinned native packages may lack matching wheels or fail to build.
- PEFT/Accelerate, COCO tools, renderer bindings and specialized CUDA packages need exact Python/ABI/wheel compatibility checks. They are not implicitly usable because PyTorch is installed. Prefer the existing PyTorch stack, standard operations and CPU FAISS. Avoid adding FlashAttention/bitsandbytes/Detectron dependencies unless required and approved.
- Python 3.14 changes multiprocessing defaults on supported POSIX platforms. Use explicit contexts and main guards for training workers, seed each worker, and begin with `num_workers=0`, then test 1–2 workers. Existing FAISS code already explicitly uses spawn. [Python 3.14 changes](https://docs.python.org/3.14/whatsnew/3.14.html).
- If a dependency is incompatible, report the concrete failing combination. A separately approved Python 3.11/3.12 training environment may be a fallback; never replace or mutate `.venv` to achieve it automatically. Freeze exact working versions, platform and source commit once approved. Do not upgrade Transformers merely to follow newer examples.

### Disk budgets

Official download sizes below are approximate published archive sizes, interpreted as decimal GB; they are not measured extraction sizes or a proposed download list. [Official archive listing](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).

| Asset | Published archive size | Planning implication |
| --- | --- | --- |
| Listings | ~83 MB | Start with metadata after approval; allow 0.5–2 GB for extracted metadata, manifests and audit indexes |
| Small catalog images | ~3 GB | Full small-image archive can be practical later; furniture-only selective acquisition is preferable initially |
| Original catalog images | ~110 GB | Avoid full acquisition initially; target selected high-resolution furniture views only |
| Spins | ~40 GB | Acquire selected products/views only if retrieval angle coverage needs them |
| 3D models | ~154 GB | Not needed for first classifier/retrieval experiment |
| Material render benchmark | ~271 GB | Defer; selected assets or local bounded rendering need a separate plan |
| Part labels / MVR split | ~2 GB / ~4 MB | Optional task-specific releases; not core room supervision |

Those six major archives total roughly **578 GB** before extraction, caches and checkpoints. The separate reconstruction renders are about **223 GB** according to the authors' repository; do not add that release by default. Keeping archive and extracted copies can approach twice an archive's disk use, with additional expansion for metadata/decoded assets. Do not assume the apparent 943 GiB free can comfortably hold every release plus working space.

Planning budgets for the proposed pilot: **10–20 GB** for selected small images, metadata, audit artifacts and feature caches; **20–50 GB** for selective original-resolution data instead; **5–15 GB** for bounded synthetic scenes/masks if later approved; **5–15 GB per active experiment** for checkpoints/logs/optimizer state. Reserve approximately **30–60 GB** for the first learning stage including working headroom. Counts and per-image sizes must be sampled before approving a fixed acquisition budget. No automatic cleanup of originals or historical runs.

### RAM and optimizer accounting

Use lazy image decoding, bounded prefetch, disk-backed feature matrices and batch queries. A float32 RGB 960×960 tensor is about **10.5 MiB** before activations; 30,000 such decoded tensors would be about **309 GiB**, so never cache the full tensor dataset in RAM. Target training-process host RSS below roughly **8–10 GiB** in this 14 GiB session, with worker/FAISS overhead measured separately. Swap is not a training-memory solution.

Feature/index storage is `N × d × bytes_per_element`: at 768 dimensions float32, 30,000 images need about **88 MiB** per matrix, all 398,212 catalog images about **1.14 GiB**. Embedding dimension must be read from the actual model. CPU IndexFlatIP stores another matrix; parent/worker copies and IPC can multiply memory. Existing multi-image search requests all image rows to aggregate products; this becomes costly at full scale. Bound query batches and propose a validated product-level top-k strategy before scaling beyond the pilot.

For 155M OWLv2 parameters, FP32 weights alone are about **0.58 GiB**. FP32 weights + gradients + two Adam moments are about **2.31 GiB**, before activations, temporary tensors and CUDA overhead; mixed-precision/master-weight arrangements can add more. Thus checkpoint size is not training VRAM. Frozen-head training avoids most gradient/optimizer storage for the backbone. A full resume checkpoint can occupy roughly **1.9–2.5 GB** or more; retain a bounded best/latest set in each new run. Adapter-only saves are smaller but must include a base-checkpoint digest and exact reconstruction instructions.

## 8. Automatic original-versus-trained evaluation design

Implement only after approval. Proposed evaluator consumes frozen manifests and writes into fresh directories; it never updates old runs. Compare on identical images, labels, prompt/category mapping, gallery membership, inference precision, hardware and scoring implementation. Record any architecture-specific preprocessing differences. Run candidates sequentially.

### Model comparison matrix

- **D0**: immutable original OWLv2 loaded by absolute local path, downloads disabled; historical confidence 0.15/NMS 0.5 as a reproducibility condition.
- **D1**: future adapted OWLv2, same inference protocol. **D2**: optional compact detector, separately named.
- **E0**: original frozen SigLIP 2 retrieval encoder, with pinned local weight digest. **E1**: projected/adapted encoder. OWLv2 has no retrieval result in the saved baseline, so do not invent one.
- Evaluate D0+E0, D1+E0, D0+E1 and D1+E1 to separate detector and retrieval contributions. Add ground-truth-crop E0/E1 retrieval to isolate embedding performance. Classification heads get their own classifier baseline, not a claimed direct replacement for detection.
- If an OWLv2 embedding retrieval baseline is desired later, define a fixed pooling/normalization protocol and separate feature/index family; do not compare vectors from different encoder spaces.

Use two detection operating-point reports: a fixed shared postprocessing configuration for attribution, and each model's validation-selected configuration for deployment comparison. Freeze confidence, prompts, NMS and optional containment before test; never optimize on test images. For AP, export sufficiently low-floor scored candidates, apply the documented suppression policy, and use the same per-image cap. Historical saved candidates below the original confidence floor are unavailable; a fresh low-floor baseline inference run is needed for a valid PR/AP sweep.

### Detection metrics and matching

At IoU **0.50** and **0.75**, sort predictions by confidence with deterministic tie-breaking and perform one-to-one same-category matching. Report TP/FP/FN, precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, F1, per-category and macro/micro summaries, confusion diagnostics, and matched-box mean/median IoU. IoU is intersection/union of boxes; the current `detection_accuracy_jaccard = TP/(TP+FP+FN)` is a count metric, not mean spatial IoU. Report matched IoU beside recall, since omitting hard objects can inflate matched IoU.

Add AP50, AP75 and COCO-style AP averaged over IoU 0.50:0.05:0.95, category AP, AR and PR curves. Use a validated COCO evaluator with specified max detections (primary maxDets 100), crowd/ignore semantics and class mapping; cross-check simple fixtures against current matching. COCO AP is interpolated PR evaluation, not precision averaged over thresholds. [Official COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

Keep missing/failed images in evaluation, with annotated objects counted as missed, and publish failure count/rate. Preserve empty-GT negative rooms: their detections count as FP unless ignored by a frozen rule. Distinguish unknown/unannotated objects from verified background. Report zero-support metrics as null with counts rather than silently converting to perfect or zero accuracy. Tag synthetic and real metrics separately.

### Duplicate detections

For each image and IoU threshold, after one-to-one matching, count an unmatched same-class prediction as a **duplicate FP** when it overlaps an already matched reference object at that threshold. All other unmatched predictions remain other FP; duplicates are never removed from the precision denominator. Report duplicate FP count, duplicate FP / all predictions, duplicate FP / annotated objects, and fraction of matched objects with extra predictions; show denominators/null rules.

Also report class-agnostic extra predictions overlapping the same matched instance as a separate cross-label redundancy diagnostic. Pairwise same-class IoU/containment can flag nested boxes when ground truth is absent, but it is a heuristic and must not be called true duplicate rate. Compute diagnostics before and after NMS/containment using saved raw candidates and filter audits; assess recall loss on adjacent chairs/overlapping furniture.

### Product retrieval and classification

- Rank **distinct products**, not photos. Report R@1, R@5, R@10, full-ranking MRR and mAP; with exactly one relevant product per query, AP equals reciprocal rank. For graded/category relevance, use a separately defined relevance set and nDCG; do not substitute category similarity for exact-product success.
- Report query-micro and product-macro averages, category metrics and gallery size. For multiple queries of one identity, average within product before product-macro scoring. Fix view aggregation and gallery image count; compare max/centroid only on validation. Disable retailer/metadata preference bonuses for the main visual retrieval comparison.
- Report oracle-ground-truth-crop retrieval, retrieval given a correctly localized detection, and **end-to-end** retrieval where detection failures/misses count as failures. Exact retrieval denominators include only verified in-catalog identities, with unknown/unverified and absent counts shown separately.
- For absent-product queries, report false acceptance, rejection recall, present-product recall at a validation-calibrated false acceptance target (proposed 5%), and AUROC/AUPRC for the stated present/absent positive convention. Calibrate cosine/margin thresholds on validation only. Similarity is not a calibrated identity probability; do not assert exact identity solely from nearest-neighbor ranking.
- Classification: top-1 accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrix, calibration/error and unknown-class rejection if trained. Crop classifier results and full-room detector results remain separate tasks.

### Segmentation evaluation

Evaluate only where independently reviewed masks or held-out synthetic instance masks exist. Report mask AP50/AP75/AP50:95, matched visible-mask IoU and Dice, per-class results, missing masks and segmentation failures. Publish oracle-box versus predicted-box mask results to expose detection dependence. Baseline OWLv2 alone has **no mask output**: mark segmentation N/A, or compare D0 plus a fixed mask generator against D1 plus the same generator as a separately identified pipeline baseline. Never score bounding-box rectangles as baseline segmentation masks.

### Uncertainty, reproducibility and selection

Store checkpoint/processor hashes, code revision (or source digest when git metadata is unavailable), source/data/split/gallery hashes, prompt/filter settings, precision, seed, device/driver/runtime, package inventory, epoch, optimizer/scheduler and deterministic flags. Fix Python/NumPy/PyTorch/CUDA and dataloader seeds; document nondeterministic kernels. Train selected candidates with seeds **20261103, 20261104, 20261105** if budget permits and report mean/spread without selecting the luckiest test run.

Select checkpoints and all hyperparameters on validation. Use a predeclared primary metric per task (classification macro-F1; retrieval product-macro R@5; detection real-room AP50:95; segmentation mask AP50:95). Freeze a release candidate before a single final test comparison. Bootstrap paired differences **5,000 times**, seed **20261106**, resampling independent product groups for ABO and household/room groups for rooms, never individual adjacent views. Report 95% intervals and class coverage.

Proposed adoption gate: primary-metric paired improvement interval lower bound >0, no more than **2 percentage points** drop in a prespecified safety metric (retrieval R@1, detection recall at the chosen operating point), no unexplained class collapse, and peak VRAM within the available budget. Define category-specific minimum support and tolerance on validation before test. A small test may yield an inconclusive result: keep the original available and collect a new independent test rather than repeatedly tuning against the same test. Show p50/p95 synchronized warm inference latency, load time separately, peak allocated/reserved VRAM and host RSS; account for laptop power/thermal variation.

## 9. Artifact layout and preservation contract

Future proposed paths (not created by this planning step):

```text
models/owlv2-original-baseline-20261103/        # immutable reference
data/abo/source-manifests/v1/                  # source records, provenance, hashes
data/abo/splits/furniture-v1/                  # assignments, role manifests, taxonomy
data/abo/derived/furniture-v1/                 # derived assets with parent lineage
models/furniture-candidates/<task>/<run-id>/   # weights/adapters + processor + config
reports/abo-experiments/<run-id>/              # logs, predictions, evaluation, plots
reports/abo-comparisons/<comparison-id>/       # summaries linking immutable runs
```

Use new run IDs and fail if an output exists. Resolve paths and reject any output equal to, nested in, or an ancestor write targeting the protected original checkpoint or existing benchmark runs. Keep loading and saving paths separate. Hash all baseline files and existing predictions before/after a run; do not save model/processor into the load path, rename baseline files, or update baseline manifests. Save adapted configs/processors alongside candidate weights, plus parent checkpoint digest and trainable-parameter specification. Rebuild separate candidate catalog indexes; metadata compatibility must include actual checkpoint/projection digests, not merely a shared model name or `main` revision.

Comparison outputs should include machine-readable `metrics.json`, per-image/per-query diagnostics, configuration/data digests and a human-readable report/CSV. New baseline-reference predictions on new test data belong in their own new run, never in `gpu-owlv2-rtx4070-test01`. Preserve the existing filter-comparison reports as historical experiments.

## 10. Approval milestones and proposed deliverables

| Milestone | Work proposed for later approval | Completion criterion |
| --- | --- | --- |
| 0 — this document | Read-only inspection and written plan | Plan delivered; original weights and historical artifacts preserved |
| 1 — data scope | Approved metadata/selective-image acquisition with explicit byte cap, taxonomy and label audit | Furniture counts/rights verified; zero-overlap split and retrieval-role manifests locked |
| 2 — implementation | Small isolated data/split/evaluation extensions, training-head entry point, any specifically necessary installation | Reviewed changes; meaningful tests for leakage rejection, geometry, duplicates, ranking and checkpoint protection pass; no training implied |
| 3 — environment readiness | Approved native import/GPU/backward feasibility probes and memory profiling | Existing `.venv` retained; concrete compatibility and memory report; explicit proposed remediation if blocked |
| 4 — first learning | Separately approved frozen-feature classifier/projection run | Versioned checkpoints, validation selection, reproducible logs and bounded resource use |
| 5 — adaptation | Limited embedding adaptation; detector work only after reviewed boxes and tested loss | Validation gain, domain-gap audit and no baseline/output mutation |
| 6 — final comparison | Frozen candidate evaluation against independently labelled locked test sets | Paired metrics/intervals, failure/duplicate analysis, separate retrieval/detection contributions |
| 7 — masks/scenes | Optional rendering and mask-label acquisition, later segmentation training | Independently reviewed mask test; explicit synthetic/real split and separate pipeline baseline |

The next approval request should specify the concrete file changes, package versions if any, exact data sources, product/image limits, download/storage cap, and whether it authorizes only preparation or an actual training run. No installation, dataset/model acquisition, environment migration, or substantial code change is performed by this plan.
