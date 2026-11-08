# Furniture AI

Local furniture detection, image-embedding, catalogue-classification and dataset
preparation experiments. This repository contains the Windows/Ubuntu WSL
machine-learning workspace code.

## Recognition models

- **OWLv2 Base (ViT-B/16)** detects seven furniture categories: sofa, chair,
  table, bed, desk, cabinet and lamp. The default inference path uses the
  original pretrained detector.
- **SigLIP 2 Base 224** provides frozen image embeddings for product similarity
  and experimental linear, cosine and MLP classification heads.
- An isolated **Phase 5C head-only OWLv2 adaptation** and a **Phase 3 expanded
  cosine classifier** are experimental checkpoints. They are not automatically
  selected by the default inference path.
- Optional modules support semantic segmentation, relative depth, room geometry
  and photo-replacement experiments. Semantic masks are distinct from furniture
  instance identities; relative depth does not establish physical dimensions.

Recorded Phase 5C development results use 18 Open Images photographs and 41
source-native annotated boxes, with confidence 0.15 and operating IoU 0.5:

| Detector | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|
| Original OWLv2 | 72.16% | 51.91% | 21.71% | 68.29% |
| Selected head-only adaptation | 79.56% | 53.60% | 30.21% | 70.73% |

These are development-set annotation-agreement measurements, not independently
verified real-world accuracy or official Open Images benchmark results. The
aggregate improvement is heavily influenced by Cabinetry; Table false positives
and Bed/Table localization regressions remain important limitations.

## Repository layout

| Path | Purpose |
|---|---|
| `src/furniture_bench/` | Detection, filtering, cropping, embeddings, retrieval and evaluation |
| `src/room_layout/`, `src/room_reconstruction/`, `src/room_multiview/` | Experimental room structure and geometry |
| `src/room_viewer/`, `src/photo_simulator/` | Local viewer and photo workflow modules |
| `tools/` | Versioned acquisition, auditing, training and comparison tools |
| `scripts/` | Supporting workflow utilities |
| `tests/`, `tests_photo_simulator/`, `tests_inpainting_eval/` | Automated checks |
| `reports/abo-training-plan.md` | Original ABO training and evaluation plan |
| `PHOTO_SIMULATOR.md` | Local photo-simulator documentation |

## Local environment

The existing training workspace uses `.venv/bin/python`, Python 3.14.4,
PyTorch 2.14.1+cu132 and Transformers 4.57.6. Recorded GPU experiments ran on an
RTX 4070 Laptop GPU with approximately 8 GB VRAM. CPU execution is supported by
the generic detector/embedder; several isolated experiment runners explicitly
require CUDA.

Project metadata and optional dependencies are declared in `pyproject.toml` and
the `requirements*.txt` files. The recorded environment has Pillow 13.0.0 despite
the declared `<13` bound; reproducing it requires an explicit compatibility
decision rather than assuming every dependency bound matches that environment.

In an already configured environment, inspect the benchmark CLI without loading
models or downloading weights:

```bash
PYTHONPATH=src .venv/bin/python -B -m furniture_bench --help
```

Acquisition and training tools are separate workflows. Their historical scripts
may require exact local manifests, checkpoint fingerprints, source licensing
evidence and an approved resource budget. Do not treat running an entire tool
directory as a generic setup command.

## Data, checkpoints and reproducibility

Datasets, source photographs, model weights, feature caches, generated reports,
runtime environments and credentials are deliberately excluded from Git. A clone
contains the implementation, not a ready-to-run copy of the local experiment
artifacts. Many tests and experiment runners need those separately provisioned
artifacts and will not run from a code-only clone.

Keep original pretrained weights, historical outputs, training/development groups
and frozen evaluation partitions separate. Preserve native source annotation IDs,
coordinates and completeness flags. Missing or unresolved labels must not silently
become background negatives. Weak-label classifier scores and cosine similarities
are not calibrated correctness probabilities.

Photograph rights, dataset annotation licences and pretrained-model licences are
separate. The repository's MIT licence applies to the code; it does not relicense
third-party photographs, datasets or model weights. Retain their own source,
licence and attribution records when obtaining them separately.
