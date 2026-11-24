# Phase 10E.14 synthetic reliability verification

Use the existing project virtual environment, from the repository root. No
installed package, real photograph, real annotation or model is required by the
new synthetic fixture. All generated binary files live in temporary directories.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B tools/phase10e14-v1/run_checks14.py targeted replay1
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B tools/phase10e14-v1/run_checks14.py broad replay2
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e14-v1/verify14.py check
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e14-v1/verify14.py check --full
```

Choose new report labels. Existing test evidence is never overwritten. Broad
execution excludes the two real TinyTransformers forward tests. Optional native
PDF and absent optional geometry packages remain documented skips.
Authoritative final evidence is `CPU-targeted-final2.*`, `CPU-broad-final2.*`,
`summary-final.json`, and `preservation-final.json`. Earlier records are retained
as intermediate development evidence, including the captured regression failures.

The full preservation check needs the local ignored inventories; a clean clone
without those inventories cannot claim to have checked private datasets. Public
reports include inventory digests. Prior source seals must be replayed at their
recorded commits; no earlier seal is rebound to current source.

`baseline14.py`, `replay14.py`, snapshot actions and `closeout14.py` create the
initial versioned evidence once. They refuse existing report paths. Baseline
capture preceded the active changes. Do not overwrite it to compare a different
source revision. `check` is the repeatable read-only operation.

## Opt-in active adapter

`furniture_bench.pipeline.run_cached_detection` is the new library entry point.
It is detection-only: legacy CLI behavior, retrieval/segmentation and historical
phase runners are unchanged. The fixture's `SyntheticDetector` implements the
same `detect`/`last_candidates`/`metadata` seams as the active detector without
loading a model. Existing OWLv2 exposes pre-NMS candidates at a score floor;
these are not necessarily all raw model queries. The adapter does not fabricate
query logits or IDs that an underlying detector does not expose.

Arguments:

- A schema-1 room manifest accepted by `load_manifest`, with explicit per-image
  `source_group_id`, `partition`, seven-category `completeness`, and optional
  `protected_regions` / `identity_family_group_ids`.
- A fresh v2 output root; legacy/unmarked directories are rejected unchanged.
- Explicit `spec` model identities, categories, ordered prompts and
  `query_categories`, taxonomy version, preprocessing and inference settings.
  The adapter's preprocessing is exactly EXIF transpose + RGB. Its three fields
  are `orientation=EXIF-transpose`, `color=RGB`, `box_frame=oriented-xyxy-pixels`.
  Unsupported transforms are rejected. Detector-owned resizing is identified by
  the actual pinned processor files, not a second guessed transformation.
- Actual local `model_files` and `processor_files` role-to-path maps. Their
  `file_identity(...)["sha256"]` digests must match the spec. Include all relevant
  weight/head/config/tokenizer/processor files. File identities are checked before
  processing and before completion. Metadata/query order must agree with the
  detector. These checks do not attest a malicious or misconfigured live model's
  in-memory parameters; loading and no-update checks remain the GPU preflight's
  responsibility.
- `settings.candidate_floor` declares the detector's raw candidate acquisition
  floor. Postprocessing may never request a lower floor than was captured.
  Keep NMS and final confidence settings in `policy`, outside raw inference keys.

Derived files are immutable, versioned under `runs/<request-key>/`. Raw candidates
use E13 v2 cache files under `raw/`; labels/groups can change without rerunning
inference. Every run has `input.json`, `predictions.json`, `postprocessed.json`,
`supervision.json`, and `evaluation.json`, followed by `completed.json` containing
their sizes and hashes. Read v2 results via `load_completed_run`, not by treating
an arbitrary intermediate file's existence as successful completion.

Cooperative POSIX/WSL locks serialize each raw identity and each run, with a
30-second default contention timeout. Locks release on process death; their
stable files are never unlinked. Native Windows locking is explicitly unsupported
rather than silently unsafe. Immutable partition bindings persist across versions
within this root. Unknown relationships across other roots still require the
authoritative historical source/family/holdout registry at dataset preparation.

Any partial annotation coverage causes unmatched predictions for that image to
remain ignored, conservatively, because the current evaluator has image-space
ignore regions rather than category-specific completeness masks. Known positives
still match first. The derived input explains this policy and keeps the original
context in `supervision.json`. That file is a geometry/provenance descriptor, not
a training-ready query-negative tensor or a new human review. Existing scoring
thresholds, matching functions, source annotations and losses are unchanged.

Conflicting or corrupt committed files cause explicit failure and are preserved.
Crash temporaries are ignored, not deleted or accepted. Resume reuses verified
raw work and republishes only missing compatible artifacts. No reader can accept
an incomplete v2 run through the completed-manifest reader. This is a logical
commit protocol, not a claim of a filesystem-wide multi-file transaction.
