# Phase 10E.13 CPU reliability verification

Run from the repository root with the existing virtual environment. These tools
do not run models, download data, or mutate historical datasets. Test images,
vectors and predictions are explicitly synthetic; saved historical predictions
are read only for output comparisons and bounded CPU filtering benchmarks.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e13-v1/verify13.py check
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e13-v1/verify13.py check --full
```

The full check rehashes all 29,229 records in the local after-snapshot. Detailed
inventories are intentionally not committed. A clone without those local
inventories cannot claim a full local-data preservation check. The public
preservation report records their SHA-256 digests. No existing phase's seal is
regenerated: replay Phase 10E.12 source checks at its recorded commit.

Tests use fresh report labels and refuse existing output paths:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B tools/phase10e13-v1/run_checks13.py targeted replay1
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B tools/phase10e13-v1/run_checks13.py broad replay2
```

`eleven` restricts execution to the 11 formerly failing test IDs. The runner
blocks ordinary source network connections and excludes the two real
TinyTransformers forward tests. Optional native/dependency integrations remain
explicit skips. It does not install or emulate those dependencies.

`measure13.py` and `dense_benchmark13.py` require the existing local prediction
artifacts described in the report and use exclusive output creation. They are
recorded audit procedures, not a request to regenerate or overwrite evidence.
`before`, `after`, `seal` and `closeout13.py` likewise create a phase's initial
evidence exactly once. `check` is the repeatable read-only operation.

## Active reliability APIs

- `furniture_bench.persistence.atomic_write` takes a binary writer and optional
  read-back validator. `atomic_json` validates strict JSON. Use `overwrite=False`
  for a new immutable artifact; shared `data.write_json` keeps its existing
  replace-on-write contract. Concurrent mutable writers need an additional policy.
- Temporary files are unique siblings. Pre-publication failures preserve the old
  target. An `AtomicCommitError` has `committed=True`: publication succeeded but
  directory durability was not confirmed. Do not blindly retry as though no file
  exists. Unrelated crash orphans are retained and never treated as valid results.
- `data.load_image_with_digest(path, expected_sha256)` hashes and EXIF/RGB-decodes
  the same compressed-byte snapshot. This trades compressed-file memory for
  provenance consistency, without changing successful image pixels.
- `cache.file_identity` hashes explicit logical roles for local encoder weights,
  head/config files and processor/tokenizer files. It does not load a model or
  certify that a running model uses those files.
- `cache.make_contract` requires image SHA, model/processor identities, explicit
  preprocessing and inference settings, and supervision context. Detection uses
  ordered categories/prompts and a prompt-to-category map. Image-only embedding
  uses task `image_embedding` with empty categories and prompts.
- `inference_key` changes with pixels/model/preprocessing/query settings;
  `context_key` changes with annotations/ignored regions/taxonomy/completeness/
  source group. File-backed masks must include content hashes, not just paths.
  If labels determine a crop, that crop also belongs in preprocessing.
- `write_prediction_cache` creates one no-clobber v2 JSON envelope with a payload
  digest. `read_prediction_cache` rejects changed inference or corrupted payloads;
  label-only changes explicitly require evaluation recomputation. It does not run
  the evaluator. `evaluation_fingerprint` binds context and declared metric rules.
- `catalog_expansion.incremental_embeddings(..., cache_contracts=...)` opts into
  atomic `features-v2/*.npz` entries, each holding vector, vector digest and
  inference provenance together. Legacy callers retain their old keys and NPY
  format. No historical caches are migrated, deleted or repaired automatically.

Declaring a contract is not proof of rights clearance, annotation completeness,
human review, or independence. Existing source/group rules still apply. Legacy
PhotoJobs and room-layout cache adapters have not been silently converted.
