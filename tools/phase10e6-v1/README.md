# Phase 10E.6 CPU-only saved-query audit

Results: `reports/phase10e6-v1/report.md`.
Private, versioned evidence: `data/open-images/preparation/phase10e6-v1/`.
No downloader, GPU runner, model constructor or training entry point is added.

The one-time preparation sequence was checkpoint verification, source-bound
indexing (versions 1–3), CPU mining, contact-sheet generation, actual viewing of
all three sheets, candidate finalization, provenance/proposal generation and
CPU aggregation. Index versions 2/3 have identical numerical snapshot sets;
the final96files are exact-output aliases. Immutable outputs are retained.
Historical code and artifacts are read only; all new candidates remain inactive.

Run regression tests:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B -m unittest discover -s tools/phase10e6-v1 -p 'test_*.py' -v
```

Validate current dataset or verify the final integrity seal:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e6-v1/validateE6.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e6-v1/verifyE6.py check
```

Do not rerun generators into an existing destination. `commonE6.atomic` uses
exclusive temporary files, an atomic no-overwrite link, retained recovery files
and a conservative512MiBoutput bound. Use a newly versioned phase for future
mining/evidence changes; do not modify this completed audit or source labels.

Independent review intake is entirely separate from optimization evidence.
Copy the empty `independent-review-template-v2.json` to a new submission, add
only actual authorized photos with original width/height, rights evidence and
blind human decisions, and run `acceptanceE6.py submission.json`. All seven
categories need explicit completeness, including zero instances. Labels use
normalized original-image `xyxy` coordinates and a boolean `ignore` field.
The validator never corrects, invents or writes reviews. Synthetic fixtures in
unit tests are never emitted as actual reviews or counted as accepted images.
An empty intake validates as zero eligible photographs.

The59forwardE5cache request and64IDrights route are disabled proposals requiring
separate authorization. Acceptance collection and any scoring also require their
own approval. No training is justified by the current zero-new-negative result.
