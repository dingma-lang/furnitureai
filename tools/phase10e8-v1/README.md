# Phase 10E.8 — CPU-only Chair-confusion qualification

Read `reports/phase10e8-v1/report.md` for the completed findings. Private evidence
and rendered diagnostic plates are under
`data/open-images/preparation/phase10e8-v1/`. Historical source data is read only.

Open the running read-only dashboard at **http://localhost:8778**. To start it
again when that port is free:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e8-v1/serve8.py --port 8778
```

Use `--port 0` to select another available localhost port; the command prints
its URL. Do not stop unrelated existing review servers to claim their ports.

Filter by decision or search a sample/photo/object. Click a diagnostic plate to
inspect the original photograph. The full-resolution view offers native boxes,
best/original or all overlapping native mask overlays, original/E5 query boxes
and zoom. Score tables show all seven categories. Expand provenance for original
source IDs, hashes, licence/creator records, exact coordinates, completeness flags
and decision limitations. CSV export is read only and contains no human approvals.
No drawing, approval, write or training endpoints exist.

The closed candidate list is exactly E7's21B candidates. `prepare8.py` binds
their E5/original arrays and source/mask frames, computes continuous intersections
and generates enlarged evidence plates. Each of the21plates was actually viewed
before the observations in `qualify8.py` were authored. These are assistant
assessments combined with source checks,not an installed vision model or human
labels. No inference or model construction occurs in this phase. Missing labels,
bbox context and partial mask overlap cannot grant a negative.

`qualify8.py` retains the existing32pixel/two-pixel-interior/zero-protected-overlap
policy. It groups five observed partial handbag queries using the exact same
native COCOinstance,not their creator or broad bbox labels. `dedup8.py`v2 also
distinguishes nine metadata reference joins from five strongly associated bag
queries. None is counted as newly trusted evidence. The original41regions and
prior five inactive C21regions remain untouched.

`dataset-index-v1.json` binds the qualification manifest, source audit, actual
query coverage, authoritative deduplication-v2 and conditional rights shortlist.
The older dedup-v1 is retained for audit;its reference counts are not object
identity counts. All uncertain decisions remain inactive. Future data must use
another versioned output root;generators reject existing paths.

Read-only completed artifact verification:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e8-v1/verify8.py check
```

CPU regression tests (no GPU work or downloads):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 .venv/bin/python -B -m unittest discover -s tools/phase10e8-v1 -p 'test_qualification8.py' -v
```

No new negatives qualified. The exact64ID rights-evidence proposal in the report
requires separate approval before network requests. Photograph acquisition,
future inference and any training require their own approvals.
