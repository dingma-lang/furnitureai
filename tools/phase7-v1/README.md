# Phase 7 — isolated metadata and synthetic preparation

Run from `/home/tristary/furniture-ai-ml` with `.venv/bin/python -B`.
No dependency installation, image/asset acquisition, inference or training is performed by offline tools.

Authoritative artifacts are under `reports/phase7-data-strategy-v1/`:

- `report.md`, `next-approval-request.json`, `ranked-dataset-comparison.json`.
- `metadata-audit-v1.json`: published LVIS native training-image counts; unknown instances/overlap are null.
- `openimages-local-counts-v2.json`: raw local validation/test counts; training remains unknown.
- `lvis-overlap-targets-v2.json`: 77 COCO join IDs, including five source-photo matches to historical Open Images slots. The v1 cross-source list is superseded because it included COCO self-matches.
- `synthetic-pilot-v3.json`: exact six-asset/33-object specification and 100 seeded camera plans over 25 layouts. v1/v2 scene plans are superseded development drafts.
- `license-and-source-ledger.json`, `objects365-vocabulary-audit.json`, `unified-detection.schema.json`.
- `test-results-final.json` (155 passing tests), `preservation-verification-final.json`, source evidence/SHAs. Earlier test/preservation records document intermediate checks.

Examples of **no-network** commands (new output paths must not exist):

```bash
.venv/bin/python -B -m unittest discover -s tools/phase7-v1 -p 'test_*.py' -v
.venv/bin/python -B tools/phase7-v1/audit.py --output /tmp/phase7-category-audit.json
.venv/bin/python -B tools/phase7-v1/synthetic.py --output /tmp/phase7-synthetic-dry-run.json
```

Once annotation acquisition is separately approved and completed, the audit accepts a local official ZIP or extracted JSON:

```bash
.venv/bin/python -B tools/phase7-v1/audit.py \
  --lvis-json data/lvis/metadata/phase7-v1/lvis_v1_train.json.zip \
  --output /tmp/phase7-lvis-train-audit.json
```

Use a separate output for validation; never promote validation annotations into training. `--oi-train-csv` accepts an already-acquired CSV for raw counts; it does not perform the image-info/human-label/rights joins needed for candidate eligibility. Source split names passed to that audit are caller assertions until those joins pass.

`contracts.py` contains structural/semantic validation and transitive identity isolation. Missing native categories remain unknown. Keep original category IDs, masks, boxes and native completeness independently from seven-class grouping. The pure `synthetic.visible_annotations` function has tested raster boxes and uncompressed COCO RLE, including invisible/unknown-ID cases.

`render_worker.py` is prepared, **not Blender/GPU integration-tested**. Blender and BlenderProc are not installed. Do not invoke a BlenderProc launcher now: it may automatically download Blender/install packages. A later approval must bind the exact pilot JSON SHA, its asset objects, separately pinned runtime and a <=100-frame budget. Every glTF object must be present locally, match published bytes/MD5 and contain no unapproved dependency URI. No downloader is supplied or invoked by this worker. It emits pending synthetic candidates, never human-reviewed labels. Mesh scale, API compatibility, alpha/depth/instance registration and resource cleanup still require a real five-frame smoke subset before a larger render run. The initial synthetic pool is training-only.

`research.py` was a one-time bounded retrieval of small public documentation/category/catalogue text. It refuses existing files; it is not a dataset downloader. Additional documented small research and HEAD checks are preserved in source ledgers. The denied lamp metadata endpoint was not retried/bypassed.

`preserve.py` reads historical sealed files and hashes; frozen ABO pixels/features remain stat-only via the existing verifier. Outputs are exclusive files. Do not rerun report/test assemblers into the sealed report directory; use a fresh report version for new work.
