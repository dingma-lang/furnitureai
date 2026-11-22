# Phase 10C.14 tools

These tools inspect historical training caches. There is no training/inference launcher, model loading interface, review exporter or automatic activation path.

Use the existing project interpreter with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c14-v1/server14.py --port 8774
```

Open http://localhost:8774/. GET-only endpoints serve the exact optimization-photo whitelist. Uncached photos remain uncached; no GPU fallback exists. Scores, reconstructed assignments, native annotations, source-supported negatives and uncertainty are separate concepts.

Initial analysis order was `preflight14.py`, `audit14.py`, `mining14.py`, `plates14.py`, targeted actual visual inspection, `visual_record14.py`, `analyze14.py`, `moments14.py`, `dashboard14.py`, `quality14.py`, `browser14.py`, `verify14.py`, `publish14.py`. Commands create immutable versioned outputs and intentionally refuse to overwrite completed records. Do not delete or reset an audit directory to rerun. A new analysis version requires new output paths. `visual_record14.py` preserves the observations actually made during this phase; executing it is not an independent new visual or human review.

CPU suites are enumerated in `verify14.py`. Historical preflight tests asserting that now-completed experiment directories do not exist are intentionally excluded; their frozen-parameter scope tests remain included. Browser fixtures are synthetic and explicitly identified, and live-browser photographs are optimization-only. The installed Windows Chrome is used without installation and with GPU disabled.

Detailed outcomes and limitations: `reports/phase10c14-v1/report.md`. The proposed source-negative BCE calculation is diagnostic only. A new explicit approval is required before any GPU gradient smoke or training.
