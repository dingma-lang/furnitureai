# Phase 10C.15 diagnostic

One authorized GPU attempt stopped after five forwards/four completed summaries. **Do not rerun `launch15.py`.** The one-shot record already exists, and repaired code does not match the consumed authorization. No automatic resume, model training or alternative GPU test is authorized.

The exact executed source is retained privately under `data/open-images/experiments/phase10c15-source-negative-gradient-smoke-v1/executed-source-v1/`. Current `loss15.py` restores historical FP32 box-loss normalization order. Current `gpu15.py` also preserves per-photo and failure telemetry for a possible future separately approved run. These repairs have only been tested on CPU.

`analyze15.py` validates existing partial prediction/gradient files and generates CPU-only reports. It never loads a model or launches inference. Output records are immutable and commands intentionally reject overwrites. `verify15.py` runs CPU tests and historical integrity checks. `prepare15.py` requires a fresh destination and the original parent; it cannot reset a consumed experiment.

Use `.venv/bin/python -B` and `CUDA_VISIBLE_DEVICES=''` for CPU work. Historical data/model files, environments and original annotations must remain unchanged. See `reports/phase10c15-v1/report.md` and the explicitly unapproved `recovery-proposal.json`.
