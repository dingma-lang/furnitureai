# Phase10C.18 — one attempt consumed, safety-stopped

The single authorized arm stopped **before update29**, after28committed updates. Do not rerun `launch18.py`, `worker18.py` or reset the output directory. No automatic recovery, new inference or extra variant is authorized.

The runtime is bound in immutable `data/open-images/experiments/phase10c18-source-balanced-v1/binding.json`. `common18.py` verifies the original model,59photos/56groups/182positives/41source regions,C8settings/order and C17seals. `worker18.py` and `update18.py` remain exactly as executed, including the safety guard that stopped the experiment. They have not been patched to continue.

`partial_analysis18.py` was used after the stop. It validates and scores only the already-complete epoch1 predictions, in a separate CPU process, alongside saved original/C8E1/C8E5 predictions. C18checkpoint28 has no scored predictions. `analyze18.py` includes helpers and a complete-run-only route that correctly refuses this incomplete attempt. All artifact-producing commands refuse overwrites; no outputs should be deleted/reset to rerun them.

`test_training18.py` covers scaled accumulation, actual CPU GradScaler/AdamW, scope, masks and all59cached photos. `test_completed18.py` verifies the actual stopped artifact counts and demonstrates a synthetic FP32 false-positive mechanism in the original extra safety measurement. The witness is not the missing failed GPU buffers. `verify18.py` reloads all29snapshots and checks historical preservation; `plot18.py`/`report18.py` use measured saved data only.

Use the existing project `.venv/bin/python -B` and `CUDA_VISIBLE_DEVICES=''` for CPU-only inspection/tests. No packages or environments changed. Reports and SHA-256 seals are in `reports/phase10c18-v1/`; checkpoints/raw arrays stay ignored and local. The proposed next action is CPU-only guard/telemetry review, followed by separate authorization before any recovery. Neither the25%cap nor the objective was silently repaired.
