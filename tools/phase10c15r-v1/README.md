# Phase 10C.15R completed recovery

One authorized recovery reuses four immutable C15 results and processes indices5–24 only. It performs20new/25cumulative forwards,140new/168cumulative autograd calls and zero optimizer steps. **The launcher is consumed;do not rerun it.**

`prepare15r.py` binds the original inputs, repaired historical loss and fresh recovery code. `cpu_preflight15r.py` runs the25pre-GPU tests, including all24cached box-loss equality cases. `launch15r.py` enforces one launch and the175/180-second stop window. `gpu15r.py` imports the repaired loss without modification, uses one shared graph for both formulas and saves per-photo fingerprints plus periodic failure telemetry. No optimizer/training interface exists.

`analyze15r.py`, `detail15r.py` and `verify15r.py` perform CPU-only combination, plots/tests and historical checks. Use the existing `.venv/bin/python -B` with `CUDA_VISIBLE_DEVICES=''` for those tools. Immutable output records intentionally reject overwrites;do not delete/reset completed output directories to rerun.

All outputs are private under `data/open-images/experiments/phase10c15-negative-gradient-recovery-v1/`, with small public summaries in `reports/phase10c15r-v1/`. The original attempt remains untouched. Its missing peak VRAM and terminal hash remain missing. The four reused box-gradient summaries predate the rounding repair;classification gradients are independently valid and unchanged. Full24signed parameter-gradient vectors are unavailable;20-only aggregates are labelled separately.

See the completed report and source-group table. The result identifies excessive/uneven gradient risk at the unchanged BCE coefficient,not a measured recall loss. No additional variant, weight search, model evaluation or training is authorized.
