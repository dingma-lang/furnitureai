# Phase 10C.17 completed zero-update smoke

The single59-forward/295-VJP attempt is consumed. **Do not rerun `launch17.py`.** One-shot records prevent replay. No training or extra GPU variant is authorized.

`prepare17.py` binds original OWLv2,C8 epoch0 ordering,59optimization photos/56known groups/182positives/41regions and unchanged C16 code. `cpu_preflight17.py` checks the original cached masks and driver before execution. `gpu17.py` computes five isolated branch VJPs per photo with12trainable heads,including calibration,then applies the unchanged C16 parameter cap after each actual accumulation group. No optimizer or model-gradient assignment exists.

Important preflight finding: C16 `Accumulator.finish()` correctly composes buffers but its diagnostic `scale` is overwritten by the AMP scale. `measure17.finish_accumulator()` invokes the same unchanged C16 `compose` on exactly the same unscaled buffers and records `amp_scale` separately. CPU tests prove bit-identical outputs;C16 files remain immutable. The algorithm and25%coefficient were not changed.

`analyze17.py`, `plot17.py`, `future17.py` and `verify17.py` are CPU-only. Use `.venv/bin/python -B` with `CUDA_VISIBLE_DEVICES=''`. They validate immutable records and refuse overwrites. Do not delete/reset a completed experiment to repeat it.

Data stays under `data/open-images/experiments/phase10c16-balanced-gradient-smoke-v1/`, the destination proposed inC16. Compact reports are under `reports/phase10c17-v1/`. All source-positive/base-box gradient buffers and model parameter/buffer hashes are preserved. The cap complied but never activated on original easy negatives;this is explicitly not proof of clipping behavior at a later trained state or of recognition accuracy.

The only proposed next model experiment is disabled in `future-training-disabled.json`. A new trainer and explicit separate approval are required before its75updates/5epochs. No production checkpoint or historical experiment is modified.
