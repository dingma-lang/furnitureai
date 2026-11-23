Phase10E.5 ran exactly one authorized original-initialized student to75 updates.
The destination is immutable and must not be relaunched or overwritten:
`data/open-images/experiments/phase10e-distillation-fp32-v1/`.

Pinned E4 precision and objective are reused unchanged. The E5 additions are a
representative three-photo combined-loss gate inside the59 original replays,
actual positive/KD parameter-gradient monitoring,and versioned CPU analysis.
No optimizer is constructed until the full safety gate completes. No automatic
precision/loss/coefficient retry exists. The launcher enforces a600s hard bound.

Read `reports/phase10e5-v1/report.md`,the comparison CSV,full metric/PR JSON and
scientific SVG. No weights,photographs or raw prediction arrays belong in Git.

Read-only verification:

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e5-v1/verify5.py --check

`analyze5.py` deliberately imports no Torch and evaluates only saved predictions
on the exact19 exposed internal-development images. It preserves the existing
metric source,scope flags,NMS,score floors and matching. It cannot perform GPU
inference. `integrity5.py` loads checkpoints on CPU only to inspect tensors,
optimizer counters and hashes;it does not construct or run a detector.

No follow-on training,downloads,threshold adoption or deployment is authorized.
