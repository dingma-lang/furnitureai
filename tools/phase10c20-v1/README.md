# C20 completed one-attempt recovery

The47-update recovery is consumed:75cumulative updates,270new/452total forwards. Do not rerun `launch20.py` or `worker20.py`, delete one-shot markers, restart from original heads, or extend the queue. No further GPU work or deployment is authorized.

`worker20.py` derives from immutable C18. It preserves preprocessing, loss branches, scaled3VJP accumulation, optimizer settings and head scope; it restores checkpoint28 before any forward, uses C19's exact direct guard and routes failure receipts to the new approved runtime destination. The original frozen backbone is reconstructed locally, not trained anew. GPU RNG roundtrip passed. NumPy RNG was absent/unused in C18 and was not fabricated; recovery snapshots add its metadata without consuming randomness. Bitwise equivalence to a hypothetical uninterrupted run is not claimed.

Runtime/checkpoints/raw outputs remain local under `data/open-images/experiments/phase10c19-recovery-v1/`, exactly the approved proposal destination. `analyze20.py` uses saved raw predictions and unchanged CPU metric code in a separate process. `verify20.py` checks all47snapshots, original-data integrity and saved metrics. Reports are under `reports/phase10c20-v1/`. Artifact writers refuse overwrites.

The pre-GPU CPU-test binding was versioned to omit only C19's old future-destination-absent assertion after C20 explicitly authorized creation. The original binding is preserved. No numerical/worker objective changed through that revision. New runtime context/output-path tests passed before GPU.

Result: no useful precision–recall advantage over matched C8. The small precision gain at0.15 accompanies lower recall/AP; matched-recall precision also worsens. No production threshold/checkpoint changed. The next recommendation is source-supported hard-negative evidence development,not more epochs. All claims remain restricted to the exposed partial-label19-photo diagnostics.
