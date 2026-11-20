This is an isolated CPU-only saved-prediction audit, not Phase10D training.
It never imports a model, loads weights or invokes inference.

The completed visual report is at
`reports/chair-fp-investigation-v1/index.html`; the prose findings are in the
neighboring `report.md`. Images and review materials remain local and ignored
by Git. No photograph is sent to an external service.

Use `.venv/bin/python -B -m unittest discover -s tools/chair-fp-audit-v1 -p
test_audit.py -v` for the nine focused CPU checks. Both preparation entry points
write exclusive artifacts and refuse existing finalized outputs. Use a new
versioned output for any future regenerated audit. Do not delete old reports
to make reruns possible.

`audit.py` reconstructs native events at the pre-existing confidence0.15,
reconciles all1,094 Chair FPs, and creates geometric triage, per-photograph and
per-prediction CSVs plus a deterministic48FP/8unscored review packet.
`visualize.py` draws existing native boxes and saved predictions on copies of
the existing local photographs. The HTML operates entirely offline.

Original and LR1e-5 per-image predictions were not saved for this exact19-photo
panel. Their overlays remain unavailable. Do not populate them using another
panel or infer per-photo values from aggregate totals. Geometry and assistant
visual observations are diagnostic hypotheses, never human confirmations or
automatic label corrections. Human CSV decisions remain blank.
