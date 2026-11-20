Phase10C.3 validates the completed65-item human export and prepares an additive
supervision overlay. No GPU/model/optimizer worker is implemented or invoked.

`preflight.py` validates source/queue/prediction identity, preserves every human
decision, records clarification flags separately and audits local native Chair
and Stool definitions. User policy includes ordinary seating/bar stools but
excludes footstools/ottomans/step stools. No native label is silently remapped.
`validate_region_record` adds a mandatory taxonomy/stool-coverage guard before
the unchanged C.2 evidence validator. All current scope decisions decline
completeness, so no regional evidence is accepted.

`mask_probe.py` compares masks on CPU synthetic query grids and immutable59-row
training metadata. With no evidence, both masks are equal for any input boxes;
the probe neither runs a model nor provides a real training-batch count.

The versioned private overlay, full review audit, clarification CSV and blank
region-evidence template live under `data/open-images/preparation/phase10c3-v1`.
The report and conditional disabled configs live under `reports/phase10c3-v1`.
Preparation entry points refuse existing outputs. Preserve historical versions;
do not delete them to rerun. Tests can be run without producing new artifacts:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c3-v1 -p test_preflight.py -v
```

No training approval is requested while the contrast is zero. See the report
for the proposed minimum six regions/three training groups and the independent
evaluation blocker. Regional review is distinct from validation diagnostics;
no held-out example can become a training negative.
