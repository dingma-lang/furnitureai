# Completed Phase6D COCO acquisition and screening

Use project `.venv/bin/python -B`.
Artifacts: `data/coco/image-pilots/phase6d-v1/`.
Read `report.md`, `summary.json`, `integrity-verification.json`,
`phase6e-training-readiness.json` and `resource-usage.json`.

All72 approved JPEGs acquired first attempt,10,853,122 bytes. Original slots
remain60 train /12 development. Final gate retains55 provisional candidates,
45 train /10 development;16 quarantined and1 rejected composite remain stored.
All362 native source boxes/polygon annotations remain unchanged. Survivors have
280 native boxes/mask annotation objects across52 metadata-known source groups.
No human application-label/boundary verification or model accuracy is claimed.

`CURRENT-SCREENING.json` pins screening-v2 by SHA. Only the55 rows in
`eligible-candidates-v1.json` passed the final gate; the earlier screening-v1 is
not a training-eligibility manifest. No permanent dataset lock or training
directory was created. The historical Phase5C27-image worker must not be reused
unchanged for expanded data. A reviewed variable-size, source-specific partial
supervision adapter/config and separate explicit training approval are required.

Read-only inspection/tests:

```bash
cat data/coco/image-pilots/phase6d-v1/summary.json
cat data/coco/image-pilots/phase6d-v1/phase6e-training-readiness.json
.venv/bin/python -B -m unittest discover -s tools/phase6d-v1 -p 'test*.py' -v
```

The original download approval is recorded in authorization.json, separately from
the historical proposed manifest's unapproved-at-the-time flags. The download
runner supports explicit `--resume` against the same bounded journal; verified
completed objects skip without another GET. No resume is needed now. Do not
delete the ledger, alter IDs or rerun exclusive output-generation programs to
force overwrites. Caps:100,000,000 body bytes,3,145,728bytes/photo,two attempts,
512MiB allocated storage. TLS default verification and size/ETag/type/range pins
are mandatory. There is no model-loading or training entrypoint in these tools.

Actual COCO photos were inspected through five bounded contact sheets and seven
targeted full-size views. Near-coincident chair/couch/bed source categories are
quarantined, not relabelled. Legitimate occlusion/border contact is a diagnostic,
not automatic rejection; no absent furniture becomes confirmed background.
Historical/frozen pixels were not decoded or scored; old files and model hashes
remain unchanged. Mac simulation and production application files are untouched.

Phase6E assessment:81 nominal combined training images,74 prospective positive-
scope candidates after conservative legacy masks,15 fresh development images.
These are accounting views, not changes to old partitions. Keep the old18-image
development panel and all frozen material out of new tuning. Broad OI Table and
COCO category67 remain separate source strata; Bed has only one fresh development
photo and Lamp has none, while Desk/Cabinet lack conservative positive training
support. No Phase6E experiment is approved or begun.
