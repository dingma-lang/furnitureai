# Completed Phase 6A approved 41-photo acquisition

Report: `data/open-images/image-pilots/phase6a-v1/report.md`.
Exactly 41 approved objects acquired, with no retries or extra photo downloads.
Final eligibility: 9 training / 5 development, provisional; 27 quarantined.
No detector updates, trained-model inference, environment changes or deployment.

`CURRENT-SCREENING.json` pins the effective **screening-v3.json**, not the first
automatic screening. Use `eligible-candidates-v1.json` for the 14 surviving
members. It is not an approved training lock. The historical preparation adapter
which references screening-v1 must not be used for future final eligibility;
final training integration requires a separately approved adapter/config bound to
the current screening hash. No old code or annotation was silently rewritten.

New diagnostic scripts create exclusive outputs. They are recorded execution
tools, not idempotent rerun commands against the completed directory. `refine.py`
preserves a separate v2 predecessor and writes the final v3 conservative scope
decisions. Assistant visual diagnostics do not set human-review flags.

Read-only inspection:

```bash
cat data/open-images/image-pilots/phase6a-v1/summary.json
cat data/open-images/image-pilots/phase6a-v1/integrity-verification.json
cat data/open-images/image-pilots/phase6a-v1/CURRENT-SCREENING.json
```

Rerun the acquisition-specific tests with no network/model operations:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase6a-acquisition-v1 -p 'test*.py' -v
```

All 75 Phase5–5C tests passed after acquisition, as did these 13 tests. The 22
Phase6A preparation safety tests passed before acquisition; their pre-acquisition
destination-absent assertion is lifecycle-specific and is not a post-acquisition
requirement. Tests use isolated processes to avoid historical module collisions.

The original authorized downloader and bounded transport stayed unchanged. Its
manifest approval is retained as `authorization.json`. No resume is needed for
this complete acquisition. If recovery had been necessary before completion, it
would have used the original downloader with that same approval and `--resume`;
do not rerun it to overwrite completion records or force another download.

`expansion-proposal-v1/` contains unapproved next-stage metadata proposals. COCO's
same-bucket HTTPS path-style source passes normal certificate verification and
has a known expected SHA from Google TFDS. No COCO archive has been fetched.
The larger Open Images training-metadata route needs explicit larger transfer
and storage approval, especially for Desk/Lamp coverage. No current approval may
be reused for either metadata proposal or a future photo/train batch.
