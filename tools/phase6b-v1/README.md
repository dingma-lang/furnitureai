# Completed Phase 6B COCO2017 metadata-only acquisition

Use the project `.venv/bin/python -B`.
Artifacts: `data/coco/metadata/phase6b-v1/`.
Read `report.md`, `summary.json`, `integrity-verification.json` and
`resource-usage.json`. No photographs, models, training or detector inference.

The 252,907,541-byte ZIP matches the independent published SHA256. Only the two
instance JSON members were extracted, 489,773,314 bytes combined, with CRC checks.
The full ZIP plus full extraction would exceed the approved 1GiB storage cap.

`candidate-inventory.jsonl` contains all 25,282 furniture-photo metadata records,
with explicit licence-claim and unverified-current-rights status. The 400-member
`proposed-image-candidates-v1.json` is an unapproved rights-audit queue, with only
draft 320/80 source-group allocation. It is not a photo-acquisition permission or
permanent split. Current creator groups, actual bytes/pixel duplicates and scene
suitability must still be verified. No Desk/Cabinet/Lamp labels are invented.

Useful read-only commands:

```bash
cat data/coco/metadata/phase6b-v1/summary.json
cat data/coco/metadata/phase6b-v1/next-rights-audit-proposal.json
.venv/bin/python -B -m unittest discover -s tools/phase6b-v1 -p 'test*.py' -v
```

Acquisition/extraction/audit/report tools intentionally preserve exclusive
versioned outputs rather than overwriting completed artifacts. Do not rerun the
audit/extraction/report in this completed directory. The authorized archive
downloader supports explicit `--resume` on the same incomplete journal/ETag;
verified completed archives skip without another archive request. No resume is
needed now. A partial must match the committed prefix byte count and SHA, and a
range response must match exact status/content-range/size/ETag; otherwise it
stops. Never delete the transfer ledger to reset the two-attempt cap.

`executed-acquisition-code-v1/` preserves initial executed acquisition/extraction
code. Final implementation hardens exclusive atomic completion/recovery and
bounded JSON scientific-number parsing; tests cover these paths. Final
`acquisition-verified-commit.json` / `extraction-verified-commit.json` atomically
bind the original receipts and actual verified files. Historical project files
and original first-completion records were not replaced.

`next-rights-audit-proposal.json` is a separate approval request for matching
creator/licence source-page metadata for the 400 listed IDs, plus body-free
official image HEADs, with no photos. It is currently unapproved. Future
photograph acquisition and any detector updates require their own explicit
approvals. Do not reuse the archive approval for a new archive or photograph.
