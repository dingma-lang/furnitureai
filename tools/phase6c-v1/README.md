# Phase6C completed — exact400-ID metadata-only rights audit

Artifacts: `data/coco/rights-audits/phase6c-v1/`.
Use project `.venv/bin/python -B`.

Read `report.md`, `summary.json`, `verification.json`,
`license-attribution-ledger.json`, `identity-audit.json` and the unapproved
`proposed-image-acquisition-v1.json` / `image-acquisition-approval-request.json`.

92 individual source grants verified;72 photos survive identity isolation,
60 train /12 development in the original slots,66 known source groups.
275 source checks remain temporarily unavailable under HTTP429. One normal
cooldown retry was still429, so all further retry-pass source requests stopped.
30 source redirects lacked a matching photo identity; one specific licence was
unrecognized; two source BY-SA licences differ from the old BY metadata and are
outside the approved policy. No unavailable/uncertain records were replaced.

`records/` preserves the initial overly narrow resolver pass. `records-v2/`
preserves the validated same-photo HTTPS rb=1 resolver pass. `records-v3/` and
`rights-audit-v3.json` contain final outcomes, retaining earlier findings when
no eligible retry was made. The common durable ledger is never reset across
these passes. No forbidden/missing-page retries, authentication or rate-limit
workarounds were used.

Network rules: only the exact frozen source IDs; TLS verification required;
manual same-ID Flickr redirects; GET only for source HTML; specified S3 images
categorically HEAD-only. Source HTML is bounded to1MiB compressed/4MiB decoded,
two GET attempts maximum,256,000,000 cumulative body bytes and512MiB allocation.
Perceptual duplicate and visual checks were not performed and are not claimed.

The photo proposal's primary identity_group uses the reconciled current source
group. Old metadata_identity_group is retained separately. Shared creator/source
families are not claimed to represent one physical furniture identity. All
cross-partition and protected historical source groups were quarantined without
moving images. Original COCO boxes, masks, IDs and categories are unchanged.

Read-only inspection/tests:

```bash
cat data/coco/rights-audits/phase6c-v1/summary.json
cat data/coco/rights-audits/phase6c-v1/image-acquisition-approval-request.json
.venv/bin/python -B -m unittest discover -s tools/phase6c-v1 -p 'test*.py' -v
```

Do not rerun completed output-generation commands in this directory: they use
exclusive versioned records. The rights runner's explicit --resume supports an
incomplete journal/pass; completed findings are cached, not overwritten or
retried to change an outcome. A new future source availability audit needs its
own bounded revision and must respect service cooldown/access restrictions.

The proposed72-photo acquisition expects10,853,122 bytes, with100,000,000 transfer
cap /512MiB storage, two attempts and3,145,728bytes/photo, under a fresh
`data/coco/image-pilots/phase6d-v1/`. This is NOT approved by the current rights
audit. All proposed image hashes remain null. No training/inference/weight or
Mac-project changes are authorized or performed.
