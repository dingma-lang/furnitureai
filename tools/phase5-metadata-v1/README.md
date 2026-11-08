# Approved Phase 5 metadata acquisition and pilot proposal

Use `.venv/bin/python -B` from `/home/tristary/furniture-ai-ml`. No dependencies or environments were changed. These tools do not train models. `preflight_images.py` only performs HEAD, never GET. No photograph downloader is implemented here.

The acquired files and all audit results are under `data/open-images/metadata/phase5-v1/`. Read `audit-report.md`, `metadata-integrity-verification-v2.json`, `CURRENT-PROPOSAL.json` and `image-acquisition-approval-request.json` there. The authoritative proposal is **pilot-proposal-v4**; earlier drafts are superseded and retained. `metadata-integrity-verification.json` is an initial diagnostic that counted the CSV header as a class; **v2** is the corrected authoritative integrity report. No raw metadata or historical artifact was changed to correct that parser issue.

The approved acquisition completed with 205,865,596 response-body bytes and no retries. The durable ledger retains charged/read bytes, attempts, partial hashes, source MD5, local SHA-256 and code/proposal bindings. All eight files were verified against the exact approved size, ETag and Last-Modified; both repeated GCS checksum headers were parsed correctly. Hard cap 220,000,000 bytes; allocated-storage cap 1 GiB. Existing finals are never replaced. Explicit resumption validates all eight remote HEADs, code, proposal and local hashes before reusing files:

```bash
.venv/bin/python -B tools/phase5-metadata-v1/acquire.py --resume
```

A killed read keeps its pre-reserved chunk charged conservatively. Resumption uses Range plus If-Match only from a durably hashed partial. Two attempts per object maximum. An incomplete local write, bad hash, changed source, unknown status or exhausted budget stops; do not delete state or increase limits to work around it. With all files complete, resume performs zero metadata GETs. The SHA ledger and bound downloader source should remain together.

Offline audit and proposal reproduction, into a **fresh** output directory:

```bash
.venv/bin/python -B tools/phase5-metadata-v1/prepare_pilot.py \
 --out data/open-images/metadata/phase5-v1/reproduction-v1

.venv/bin/python -B -m unittest discover -s tools/phase5-v1 -p 'test_*.py' -v
.venv/bin/python -B -m unittest discover -s tools/phase5-metadata-v1 -p 'test_*.py' -v
```

`metadata_dictionary.py` explicitly handles the acquired V7 `LabelName,DisplayName` header and 601 native IDs. Original Phase 5 tools and original verified Phase 2B rules are unchanged. Exact furniture mappings and counts match the original offline audit. All 601 IDs occur in the supplied hierarchy, whose additional root is Entity.

`prepare_pilot.py` validates source SHA-256s, applies documented metadata eligibility, groups the full source tables by photographer/source-photo/original-MD5 (including ineligible bridge photographs), ranks indoor metadata evidence, and selects the frozen proposal first. Training/development are interleaved 2:1 to share scarce classes without sharing identities. No predictions or holdout decisions influence selection. It excludes near-coincident sofa/bed and table/desk annotation ambiguities. Same original MD5 is not confused with the SHA of a resized CVDF object.

The proposed splits and image identities are not verified training data. Every actual image SHA, indoor status, image quality, current licence verification and orientation verification remains missing/false until authorized acquisition and screening. Existing evaluation bundle validators reject this proposal schema. It is prohibited to set those fields true just to satisfy a loader.

After separate image approval, verify per-photo current source rights and attribution, revalidate HEAD sizes/ETags and limits, acquire only the approved list, and verify decoding, orientation and annotation overlays, exact/pixel/perceptual duplicates and protected dataset overlap. Then freeze actual partitions using known groups and quarantine unresolved relationships. Do not substitute unapproved sources or download replacements from an expanded candidate list. Neither image acquisition nor detector training has started.
