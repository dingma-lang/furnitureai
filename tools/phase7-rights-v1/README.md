# Phase7 LVIS photograph rights audit

Use the existing project `.venv/bin/python -B`. Isolated code; destination `data/lvis/rights-audits/phase7-v1/`.
This stage retrieves licence evidence only. Exact approved400 IDs, provisional320/80 and905 core masks are SHA-bound to metadata proposal `15340ef1f78c1f34c6cb9be10fd3b7d6e5d07528bc393f514f6f4a5ef89e014f`.

Preflight compares all1080 original native annotations against the sealed LVIS source archive and validates dimensions, previous400 exclusions, five frozen-related exclusions, known source groups and assignments. It has no network client. Historical source files are read-only.

`run.py` performs a single candidate evidence attempt: <=4 same-ID HTTPS source HEAD resolutions, <=1 HTML GET, and an official-photo HEAD only after source rights pass. It never GETs a photograph. One worker, three-second source pacing, original stable client identity, verified TLS, 1MiB encoded/4MiB decoded page limits, durable byte reservations and a hard128,000,000-byte response-body cap. A429 opens a durable provider circuit, retains Retry-After and prevents all later requests to that provider. No automatic retry occurs, even after cooldown. Unattempted candidates receive explicit checkpoint status rather than fabricated verification. The512MiB allocated cap includes this code and a16MiB reserve.

`source_evidence.py` reuses the sealed same-photo JSON-LD/licence parser. Source licence/creator/title must match the photo ID; a dataset licence ID alone cannot pass. `finalize.py` keeps rights and identity exclusions distinct, propagates source-bound creator bridges, rechecks historical groups after unions, and quarantines mixed train/development groups without reassigning them. Equal HEAD signatures are suspicions, not proven pixel duplicates.

`registration.py` hashes ten already-local COCO photos and compares source coordinate metadata for33 extra LVIS annotations. It does not decode photographs, claim visual alignment or activate annotations. All semantic/visual and partial-label requirements remain open.

`unattempted-checkpoint-v1.json` is **not authorization to continue requests**. Any future continuation of never-attempted records needs separate approval; source circuits are not automatically cleared. `--resume` may recover same-phase bookkeeping or reuse completed results; it cannot create a second candidate attempt or retry a suspended provider. If a crash consumed an attempt without a final record, it remains unresolved rather than re-fetching the page.

After finalization: report.md; license-and-attribution-ledger.json; rights-decisions.csv; annotation-value.json; identity-audit.json; existing-ten-photo-registration-assessment.json; proposed-image-acquisition-v1.json; next-approval-request.json; integrity-verification.json; test-results.json and source response records/hashes. No proposal or stage automatically grants photograph acquisition, training, inference or rendering approval.

No-network tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase7-rights-v1 -p 'test_*.py' -v
```

Once acquisition records are complete, finalization, verification and reporting read local artifacts only:

```bash
.venv/bin/python -B tools/phase7-rights-v1/finalize.py
.venv/bin/python -B tools/phase7-rights-v1/run_tests.py
.venv/bin/python -B tools/phase7-rights-v1/verify.py
.venv/bin/python -B tools/phase7-rights-v1/report.py
```

Outputs are exclusive. Do not rerun assemblers into completed/sealed directories; use a new version. Do not use an acquisition manifest unless separately approved by the user.
