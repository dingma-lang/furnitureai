# Phase7 LVIS metadata acquisition/audit

Isolated implementation. Use `/home/tristary/furniture-ai-ml/.venv/bin/python -B`.
Outputs are exclusively metadata/provenance under `data/lvis/metadata/phase7-v1/`.
No photograph download, model, inference, training, environment installation or renderer interface exists.

The exact approved proposal is SHA-bound in core.py and authorization.json. preflight.py checks current official source/licence text, TLS, pinned sizes/ETags and independently published TensorFlow SHA256 values. acquire.py enforces durable cumulative reservations, two GET attempts/object, safe source-bound range recovery, exclusive publication and a 1GiB allocation cap including this implementation. It stops on error rather than silently retrying sources.

Safe recovery command (already tested on completion; it skips verified successful objects):

```bash
.venv/bin/python -B tools/phase7-acquisition-v1/acquire.py --resume
```

Never restart preflight or recreate a completed audit index in place. Outputs are exclusive; future changes need fresh versions. An uncertain partial hash/offset is refused, not truncated. Hard termination between file and ledger writes can require inspecting the retained prefix; it never permits an uncharged retry or overwriting a completed archive.

reader.py verifies published hashes and ZIP CRCs and reuses the sealed bounded JSON reader without extraction. audit_lvis.py validates native taxonomy/counts/scope into a compact SQLite index. cross_reference.py joins only local metadata and saved rights evidence, and selects 400 unapproved rights-audit IDs. refine_audit.py checks all known frozen/source groups; resolve_status.py preserves old `eligibility_status` and newer `status` fields in authoritative v2 overlays. ancillary.py independently counts related seating/storage/part categories without merging them. report.py creates the report, licence ledger and blocked conditional photo proposal. verify.py checks sources, ledgers, historical model/data seals and the preserved Phase7 preparation report/implementation.

Authoritative outputs: report.md; furniture-counts.json/csv; additional-unmapped-furniture-categories.json; existing-photo-cross-reference-v2.json; annotation-overlay-summary-v2.json; frozen-and-source-group-isolation.json; licensing-and-attribution-ledger.json; proposed-photo-rights-audit-v1.json; conditional-photo-acquisition-proposal-v1.json; next-approval-request.json; integrity-verification.json; test-results-final.json. Earlier v1 overlay/status diagnostics are retained and explicitly superseded; source datasets were never edited.

No-download CPU tests:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase7-acquisition-v1 -p 'test_*.py' -v
```

The original LVIS and COCO annotation arrays stay source-specific. `same-object` IoU pairings are geometry diagnostics only. Unmatched or newly named native labels are not proof of additional physical furniture. Missing/federated nonexhaustive labels cannot create background negatives. All new photo rights remain unverified, and the provisional 320/80 allocation cannot become permanent until identity and rights checks pass.
