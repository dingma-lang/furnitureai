# Phase 6A commercial-license detection data preparation

Use `/home/tristary/furniture-ai-ml/.venv/bin/python -B`.

Final artifacts: `data/open-images/preparation/phase6a-v1/`.
Read `report.md`, `summary.json`, `acquisition-manifest-v2.json`,
`license-attribution-ledger.json`, `annotation-scope-audit.json` and
`verification.json`. Final manifest v2 replaces only the *proposal* partitioning
of the retained v1 draft; no historic dataset or frozen partition changed.

Preparation completed with 41 individually source-verified CC BY2.0 photographs
proposed, not downloaded. Seven original-source rights checks were quarantined.
Only 15 train / 6 development images have potentially usable unambiguous coverage
after conservative taxonomy masks, before image screening. It is not a verified
400-photo dataset or a ready seven-class training set.

Implemented modules:

- `prepare.py`: existing official Open Images metadata, full source grouping,
  category conflicts, indoor evidence, deterministic candidate ranking.
- `rights.py`: canonical original Flickr ImageObject rights/creator evidence;
  bounded source-page HTML requests and body-free official photo HEADs only.
- `finalize.py`: deterministic source-group-stratified final proposal.
- `network.py`: durable capped ledger, source whitelist, no redirects, at most
  two attempts, conservative partial-transfer accounting and safe restart.
- `acquire.py`: future approved-manifest-only photo acquisition and integrity.
- `annotation_scope.py`: preserve raw rows, detect child-only annotation scope,
  mask unknown optimization coverage instead of inventing parent boxes.
- `screen.py`: future exact/perceptual hashes, quality and overlap quarantine;
  no human flags, listing merges or confirmed indoor claims are fabricated.
- `training.py`: variable-size batching/schedule/data adapter, original and
  Phase5C offline baseline interfaces, proposed config and separate training
  approval gate. No detector-update entrypoint is launched by preparation.
- `dry_run.py`: no-download metadata-only prospective schedule, not training.
- `verify.py`: old artifact/code/weight hashes and proposal/rights isolation.

Tests can be rerun without network, model downloads, GPU or training:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase6a-v1 -p 'test*.py' -v
```

Preparation programs create exclusive, versioned outputs and intentionally reject
overwriting completed artifacts. Do not rerun them against the existing output
directory. Existing tests do not alter sealed Phase 1–5 files. The Mac simulation
application is never accessed.

The future image command, **only after separate approval bound to the final
manifest SHA**, is:

```bash
.venv/bin/python -B tools/phase6a-v1/acquire.py --approval /path/to/explicit-image-approval.json
# Only for the same incomplete manifest-bound acquisition:
.venv/bin/python -B tools/phase6a-v1/acquire.py --approval /path/to/explicit-image-approval.json --resume
.venv/bin/python -B tools/phase6a-v1/screen.py
```

The approval record must explicitly contain `image_acquisition_approved: true`,
the exact `manifest_sha256`, destination absolute path, `max_transfer:1073741824`
and `max_storage:2147483648`. No such authorization record has been created.
Never flip flags in a report or reuse an old approval. Rights older than seven
days require a newly audited/versioned manifest and a new approval. No image
source substitution or replacement for quarantine is permitted.

No new GPU updates are authorized. Final screened row count, class coverage,
runtime/checkpoint queue and update budget must be pinned and separately approved
before future training. Frozen evaluation and official Phase2B remain unused.
