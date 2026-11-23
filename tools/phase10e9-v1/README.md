# Phase 10E.9 — bounded photograph rights audit

Read `reports/phase10e9-v1/report.md` and `rights-audit-index-v2.json`.
The approved E8proposal's rights destination is
`data/coco/rights-audits/phase10e8-v1/`;this is the E9audit's new data root,
not the older Open Images E8preparation directory.

The audit is **closed**. All64source IDs were audited against local COCO/native
annotations and historical identity records. The only external request retrieved
Flickr's robots policy with certificate-verified TLS. Its default rule denies
this audit client. No individual photo pages or image objects were requested.
No user-agent rotation, alternate client, new provider/API access or retry ran.

Results:0rights-supported,64unresolved,0unsuitable. Access restrictions are not
copyright findings. Creator,current source licence and specific training-use
conditions remain unverified. `license-attribution-ledger.json` preserves one
explicit record perID,with null verification fields and shared policy-access
evidence distinguished from nonexistent photograph-specific licence evidence.

`network-final-ledger.json` is authoritative for final circuit/transfer state.
The first ledger is the earlier policy-stage snapshot,whose circuit field predates
the explicit persistent robots-policy circuit. Completed audits cannot resume
network requests. Do not rerun `policy9.py`,change the client name or invoke an
alternate endpoint to work around the restriction. Future permitted access must
be assessed under a fresh,explicitly authorized rights-only stage.

TLSMemoryBIO meters encrypted bytes in both TCP-stream directions,including
handshakes and read-ahead;response-body bytes are separately recorded. Physical
IP/TCP/DNS/retransmissions are not directly observed. The conservative charge
adds64KiBpertransaction and retains full reservations for interrupted transactions.
The whole source audit's conservative charge is77,260bytes,below32,000,000.
This source-data ledger excludes the separately requested small Git backup.

No photograph bodies,models,datasets or packages were downloaded. Source HTML
would have been bounded and parsed only as documentary metadata,but provider
policy prevented those requests. Raw policy bytes,headers and full native
annotation bodies stay local;no response cookies or photo bodies are published.

The pre-network test fixtures live under `cpu-fixtures/` and are marked synthetic;
they never enter actual network totals,per-ID rights records or acquisition lists.
Only direct children of `requests/`describe real external HTTP transactions.

Read-only final verification:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e9-v1/verify9.py check
```

Post-audit CPU tests can be repeated without network access:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
 -s tools/phase10e9-v1 -p 'test_completed9.py' -v
```

`proposed-training-acquisition.json` and the separate independent-evaluation
acquisition manifest are both empty. The latter retains an independent new-room
collection workflow;none of the64training candidates is reassigned to evaluation.
`later-annotation-verification-plan.json` keeps automatic proposals separate from
human confirmations and does not require routine manual rectangle drawing.

Before any photo acquisition: obtain provider-permitted evidence access or
acceptable photograph-specific creator documentation,verify rights and groups,
produce a nonempty exact survivor manifest and resource budget,and request
separate photograph approval. The earlier64image size estimate is not approval.
