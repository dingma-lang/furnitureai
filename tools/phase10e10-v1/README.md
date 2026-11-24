# Phase 10E.10 metadata discovery

This closed audit used only existing project files, public documentation and the
Wellcome Collection's documented no-auth catalogue/image metadata APIs. It has no
photo downloader, model loader, GPU or training entry point. Binary image requests
and Flickr API requests are rejected. Blocked providers, robots restrictions,
rate limits, retries, redirects, credentials and interrupted transactions fail
closed. Source documents and receipts live under ignored `data/rights-discovery/phase10e10-v1/`.

Safe read-only verification:

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e10-v1/verify10.py check
```

Do not rerun discovery/finalization after closure. The closed marker prevents
new requests; raw receipts preserve the exact prior responses. Any new discovery
requires a new authorized version and must honor existing provider restrictions.
Tests use synthetic TLS responses and real sealed metadata, without accessing the
network or loading images/models. `python -m unittest discover` is the test entry
point. A separate authorization is required for the seven-photo inspection plan;
it would not authorize training, inference, annotation certification or evaluation.
