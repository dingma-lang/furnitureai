# Phase 10E.11 — closed exact-seven suitability pilot

The approved seven images are under the ignored destination specified by the
user. Acquisition is closed. No retries, alternate IDs, resizing requests, model
loads, inference, training or active annotations may be added to this run.
The downloader refuses an existing destination and pins all source URLs and IDs.
Receipts include failed/redirect transactions, TLS certificate verification,
measured stream/body bytes and separate conservative transfer allowances.
The output allocation check covers data, tools and reports; successful files are
never overwritten. Interrupted/failed work is retained and stops automatically.

Read-only artifact verification:

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e11-v1/verify11.py check
```

Read-only gallery and blank future intake interface:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e11-v1/serve11.py --port 8782
```

Open `http://localhost:8782/gallery.html`. Images are the original downloaded
1600px derivatives; clicking shows them at full acquired size. Only the gallery,
blank intake page, review JSON and exact seven images are served. Uploads,
directory listing, arbitrary files and cross-origin network scripts are disabled.
The intake page exports metadata only after explicit fields/checkboxes; no photos
are uploaded, solicited or used by this phase. The blank template is not consent.

CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tools/phase10e11-v1 -p test_pilot11.py -v
```

Transport tests use synthetic replies; actual image tests only decode the
approved existing photographs. Duplicate comparison uses 59 optimization and19
previously exposed internal-development references, never frozen/reserved photos.
Visual observations are assistant assessments, not human ground-truth annotation.
