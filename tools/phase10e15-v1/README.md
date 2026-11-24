# Private modern-room intake

This phase prepares tools and an acquisition proposal. No real photograph is
imported, copied, annotated or downloaded by these instructions automatically.
The old Wellcome pilot remains closed and unchanged.

## Local interface

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e15-v1/serve15.py --port 8783
```

Open **http://localhost:8783/** in your Windows browser. The server binds only
127.0.0.1, serves three exact static routes and rejects uploads, private paths and
unexpected Host headers. All photo/evidence files stay in the browser. Network
connections from the page are blocked by CSP. Closing/reloading clears unsaved
state: export a new version before closing. No remote model or annotation API is
called. The server does not create a photograph library or activate a dataset.

After separate acquisition/intake authorization:

1. Choose one already authorized photograph. The browser computes its original
   SHA-256 and shows the oriented image/dimensions. Check orientation explicitly.
   Changing the photograph resets permission confirmations and completeness.
2. Enter stable photo, creator, source, physical-room and capture-session IDs.
   Record actual source/credit, fixed purpose and prior exposure. Use opaque room
   IDs, not addresses. A previously exposed photo cannot become acceptance data.
3. Identify the genuine licence/grant and load its evidence file for a local hash.
   Explicitly confirm the separate rights/privacy scopes. A template or checkbox
   does not establish actual ownership or authority; qualification remains pending.
4. Optionally import **existing source boxes** in the envelope below. They must
   match the exact image hash, oriented dimensions and coordinate frame. Original
   labels/coordinates/masks stay in native evidence. Every imported box becomes
   a proposal, regardless of decisions embedded in an untrusted input file.
5. Correct category/coordinates if needed, enter reviewer/date and concrete
   evidence, and explicitly confirm individual objects. Ignore ambiguous regions
   with reasons or reject proposals; rejection is never a negative label. Optional
   dragging creates a new proposal; numerical coordinate editing and undo are
   supported. Editing/undo resets category completeness to avoid stale attestations.
6. Review completeness for **all seven** categories. Complete/absent decisions
   need category-specific evidence. Independent-evaluation export also requires
   unexposed attestation and resolved proposals. Missing annotations remain unknown.
7. Export a uniquely named inactive JSON. Save exports, originals and permission
   evidence privately, for example under ignored
   `data/licensed-rooms/intake/modern-room-v1/` after approval. Do not add private
   photographs, evidence, contact details or these submissions to Git.

Offline verification of a future authorized submission:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e15-v1/intake15.py /private/path/export.json --image /private/path/original.jpg --existing /private/path/other-record.json
```

The default exclusion registry is the existing E6 historical image/hash/group
registry. Supply an expanded authoritative registry with `--exclusions` when new
records or aliases are known. String matching cannot discover unknown physical
room aliases. The validator reuses E11 permission rules, verifies explicit image
hash/dimensions when `--image` is given, and never declares rights verified,
activates annotations or grants training/evaluation eligibility automatically.

## Source-annotation envelope

The following is a **synthetic format example**, not real annotations or a grant:

```json
{
  "schema": "modern-room-source-annotations-v1",
  "image_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "width": 100,
  "height": 80,
  "box_frame": "exif-oriented-xyxy-pixels",
  "source_id": "SYNTHETIC-SOURCE",
  "source_file_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "annotation_license": "SYNTHETIC EXAMPLE ONLY",
  "annotations": [{
    "id": "native-instance-1",
    "native_label": "chair",
    "suggested_category": "chair",
    "box": [10, 10, 40, 60],
    "native_annotation": {"id": 1, "bbox": [10, 10, 30, 50], "format": "synthetic-COCO-XYWH"}
  }]
}
```

An authorized metadata adapter must convert normalized/raw-EXIF/XYWH source boxes
to this explicit oriented frame, retaining the native record and transformation
provenance. This page does not silently guess those conversions or interpret
native federated annotations as complete negatives. Native masks remain
unreviewed unless separately checked. No source dataset was modified this phase.

## Checks and limitations

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tools/phase10e15-v1 -p test_intake15.py -v
```

Tests use synthetic evidence/photos and local HTTP requests. No browser automation
runtime or Node is installed: HTML/privacy/export gates and metadata/EXIF behavior
were checked by CPU tests, not a claimed browser end-to-end run. Manually walk
through the page using a synthetic image before first real intake. No dependency
was installed. E14's 425 passes/33 skips and 34 synthetic passes are the recorded
reliability baseline, not an expensive suite rerun in this phase.

`prepare15.py` is a one-time read-only metadata/count/hash report generator. It
refuses existing reports and has no network code. The Commons receipt records
the single bounded metadata-only API request separately. Acquisition requires
approval of the exact IDs and limits in `proposed-photo-acquisition.json`.
