# Phase 5 approved photographs — acquisition and screening only

Project interpreter: `.venv/bin/python -B`. Data/artifacts: `data/open-images/image-pilots/phase5-v1/`. The destination was absent initially; it now contains a completed, versioned acquisition. A completion marker blocks all network/budget reopening; only read-only tests/inspection should run after completion. Do not run the fresh probe again or create replacement photographs.

The frozen manifest SHA-256 is `b07ee0e045a7d005d26778b8c27e76bbaa78129dd05e113540efe28094e3e131`. Only its 150 IDs are eligible for requests; only 118 current individual Flickr pages confirmed the matching photo's CC BY 2.0 licence/author. The other 32 IDs remain unacquired licensing quarantines, with original partition slots retained. Two current source pages pointed at non-CC licensing help pages, one at BY-NC 2.0 and one at BY-NC-ND 2.0; 28 returned 403/404. Historic dataset licence metadata was not used to override this result.

`network.py` is a durable response-body budget: 100,000,000 bytes including compressed licensing/error responses and photographs, 3,145,728 bytes/image, at most two image attempts, 512 MiB allocated storage. Requests reserve bytes before reads; uncertain failed transfers remain conservatively charged. It does not follow redirects or execute/fetch page scripts, embedded photos, CSS, fonts or other resources. Individual JSON-LD matching a specific source photo ID and author account is required; a generic Creative Commons string or licence on another image cannot pass. The gzip decoder handles case-insensitive headers and bounds decoded HTML. All saved response bodies have SHA-256 records.

`licensing.py --resume` uses matching cached evidence only; `acquire_photos.py` revalidates it and exact CVDF HEAD sizes/ETags against the approved quote before image GETs. Completed JPEGs are reused only with matching URL/SHA. Changed sources, integrity failures and resource failures stop. Each final JPEG's source MD5/ETag, SHA-256, decoding, dimensions and EXIF are recorded under `integrity/`. A transport failure may consume the second image attempt in a separate retained path; no existing response/image is overwritten. A stage with an incomplete local response stops for audit rather than silently repairing or deleting it.

`screen.py` computes exact file/pixel hashes, dHash and a defined 63-bit DCT perceptual signature. Similarity thresholds (dHash <=4 or DCT hash <=6) are unfitted conservative diagnostics. No similar listings/products are merged. Raw source annotations, category IDs, attributes and positive/negative image labels remain unchanged. Hierarchy descendants, overlapping different native categories and low-detail heuristics create unresolved records; they do not become corrected ground truth. No class lacking published image-label coverage is invented as a negative.

`finalize.py` incorporates explicit assistant observations from eight contact sheets and three targeted full-resolution photographs. It never marks human review or scene verification true. Five clear product/composite images are quarantined as unsuitable for a room pilot. Fifty acquired photographs remain unresolved, and 63 are provisional candidates. Provisional is not independently validated. The immutable initial screen is retained; the authoritative final screen is `screening-final-v1/`.

`verify.py` checks original box/membership equality, image/response hashes, all existing Phase 1–4/model preservation records, prior Phase 5 metadata/artifact hashes and stored historical image/pixel signatures. It does not open official heldout photos or feature arrays. See `verification.json` for exact coverage and residual identity limits. Current exact-pixel checks cover 2,111 historical image metadata records; different-resolution near-duplicates against past ABO images are not exhaustively excluded.

Tests (offline, no model training/inference):

```bash
.venv/bin/python -B -m unittest discover -s tools/phase5-images-v1 -p 'test_*.py' -v
```

Do not rerun completed screen/finalizer/verification commands into the same output paths; they require fresh artifacts. All authorised work is complete. No training or detector scoring was run. Follow `baseline-instructions.md` in the artifact directory for the separately approvable next stage; its future bundle/commands deliberately cannot bypass existing evaluator guards.
