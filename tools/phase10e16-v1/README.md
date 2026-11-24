# Exact-three modern-room pilot — acquisition closed

Three approved Commons originals were acquired and remain under the ignored
pilot destination. No substitutions, extra photographs, source retries, model
calls, active boxes, new negative regions or dataset memberships were added.
Do not rerun `acquire16.py`: it rejects an existing destination and a closed
pilot. Source transactions, including HEADs and metadata/licence retrieval,
are individually journaled with TLS measurements and conservative allowances.

## Gallery and isolated E15 intake

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e16-v1/serve16.py --port 8784
```

Open **http://localhost:8784/gallery.html**. Each preview links to the unchanged
original. The gallery shows attribution/licence, assistant observations,
uncertainty and annotation requirements. Previews are local resized derivatives;
they are not additional downloaded photographs. Files, previews, browser profiles
and intake records stay outside public Git.

The linked `/intake.html` serves the unchanged E15 page and script. You can choose
the corresponding local original to inspect its hash and dimensions. Three
inactive, content-verified E15 metadata records are already in `intake/`; their
human confirmations remain false, all category completeness is unknown, and
annotations remain empty. No creator grant, privacy clearance or source-native
box was fabricated to make an eligibility check pass. The validator correctly
keeps them unqualified pending actual evidence.

There are **zero native boxes/masks** for these photos. Existing-source import
and correction logic was tested with synthetic data by the E15 CPU regressions;
real photographs received no invented boxes. Never turn the visual review
descriptions into ground truth or absence labels automatically.

## Browser smoke status

Existing Windows Chrome was tried with GPU disabled, an isolated fresh profile,
background networking/component updates disabled, bounded cache sizes, a
30-second process bound and a monitored 64MiB profile increment reserve. No
installation or personal browser profile was used. Only the fresh profile GUID
was eligible for cleanup of owned browser processes.

The first attempt stopped on a vanished browser temporary during storage
enumeration. A narrow `FileNotFoundError` handling correction was covered by a
regression test; permission errors still fail. The one versioned follow-up
finished with a bounded async DOM wait failure before the first photo hash was
reported. Both results are retained; **browser end-to-end validation is not
claimed**. Neither attempt requested additional source photographs or created
human labels. Local HTTP bodies are separately recorded from source acquisition.

Manually check the page before relying on new annotation exports: load an
approved original, verify displayed SHA/dimensions, observe unconfirmed scopes,
and use a separate synthetic image/envelope for import/edit/undo practice. Do not
confirm actual boxes or permissions merely to complete a smoke test.

## Focused CPU checks and preservation

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tools/phase10e16-v1 -p test_pilot16.py -v
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tools/phase10e15-v1 -p test_intake15.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e16-v1/verify16.py check
```

Network tests use synthetic transport replies only. Current-pilot checks read
the approved acquired originals and metadata; the duplicate comparison decodes
only the existing 59 optimization + 19 already-exposed internal-development
references, never frozen/reserved photographs. Perceptual hashes are heuristics,
not proof of physical-room or global source independence.

No new broad reliability phase was performed. The old E15 local commit was
backed up to its own GitHub branch without changing `main`. This phase's public
commit contains only code, source metadata, manifests and small reports.
