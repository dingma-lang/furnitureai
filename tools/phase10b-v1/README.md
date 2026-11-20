# Phase 10B: recovery and continuous diagnostic training preparation

Use `/home/tristary/furniture-ai-ml/.venv/bin/python -B`.
Raw manifests, original-source overlays, rights joins, historical metadata and
local reviews live under `data/open-images/preparation/phase10b-v1/` and remain
ignored by Git. Small aggregate reports and source are committed separately.

The attached request ends abruptly at “2. Full model”; its complete preceding
steps1–7 define this implementation. No training permission is inferred.

## Annotation recovery

All130 inactive annotations have unchanged source boxes, polygon overlays where
available, full context and enlarged crops. Assistant triage is explicitly not
human review.17 apparent complete-object candidates are priorities for optional
review, not new training examples. All130 and the separate33 LVIS overlays stay
inactive. The new eligibility version preserves78photos/75known groups/243boxes,
with59optimization and19training-derived validation photographs.

Start the optional review UI, bound only to loopback:

```bash
.venv/bin/python -u -B tools/phase10b-v1/review_server.py --port 8786
```

Open http://127.0.0.1:8786 in the Windows browser. Review the full photograph and
overlay, choose a decision, enter reviewer and evidence, and explicitly check
confirmation. Each submission creates a unique new JSON under `review/submissions/`.
No prior decision or annotation is overwritten, no missing evidence is generated,
and no submission activates a training target. A later eligibility audit must
approve any mapping/target changes in another manifest version. The three-class
diagnostics do not require completing all130 reviews.

## Prepared worker

`worker.py` is a genuine CUDA execution implementation, but was not run on the
GPU in this phase. Its authorization gate executes before model imports, image
access or experiment-directory creation. Approval is required separately for
the exact manifest/config/source hashes. Do not turn the proposal's Boolean into
an approval; an approval record must refer to the user's subsequent instruction.
No approval record is supplied by Phase10B.

The fixed queue uses original OWLv2 initialization, LR1e-5 and3e-5, class/box
heads only, frozen eval-mode backbone/objectness, batch1/accumulation4,
five epochs,75expected updates per fit,150maximum per fit/300total,
30minutes per fit/one hour queue,6.5GiB allocated-memory target. Missing,
nonexhaustive, filtered and unresolved labels remain unknown. Source masks are
retained, not trained by OWLv2's box head. Shared head updates can affect all
outputs, but four unsupported categories receive no positive/negative labels.

The original and Phase5C models will be compared on the same19 internal photos
before fitting. All selection uses training-derived validation and matched-recall
criteria with per-category localization/recall guards. No18/36 exposed panel or
reserved photo loader exists. Results remain provisional source-native agreement.

Checkpoint snapshots include heads, optimizer, scaler, Python/Torch/CUDA RNG,
epoch/group cursor, best candidate and binding. Durable call reservation precedes
the optimizer; commitment follows an immutable checksum-verified snapshot. A
crash with uncertain call completion blocks automatic replay. Thirty-second time
slices are precharged and never refunded on crashes. Sequential models, finite
gradient checks, OOM propagation and disk/VRAM guards are implemented. No silent
CPU fallback, retries, resolution changes, hyperparameter expansion or cleanup.

Every update snapshot is preserved. Expect about3GiB for the planned150total
updates, with an8GiB queue storage cap. This is future allocated storage, not used
by preparation. Estimated GPU runtime5–15minutes/config and2–4GiB allocated VRAM
remain estimates until a separately approved smoke test. Full CUDA/AMP behavior
has not been executed by CPU tests.

After separate approval and creation of an exact bound authorization record:

```bash
# Future commands only; not authorized to execute during Phase10B.
# The first stage verifies GPU forward/backward without any optimizer calls.
.venv/bin/python -u -B tools/phase10b-v1/worker.py --authorization data/open-images/preparation/phase10b-v1/approvals/training-v1.json --smoke-only
# Resume the exact unchanged approved queue, skipping completed fits:
.venv/bin/python -u -B tools/phase10b-v1/worker.py --authorization data/open-images/preparation/phase10b-v1/approvals/training-v1.json --resume
```

For background operation after approval, launch the same command with `nohup`,
redirecting into a fresh timestamped log under the preparation directory. Keep
the PID. SIGINT/SIGTERM or `touch data/open-images/experiments/phase10b-heads-v1/STOP`
requests a graceful stop; it never authorizes exceeding the queue budget. Do not
delete STOP/ledger/checkpoints to force a completed or uncertain run to restart.
Check `ledger.json`, per-fit `complete.json`, validation JSON and append-only
`history/` records. OOM/nonfinite failures propagate and preserve previous receipts.

## Reproducible preparation checks

```bash
.venv/bin/python -B -m unittest discover -s tools/phase10b-v1 -p 'test_*.py' -v
```

`dry_run.py` completed a fixture-only2x75-call bookkeeping rehearsal including
interrupt/resume. It refuses to overwrite that rehearsal. No optimizer or model
was constructed. Historical model/dataset byte hashes and Phase10A source/data
seals are independently checked at closeout.

`tracking.py` defines immutable, scope-labelled events. `history.py` inventories
historical cohorts/model families, retains actual per-run metadata and classifies
storage. Missing metrics are null/not evaluated, never synthesized. Historical
and conditional space estimates are proposals only: no data was deleted/moved.

GitHub backs up code, not ignored photos, weights, evidence or feature caches.
Return to Phase10A with `git switch experiment/phase10a-expanded-preflight` or
baseline with `git switch main` after saving work. Models/defaults never changed.
