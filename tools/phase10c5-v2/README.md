# Phase 10C.5 follow-up — automatic review progression and category masks

This version builds on commit `85c3cc5` without changing the C.4/C.5-v1 code,
registry, source photographs, models or experiments. Use the project interpreter:

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B tools/phase10c5-v2/server2.py --port 8769
```

Open <http://localhost:8769/>. Enter reviewer/date once. Use **1 Correct Chair**,
**2 Other object / Not Chair**, **3 Missing annotation**, **4 Uncertain**. Each
explicit decision advances to the next unreviewed object. Uncheck **Advance after
a decision** to inspect or combine actions on one object. Keyboard N/P and arrows
navigate; Undo/Ctrl+Z returns to the edited object even after automatic advance.
Completed/remaining counts refer to object decisions, including uncertain ones;
they are not accepted-training-label counts. The final item does not wrap into a
reviewed object. With Unreviewed only enabled, completion leaves no active item.

The permanent legend and toggles distinguish source furniture/non-furniture,
ignored regions and saved model predictions. Diagnostic predictions can be
reviewed but never enter training. Known source categories are suggestions,
not pre-filled human decisions. Other/Not Chair confirms an applicable native
object label, without asserting Chair-free surroundings.

For a trustworthy absence region use **I confirm this exact box is Chair-free**.
Chair includes ordinary seating chairs/stools/bar stools, including ambiguous or
partial seating. Footstools/ottomans/step stools remain separate protected objects.
The explicit absence action stands on its own; no preliminary Other click or
repetitive checkboxes are required. Protected/uncertain overlaps block eligibility.
The remainder of the photograph stays ignored for new negative supervision.

Optional drawing/correction revokes the old box's review. Unreviewed draft boxes
now survive export/restore. New positive annotations remain inactive proposals.
Edit history records source hashes, timestamps, before/after values and Undo as
a reversal, without deleting the original action. Server validation replays the
history and rejects decisions that disagree with it.

**Check**, then **Save versioned review**. The browser downloads the JSON and the
server creates a new ignored folder under:
`data/open-images/preparation/phase10c5-v2/exports/review-<UTC>-<random>/`.
Files include human decisions/history, validation, an inactive supervision
manifest, SHA-256 records and an atomic completion marker written last. Restore
the v2 JSON to continue. v1 reviews remain supported by the untouched v1 interface
on port8768; this version does not silently migrate their schema or flags.

`dataset2.compile_dataset()` preserves all source positives and ignored boxes,
native category/annotation provenance, rights evidence, and exact source groups.
It separates named auxiliary positives from category-specific Chair negatives.
It refuses blocked decisions and train/development identity/hash/group overlap.
`category_masks()` is a CPU-tested additive contract: unknowns have zero new
weight, trusted regions need full query containment, known seating/uncertain
regions are protected, and supplied legacy targets/masks remain intact. This
component is not wired to a GPU runner and never changes production loss.

No training or inference is authorized. The initial 59-photo/56-group/182-box
overlay contains zero new accepted negatives. Native metadata candidates are not
human-confirmed absence. The prior bounded A/C training proposal remains inactive;
reviewed evidence, actual-query mask contrast and separate approval are required.

Tests use synthetic review fixtures and the already-installed Windows Chrome:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover -s tools/phase10c5-v2 -p 'test*.py' -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c5-v2/browser_test.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c5-v2/verify2.py
```

The comprehensive verification creates immutable reports once; use a new output
version before repeating it. Browser fixtures write only synthetic test state,
never human evidence for real photographs. Stop only the recorded v2 server PID
or Ctrl+C. All older services and exports remain available for rollback.
