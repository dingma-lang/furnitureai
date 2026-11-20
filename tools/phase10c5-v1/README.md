# Phase 10C.5 — local object review and inactive hard-negative preparation

Use the project virtual environment. No model constructors, GPU inference,
training launcher, downloads or environment changes are required.

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B tools/phase10c5-v1/g5_server.py --port 8768
```

Open <http://localhost:8768/> in Windows. Set reviewer/date once. Review the
highlighted object with **1 Correct Chair**, **2 Other object / Not Chair**,
**3 Missing annotation**, **4 Uncertain**. **N/P** or arrows move between objects;
**Undo** or Ctrl+Z restores the last action. Labels are native dataset evidence,
not new model predictions. Diagnostic scope shows saved predictions and cannot
produce training negatives. Previous reviews are context, never auto-converted.

When its full box visibly contains no Chair, seating/bar stool, ambiguous or
partial seating, use **I confirm this exact box is Chair-free**. This is a
separate explicit absence decision. Person/Plant labels alone do not prove it.
Known seating and ignored areas block approval. The rest of the image stays
unknown. Footstools, ottomans and step stools remain separate protected objects.

The ordinary Other action explicitly confirms the displayed native object label
where applicable, without claiming Chair absence. Optional notes preserve your
words. Otherwise evidence records the actual button action and its source task;
it never fabricates a visual narrative. No decision is pre-filled.

Use **Draw or correct a box only when needed** for corrections or missing objects.
Drawing uses original pixel edges with EXIF/display coordinate transforms; it
revokes the old box's confirmation. Corrected positives remain inactive proposals.
Use category toggles and Zoom/Open full photograph for dense or small objects.

**Check**, then **Save versioned review**. Server exports live under the ignored
`data/open-images/preparation/phase10c5-v1/exports/review-<UTC>-<random>/`:
human decisions, native provenance, validation, additive supervision, CPU-only
mask fixture comparison and SHA-256 seals. The browser also downloads the same
review JSON. Restore that JSON to continue. No source annotations are overwritten;
browser drafts are convenience state, not independent backups.

No training activation follows export. The initial overlay has zero reviewed
negatives and zero changed mask terms. Six accepted regions in at least three
optimization source groups are the current minimum pilot gate, followed by
actual-query contrast verification in a separately authorized zero-update smoke.

CPU and synthetic browser tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c5-v1/g5_verify.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c5-v1/g5_browser_test.py
```

The browser test uses already-installed Windows Chrome, synthetic photos, an
isolated profile and in-memory fake export response. It does not create human
decisions about real photographs. `g5_objective.py` contains isolated CPU-tested
research components, not an enabled training pipeline. The primary loss A/B and
later auxiliary/residual ablations are deliberately separate.

Keep the original C.4 server on port 8766 available if needed. Stop this server
with Ctrl+C (or its recorded process PID); restart the command and restore an
export. Historical code/data/weights remain untouched. Code-only rollback is
switching to `experiment/phase10c4-visual-negative-regions`; verify the branch name
in Git before switching. Never delete local exports or restore dataset/model files.
