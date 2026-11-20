Phase10C.4 is a local, training-only human rectangle editor. There is no model,
training, GPU, acquisition or production-deployment endpoint.

Start from the project root with the existing virtual environment:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -u -B tools/phase10c4-v1/server.py --port 8766
```

Open http://localhost:8766/ in the Windows browser. The prepared server is left
running at handoff. Do not start a second server on an occupied port. Stop your
foreground instance with Ctrl+C; browser drafts and versioned exports persist.

1. Choose a training photo. The nine earlier scope-review images are starred.
   All59 photos are existing optimization members. Blue boxes are native
   positives, not detector predictions. Grey areas are restricted/ignored.
2. In **Chair-negative candidate** mode, drag a rectangle in inspected content
   containing no Chair or seating stool/bar stool, including partial or
   ambiguous seating. Footstools/ottomans/step stools are separate concepts.
3. Use **Select / move / resize** or click a listed rectangle to select it.
   Drag its body/corner handles or enter original integer pixel-edge XYXY
   coordinates. Zoom and scroll for careful inspection. Protected overlaps
   cannot be applied. Edits revoke previous confirmation.
4. For possible unannotated/occluded seating, switch to **Potential seating /
   uncertainty** and mark that area. It is never a negative; overlapping prior
   confirmations are revoked. Unknown content elsewhere stays ignored.
5. Set reviewer/date, explain the exact rectangle, check all three confirmation
   boxes, and click **Confirm selected negative**. Nothing is preapproved.
6. Use reject/remove/undo as needed. Undo cannot newly restore approval of a
   changed rectangle. Evidence edits also revoke confirmation.
7. Aim for at least six individually confirmed, nonoverlapping rectangles across
   three or more distinct known source groups. Multiple photos from one group
   count as one. This gate never authorizes training.
8. Click **Validate and export a new version**. The server validates and stores
   an immutable workstation version, downloads JSON to the browser, creates a
   separate additive supervision manifest, and calculates a CPU fixture mask
   comparison. Draft/uncertain/rejected rectangles are excluded from masks.
   Partial progress can be exported; its counts and unmet gates remain clear.

Exports are local under
`data/open-images/preparation/phase10c4-v1/exports/review-UTC-UUID/`.
They contain `human-regions.json`, `validation.json`, proposed trusted evidence,
`supervision-manifest.json`, `mask-comparison.json` and SHA256 sums. Original
photos, native annotations, reviewed submissions and production defaults are
not changed. Restore the downloaded JSON with the file input; identity, source,
taxonomy and coordinates are revalidated. Browser drafts are scoped to the
exact registry. Export regularly as browser storage is not an independent backup.

The24 clarification items are on a separate tab: three concern training-scope
interpretation,21 are validation diagnostics. All original decisions/evidence
remain read-only. A response or defer action requires an explicit click and
reviewer/date; it does not approve any rectangle or move an evaluation image
into training. The tool never serves those validation photographs.

Coordinates are integer edges in the original stored image, half-open XYXY.
SVG screen-to-image transforms account for CSS resizing, zoom, scrolling and
letterboxing. EXIF1–8 mappings are reversible; display orientation is applied
only to served PNG previews. Source byte hashes/dimensions/orientation are
checked, and all59 current photos have orientation1. Native positives and
restriction overlap tests use original coordinates. Positive/ignored overlaps,
uncertainty and overlapping candidates block confirmed-negative acceptance.
The exact full-image rectangle is disallowed in this regional workflow.

CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c4-v1 -p test_regions.py -v
```

`browser_smoke.py` uses an already installed Windows Chrome with GPU disabled,
a dedicated temporary test profile and a synthetic registry on localhost8767.
It neither confirms real photos nor writes real human exports. Synthetic pointer
events stub pointer capture only in the test harness; production uses native
pointer capture. Real Windows Chrome rendering and loopback access were also
checked without any annotation action. No browser package was downloaded.

`prepare.py` is an exclusive preparation step, already completed. Preserve old
versions; do not delete them to rerun. Never copy fixtures into human evidence.
After genuine exports, nonzero CPU fixture mask terms are necessary but do not
prove actual model-query coverage or accuracy. A separately approved zero-update
GPU smoke, controlled A/B approval and independent evaluation protocol remain
required. Phase10D box-only training stays unauthorized.
