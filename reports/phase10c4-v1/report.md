Furniture AI Phase10C.4 — local Chair-negative rectangle annotation

The working interface is http://localhost:8766/. It was tested from Windows
Chrome and is left running on127.0.0.1 only. Start/stop/drawing/export instructions
are in `tools/phase10c4-v1/README.md`. No training, model inference, new dataset,
model download, production change or source-annotation edit occurred.

The registry reuses the pinned Phase10B optimization manifest and Phase10C.3
taxonomy/evidence template. It includes59 existing training photographs and56
known source groups; the nine prior scope-review images appear first. Internal
validation, reserved and frozen images cannot be selected or served. Every photo
is tied to source ID, byte SHA256, dimensions, EXIF orientation, group and native
annotation digest. The registry has no human rectangles or confirmations prefilled.

Chair includes ordinary chairs, seating stools and bar stools. Footstools,
ottomans and step stools remain separate, without automatically labeling them
positive or negative. Blue native-positive boxes and grey ignored/restricted
regions are overlaid. The user can draw negative candidates, move them, resize
corners, edit exact pixel coordinates, reject/remove them, undo changes, or mark
potential unannotated/occluded/ambiguous seating as uncertainty. The entire photo
never becomes negative merely because one rectangle is approved.

Each negative rectangle requires reviewer, date, specific explanation, explicit
taxonomy/absence/confirmation checkboxes and a confirmation button action.
Geometry revisions and evidence changes revoke approval. Undo cannot create a
new approval for a changed/restored rectangle. Adding intersecting potential
seating revokes an existing negative confirmation. Uncertainty elsewhere and
outside accepted rectangles stays ignored. Source images and annotations are
read-only; the browser and server store a separate annotation layer.

Drawing uses SVG's inverse screen transform, accounting for actual CSS size,
zoom, page/frame scroll and aspect ratio. Raw pixel-edge XYXY coordinates are
outward-rounded to integer original-image edges, not guessed from screenshot
dimensions. EXIF1–8 transforms are implemented and tested in both directions;
all59 current photographs have EXIF1. Full-resolution oriented PNGs are streamed
from hash-verified originals without changing them. Normalized loss coordinates
are derived server-side from original integer pixels and dimensions.

The backend independently rejects wrong registry/taxonomy, forged photo/group/
dimension/orientation/annotation bindings, fractional/out-of-frame coordinates,
stale geometry confirmations, duplicate IDs and non-training photo keys. Known
Chair/seating positives, ignored native regions, potential/uncertain regions and
overlapping candidate rectangles prevent acceptance of an intersecting negative.
Touching rectangle edges do not share pixels. Whole-image absence needs a
separate completeness review and is not accepted through this regional tool.
Host/origin/session checks restrict writes to the local review session; there
is no filesystem-browser, model, inference or training route.

Each export is a fresh timestamp/UUID version. It saves the full human session,
validation counts, accepted proposed evidence, an additive supervision manifest,
a CPU mask comparison and checksums. The browser downloads the JSON and the
workstation retains an immutable copy under
`data/open-images/preparation/phase10c4-v1/exports/`. Restoring JSON revalidates
source/registry/taxonomy/coordinates. Partial draft exports are allowed, but only
individually confirmed, nonoverlapping, valid negatives enter the supervision
overlay. The target is at least six confirmed rectangles across at least three
distinct known source groups. Reaching it never enables training automatically.

At implementation handoff there are **zero confirmed human negative rectangles,
zero confirmed source groups and zero actual human exports**. The initial CPU
mask comparison has zero changed terms, as expected with no evidence. A synthetic
integration fixture verifies that six accepted rectangles on three synthetic
photos create safe, nonzero added Chair-negative terms, preserve positives,
retain regional rather than whole-image masks, and refuse overwriting a version.
Those confirmations exist only in temporary synthetic test data, not real
photographs or the human output directory.

When actual reviews are exported, accepted rectangles are grouped by source photo
while retaining every individual confirmation. The existing taxonomy-aware C.3
evidence validator and unchanged C.2 negative-loss adapter are reused. Only new
Chair-negative masks are added; positive/native labels, original partitions and
other source-native negative rules remain unchanged. Any model query must be
fully contained within trusted rectangles and avoid protected positives and
uncertainty. The CPU comparison uses coordinate fixtures, not image embeddings
or detector inference. Nonzero fixture terms do not prove actual future model
query coverage; that needs a separately approved zero-update GPU smoke.

All24 earlier clarification items are clearly presented on a separate tab with
their original decision, evidence and questions. Three relate to training-scope
interpretation;21 are validation diagnostics. The latter have no image route in
this training annotation server and cannot create training negatives. Responses
or defer decisions are separately recorded after explicit user action. Nothing
is silently corrected or marked resolved. Local rectangle approval does not
certify whole-image completeness, resolve all earlier semantics, or relabel
validation boxes.

The independent real-room evaluation proposal is separate in
`independent-evaluation-proposal.json`:120 new evaluation-only photos/source
groups are proposed, with category-support targets and complete whole-object
annotations, including stools under Chair. These are planning targets, not
verified available image counts or a download manifest. It requires separate
rights/acquisition/annotation approval. Exact training/development/exposed/frozen
identity, creator/family/hash and perceptual duplicates must be excluded before
membership freezes. Evaluation images/reviews can never become this tool's
training masks. Native class/unit limitations and per-category completeness must
be explicit. Original/A/B checkpoints, thresholds/preprocessing and endpoints
freeze before a single approved evaluation; no tuning on that set. Precision,
recall, AP50/AP75, localization, duplicate/fragment errors and source-group
uncertainty are reported. The80%precision/90%recall goal remains unestablished.

The bounded A/B recommendation remains conditional: original OWLv2 initialization
for both arms; same59 optimization photos,182 positive boxes, seed20261118,
LR1e-5, decay0.01, FP32, batch1/accumulation4, five epochs/75updates per arm,
150updates total,60minutes,6.5GiB allocated VRAM and4GiB new experiment storage.
Only trusted negative-mask handling differs. First require six regions/three
groups, valid evidence and a nonzero safe mask difference; then settle the
independent evaluation plan and seek separate GPU smoke/training approval.
Actual per-image predictions must be atomically saved before isolated CPU metrics.
No box-only or joint-head training is started by annotation, validation or export.

Validation passed224 CPU tests, including26 new geometry, evidence, HTTP security,
source-isolation and synthetic export/mask tests. Browser testing passed27 DOM/
coordinate/confirmation-flow checks using synthetic images, an installed Chrome
with GPU disabled, a separate test profile and intercepted downloads. The fixture
stubs pointer capture for synthetic DOM events only; production uses native
browser capture. The actual Windows localhost page and full-resolution training
photo rendered successfully with no annotation actions. No browser or package
installation occurred. Existing9,973 dataset/model files,160 Phase10B snapshots
and prior prediction, review and report seals passed preservation checks.

Only code/tests, the independent-evaluation proposal and small summary reports
are committed. Photo bodies, the full registry, previous human evidence,
annotation exports, browser screenshots/profiles and dataset manifests remain
local and ignored by Git. Production defaults, historical weights and the Mac
simulation application are unchanged.
