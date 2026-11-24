# Modern-room annotation specification v1

Taxonomy ID: `modern-room-seven-v1-chair-includes-seating-stools`. This specifies a
new collection; it does not relabel historical Open Images, COCO, LVIS or ABO
annotations, change old matching definitions, or activate new training targets.

| Category | Include | Exclude or leave unresolved |
|---|---|---|
| Sofa | Upholstered multi-person sofa/loveseat; a visibly connected sectional is one object | Individual cushions/arms, upholstered bed headboards, disconnected modular parts, ambiguous standalone chaise. A deployed sofa-bed is Bed only when its configuration is clear; otherwise ignore |
| Chair | Ordinary dining, office and armchairs; **seating stools and bar stools**, with or without backs | Footstools, ottomans, step stools, wheelchair/device ambiguity, fragments without identifiable seating. Never infer absence when a stool-like form is unclear |
| Table | Whole dining, coffee, side and bedside tables | Kitchen worktops/counters, fixed islands, tabletop fragments without identifiable support, desks. A shared dining/work table stays Table unless evidence establishes a dedicated Desk |
| Bed | Identifiable sleeping bed as one object including its visible frame/mattress; bunk levels are separate only if independently identifiable bed units | Headboards, pillows, bedding-only crops, ambiguous couch/bed configuration. Institutional/medical beds are a separate domain flag, not automatically representative residential examples |
| Desk | Identifiable dedicated working/writing/computer desk, supported by form and room context | Generic tables merely holding a laptop, laboratory workbenches, reception counters, kitchen islands. If purpose cannot be established visually or by trustworthy source context, ignore rather than guessing |
| Cabinet | Whole enclosed storage unit: cupboard, wardrobe, filing cabinet, bounded built-in cabinet unit whose extent is identifiable | Individual doors/drawer fronts/handles, an entire kitchen wall presented as one object, open shelving/bookcases, unclear adjacent-unit boundaries. Preserve native Cabinetry labels separately; never equate them automatically to whole Cabinet |
| Lamp | Whole standalone table, bedside, desk or floor lamp, including connected base/stand/shade | Ceiling pendants, recessed lights, wall sconces, bare bulbs, lampshades and fixture parts. Multiple bulbs on one physical lamp are one object |

## Boxes, visibility and identity

- Canonical boxes are continuous `[left, top, right, bottom]` pixel-edge XYXY in
  the **EXIF-oriented** full-resolution image. The left/top edges are inclusive
  and the right/bottom are exclusive; require `0 <= x1 < x2 <= width` and likewise
  for y. Preserve original compressed bytes and their SHA-256. Decode orientation
  consistently; record oriented dimensions and coordinate-frame version.
- For new annotations, box the visible extent of the object, enclosing its
  observable pixels, including disconnected visible portions around an occluder.
  Do not invent an amodal extent behind a person or another piece of furniture.
  Record `occluded`, `truncated`, `partially_visible` and visibility uncertainty.
  Do not invent a numerical visibility percentage without evidence.
- A clipped object at the image border remains one object if identifiable; clip
  the box to the image and mark truncation. If category/extent is unreliable,
  record an ignore/protected region and reason. Tiny or blurred instances are
  explicit difficult/ignored cases, not silently absent negatives.
- One physical object has one `object_id` per image. Do not create a second object
  for its reflection, cushion or alternate source annotation. Record mirror,
  screen depiction, group-of-objects, fragment and duplicate proposals explicitly.
  Preserve source instance aliases; review COCO/LVIS overlap rather than summing
  their records as independent objects. Multi-object boxes remain unresolved
  until instances can be separated reliably.
- Preserve native dataset name/version, photo/category/annotation IDs, original
  coordinates, masks, native flags and annotation licence. A source box in raw
  EXIF coordinates, normalized XYXY, XYWH or an amodal policy needs a documented,
  tested conversion before activation. The intake import accepts only an envelope
  already bound to the exact image SHA, oriented dimensions and canonical frame;
  it rejects incompatible coordinates instead of guessing.
- Existing reliable source boxes may be reviewed/accepted or corrected directly;
  drawing is optional. Every proposal begins inactive, even if imported with a
  confidence score or a purported prior confirmation. Keep the immutable native
  record and record each correction separately. Native masks are preserved but
  not declared verified merely because a box was accepted.

## Completeness, ignores and negative evidence

For **each image and each of all seven classes**, store one of `unknown`,
`partial`, `complete`, or `reviewed_absent`. Complete and absent require a named
human reviewer, date and concrete evidence describing the actual inspection.
An imported annotation list, model score, caption or LVIS missing category does
not establish completeness. Explicit ignores/difficult regions delimit where
completeness cannot be claimed. A confirmed positive contradicts reviewed absence
for its category and must block qualification.

Training can use supported positives with the rest ignored. A reliable negative
is class-specific: its reviewed region, target class(es), exact photo hash,
taxonomy, evidence and protected-positive intersections must be explicit. Missing
boxes never create whole-image background targets. Source-supported class
completeness can support negatives only where its native taxonomy agrees with
this one. Chairs and seating/bar stools must be protected from Chair-negative
targets, including ambiguous/occluded seating. The intake does not automatically
turn rejected proposals or ignored boxes into negative training regions.

Independent evaluation requires complete seven-category review over the
assessable scene, all unresolved objects accounted for as explicit ignores, and
an independent annotation check before freezing. Report ignored-region/image
coverage and category counts alongside results. If applicable coverage is
insufficient, restrict the claim or defer the evaluation; do not suppress errors
through expanding ignores after seeing model outputs. The current E14 matcher
conservatively ignores unmatched predictions on incomplete images: those images
cannot establish full-scene precision. No matcher is changed by this document.

## Rights, groups and qualification

Record original source/photo ID, photographer account or private pseudonym,
copyright/permission basis, licence URL/version, creator credit, grant conditions,
evidence file/reference/SHA-256 and verification timestamp. Photograph permission
and annotation licensing are separate. Explicit creator grants must cover local
storage, annotation/modification, the assigned ML use, and commercial derived
models. Public redistribution of originals is **not** implied.

Use opaque creator/source, physical-room, capture-session and known related-group
IDs. Build connected components over those relationships, original/modified
hashes, reuploads/crops and established aliases before assigning purpose. All
connected material stays in one purpose, including different photographers in
the same room. Unknown capture identity remains unresolved; different image IDs
or accounts alone do not prove independence. Keep contact details, signatures,
addresses and private EXIF in the private evidence store, outside Git.

The localhost page records explicit submissions and per-object decisions; its
exports are **not activated datasets**. Run the metadata/content checks against
historical exclusion records and other submitted records, then qualify actual
rights, grouping, visual suitability and annotations. Freeze a versioned approved
manifest before model use. Automatically proposed annotations remain inactive
until that process is complete. No new human review is fabricated by this phase.
