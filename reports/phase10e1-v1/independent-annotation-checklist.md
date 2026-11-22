# Independent room annotation requirements

No photographs or annotations are accepted by this checklist alone. Current
complete independent image and instance counts remain zero.

Use a separate source-disjoint collection, not current optimization/development
photos or automatic reuse of reserved material. Preserve individual rights,
creator, room/capture-session IDs, source hashes and original pixel dimensions.
Reject uncertain group independence. Freeze the source list before model scoring.

Before annotation, freeze whole-object category definitions:

- Sofa: identify the complete seating unit; mixed sofa/chair sets need separate
  object boxes, not one furniture-set target.
- Chair: include ordinary seating chairs, seating stools and bar stools. Keep
  footstools, ottomans and step stools separate; ambiguous seating is ignored.
- Table and Desk: preserve original native terms; adjudicate work desks versus
  dining/coffee/side tables explicitly. Do not fix historical labels to improve
  a detector's apparent results.
- Bed: annotate the supported whole bed, recording visibility and truncation;
  headboards, rails, mattresses or bedding alone are not whole-bed evidence.
- Cabinet: define the storage unit and treatment of built-in units before intake.
  Drawer/cabinet fronts and a kitchen-wide Cabinetry box are not automatically
  equivalent to a whole cabinet.
- Lamp: distinguish standalone floor/table/desk lamps from ceiling fixtures,
  chandeliers, lampshades and bulbs. Keep semantically uncertain units ignored.

Record each box in validated original-image coordinates, native label/source
where applicable, project label, instance ID, occlusion, truncation, crowd/mixed
unit status and explicit ignore reason. Preserve masks as native evidence; do
not manufacture a mask from a bounding box or invent occluded extents.

Each room needs an explicit completeness review for every project category,
including possible missed furniture. Missing native COCO/LVIS/Open Images
annotations do not establish absence. Reviewers must inspect independently of
model predictions, record their own evidence/name/date and adjudicate uncertain
units. Double-review at least 20% of boxes and independently check every image's
completeness. No automated human-review flags are generated.

Freeze annotation/protocol hashes and eligibility before original versus
preselected-model scoring. Maintain a separate calibration collection for any
threshold choice. Evaluation examples and their annotations must never create
training masks or select the distillation coefficient.
