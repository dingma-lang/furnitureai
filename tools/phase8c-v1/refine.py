"""Explicit assistant observations after viewing all27 originals and unchanged overlays.

No human flags, source annotations or category IDs are edited. Unselected targets
are unresolved/ignored, never background; retained positives remain native/provisional.
"""
import collections as C
from pilot_core import *

# Exact source annotation allowlists, not inferred or newly drawn labels.
DIAGNOSTICS={
 452459:([708231,708232],'Workstation and chair identifiable. Wall/arm-mounted reading light is not admitted as a portable whole lamp.'),
 192403:([247156],'Bed identifiable with border truncation. Thin chrome/reading-light regions are too ambiguous to confirm as portable lamp units.'),
 39844:([],'Dark scene, people and severe furniture occlusion prevent confident sofa/lamp unit and mask assessment.'),
 305093:([1033799,1033800,1033802],'Sofa and two standing lighting units identifiable; one lamp is edge-truncated. Preserve visible source extents and partial-label scope.'),
 484123:([954106,954107,954108],'Bed and two bedside table lamps identifiable; original polygons broadly register to visible units.'),
 516834:([952952],'Articulated task lamp identifiable, with base occlusion. Desk annotation shows only a foreground support strip, insufficient whole-unit context.'),
 22890:([517677,517678],'Two beds identifiable in dim indoor scene. Three lights are fixed wall/headboard fixtures; not admitted as portable whole lamps.'),
 287886:([338815,338816],'Bed and bedside table lamp identifiable. Border-cropped side furniture labelled Table has unresolved nightstand/table unit semantics.'),
 296897:([39981],'Bed identifiable. Two small headboard-mounted reading fixtures remain outside the retained portable-lamp unit scope.'),
 161047:([],'Kitchen cabinet annotations mix individual doors, fronts, bays and a base run. Range-hood light is not a whole portable lamp. All source units unresolved.'),
 469191:([773660,773661,773663],'Bed and two bedside table lamps identifiable; one lamp and bed reach image border. Source visible-mask bounds preserved.'),
 277630:([715632,715633,715635],'Bed and two physical table lamps identifiable. Two fixed sconces and two reflected lamp regions remain unresolved/excluded; reflections are not extra physical units.'),
 420490:([],'Two cabinet-door/front masks and three separate shades of one chandelier are not verified whole Cabinet/portable Lamp objects.'),
 397362:([663079,663082,663083],'Central two-seat sofa and two standing lighting units identifiable. Right single-seat armchair-like region labelled Sofa remains semantically unresolved.'),
 469638:([],'Bathroom medicine/vanity fronts and doors are not established whole cabinet units; fixed vanity-light fixtures are not portable lamps.'),
 104737:([736435],'Standing lamp identifiable. Cat-occupied single-seat armchair-like region labelled Sofa remains unresolved; preserve native source label without correction.'),
 273855:([],'People/foreground obscure kitchen storage fronts and nearly all chair structure. Cabinet units and heavily occluded chair are unresolved.'),
 262727:([],'Dominant foreground cat and shallow focus obscure cabinet fronts/drawers and hanging lamp. Unit identity and fine registration cannot be reliably established.'),
 397877:([371828],'Physical bedside table lamp identifiable. Second small Lamp region appears reflected; heavily people-occluded border-cropped bed remains unresolved.'),
 238630:([436755,436756,436768],'Two beds and central bedside table lamp identifiable; foreground bed is border-truncated. Visible source mask units preserved.'),
 269386:([755469,755470,755471],'Bed, central standing lamp and bedside table lamp identifiable. Extreme left clipped/occluded lighting unit excluded for incomplete unit evidence.'),
 257853:([],'Canopy-bed source mask omits most visible canopy/frame and tiny chair is substantially occluded. Whole-object unit extent remains unresolved; no geometry repair.'),
 265843:([665976],'Small bedside task lamp identifiable. Ceiling chandelier is fixed lighting; Bed source target is a narrow bottom crop with incomplete unit extent.'),
 330770:([],'Soft/noisy canopy-bed close-up and crate-like object labelled Table do not support reliable unit/category/whole-object registration confirmation.'),
 78760:([1089723,1089724,1089726],'Bed and two bedside table lamps identifiable. Two fixed wall sconces excluded from portable-unit supervision.'),
 122787:([207613,207614],'Sofa and partially visible bed through the open doorway identifiable; no mirror substitution. Preserve occlusion and incomplete seven-class coverage.'),
 483028:([1186989],'Bed through doorway identifiable. Sofa annotation includes a separate foreground ottoman/footstool, leaving multi-object unit semantics unresolved.'),
}
TARGETED={192403,22890,122787,483028,516834,397362}

def counts(rows):
 result={}
 for cid,name in CLASSES.items():
  present=[r for r in rows if any(a['category_id']==cid and a['id'] in r['retained_annotation_ids'] for a in r['original_annotations'])]
  annotations=[a for r in present for a in r['original_annotations'] if a['category_id']==cid and a['id'] in r['retained_annotation_ids']]
  result[name]=dict(native_id=cid,images=len(present),boxes=len(annotations),masks=sum(bool(a.get('segmentation')) for a in annotations))
 return result

def run():
 source=read(OUT/'screening-v1.json');rows=source['images']
 if set(DIAGNOSTICS)!={r['coco_image_id'] for r in rows}:raise ValueError('Uninspected or extra image')
 decisions=[]
 for r in rows:
  retained,note=DIAGNOSTICS[r['coco_image_id']];ids={a['id'] for a in r['original_annotations']}
  if not set(retained)<=ids or not set(retained)<=set(r['valid_core_annotation_ids']):raise ValueError('Retained annotation not valid native source target')
  # A failed automatic quality gate cannot be waived by these visual observations.
  if r['flags']:retained=[]
  r.update(retained_annotation_ids=retained,valid_core_annotation_ids=retained,
   unresolved_annotation_ids=sorted(ids-set(retained)),status='provisional_source_candidate' if retained else 'quarantined',
   human_verified=False,independently_verified_registration=False,scene_verified=False,shared_project_unit_verified=False,
   assistant_diagnostic=dict(method='All27 original/unchanged-overlay paired views; six targeted original or overlay inspections',
    checked_utc=now(),human_reviewed=False,reviewer_kind='assistant',targeted_original_or_overlay=r['coco_image_id'] in TARGETED,observations=note),
   no_training_authorized=True,unresolved_regions_must_not_be_background=True)
  derived_scope={}
  for cid in CLASSES:
   native={a['id'] for a in r['original_annotations'] if a['category_id']==cid};kept=native&set(retained)
   derived_scope[str(cid)]=dict(native_source_ids=sorted(native),retained_ids=sorted(kept),all_native_targets_retained=bool(native) and kept==native,
    source_not_exhaustive=cid in r['source_image_scope']['not_exhaustive_category_ids'],
    derived_positive_nonexhaustive=bool(kept) and (kept!=native or cid in r['source_image_scope']['not_exhaustive_category_ids']),
    background_or_unmatched_negative_allowed=False,shared_project_mapping_confirmed=False)
  r['derived_training_scope']=derived_scope
  for a in r['original_annotations']:
   decisions.append(dict(coco_image_id=r['coco_image_id'],annotation_id=a['id'],native_category_id=a['category_id'],
    status='provisional_native_positive' if a['id'] in retained else 'unresolved_inactive',
    observation=note,geometry_modified=False,human_reviewed=False,independent_semantic_verification=False))
 eligible=[r for r in rows if r['status']=='provisional_source_candidate']
 commit(OUT/'screening-v2.json',dict(source,images=rows,status_counts=dict(C.Counter(r['status'] for r in rows)),
  eligible_partition_counts=dict(C.Counter(r['proposed_split'] for r in eligible)),
  eligible_category_counts={p:counts([r for r in eligible if r['proposed_split']==p]) for p in ('train','development')},
  source_native_positives_only=True,whole_seven_class_training_not_approved=True,unresolved_targets_never_negatives=True,
  independent_human_review=False,source_geometry_and_category_ids_modified=False))
 commit(OUT/'annotation-decisions.json',dict(records=decisions,all99_source_annotations_preserved=True,
  decision_counts=dict(C.Counter(r['status'] for r in decisions)),previous33_inactive_annotations_activated=0))
 commit(OUT/'eligible-candidates-v1.json',dict(images=eligible,training_approved=False,
  status='Prospective native positives after assistant screening; not human-reviewed or a frozen training lock',
  unresolved_annotations_masked=True,screening_sha256=sha(OUT/'screening-v2.json')))
 print('REFINED',dict(C.Counter(r['status'] for r in rows)),dict(C.Counter(r['proposed_split'] for r in eligible)),
  'native decisions',dict(C.Counter(r['status'] for r in decisions)),flush=True)

if __name__=='__main__':run()
