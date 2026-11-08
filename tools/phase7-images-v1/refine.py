"""Assistant-only bounded source/photo diagnostics; never human confirmation or relabeling."""
import collections as C
from pilot_core import *

QUARANTINE={
 452229:('occlusion_and_unit_uncertainty','People heavily occlude background furniture; the native Desk units cannot be resolved confidently. No chair/sideboard correction is invented.'),
 568281:('native_unit_uncertainty','Desk polygon surrounds a distant closed wooden storage/work unit behind a chair; Desk versus cabinet/sideboard semantics remain unresolved.'),
 499408:('low_illumination_visibility','Dark foreground and person obscure the Bed/Desk regions; useful category/registration assessment is uncertain.'),
 315013:('native_unit_uncertainty','The cropped unit at the bed edge could have bedside-storage or desk semantics; preserve source Desk without forcing a category.'),
 461567:('occlusion_and_unit_uncertainty','Foreground people and partly visible multiple furniture units make individual Desk unit boundaries uncertain.'),
 323499:('surface_only_unit_uncertainty','Food/paper close-up contains only a generic support surface; the native Desk object unit is not sufficiently identifiable.'),
 319228:('surface_only_unit_uncertainty','Phone/lighter/keyboard close-up provides insufficient furniture structure to verify a Desk unit.'),
 157102:('native_unit_uncertainty','Craft activity at a white table/work surface does not resolve the native Desk versus generic Table unit.'),
 86008:('overlapping_native_class_units','Desk361 and Table1050 source masks overlap the same close-up support-surface region; separate physical object units are unresolved.'),
 403317:('surface_only_unit_uncertainty','Monitor/base/mouse close-up shows too little furniture structure for a reliable Desk unit diagnostic.'),
 485984:('surface_only_unit_uncertainty','Mouse product-display foreground dominates; only tabletop context is available and Desk unit semantics remain uncertain.'),
 301039:('surface_only_unit_uncertainty','Overhead keyboard/headset/mouse close-up shows a generic flat background rather than an identifiable furniture unit.'),
 62175:('surface_only_unit_uncertainty','Laptop/sticker foreground occupies most of the frame and leaves insufficient furniture-unit context.'),
 238488:('specialized_unit_uncertainty','The narrow foreground operator/control-console work surface is not confidently comparable to an ordinary furniture Desk unit.'),
 326313:('dominant_occluder','Cat and laptop dominate; the visible native Desk support surface is insufficient for unit/registration confidence.'),
 260135:('surface_only_unit_uncertainty','Flowers and small foreground objects obscure most support geometry; furniture-unit semantics remain uncertain.'),
 295970:('dominant_foreground','Fruit and notes dominate a tightly cropped support surface; furniture-unit context is insufficient.'),
 78741:('dominant_occluder','Cat and keyboard obscure most of the native Desk region; object-unit visibility is uncertain.'),
}
TARGETED={452229,568281,375590,192834,461567,315013,323499,166565,139326,326313,576072,157102,80853,295970,86008}

def counts(rows):
 result={}
 for cid,name in CLASSES.items():
  present=[r for r in rows if any(a['category_id']==cid and a['id'] in r['valid_core_annotation_ids'] for a in r['original_annotations'])]
  anns=[a for r in present for a in r['original_annotations'] if a['category_id']==cid and a['id'] in r['valid_core_annotation_ids']]
  result[name]=dict(native_category_id=cid,images=len(present),instances=len(anns),boxes=len(anns),masks=len(anns))
 return result

def run():
 source=read(OUT/'screening-v1.json');rows=source['images']
 if not set(QUARANTINE)<={r['coco_image_id'] for r in rows}:raise ValueError('Unexpected diagnostic photo ID')
 for r in rows:
  iid=r['coco_image_id'];notes=[]
  if iid in QUARANTINE:
   reason,note=QUARANTINE[iid];r['flags'].append(reason);notes.append(note);r['status']='quarantined'
  elif not r['flags']:r['status']='provisional_source_candidate';notes.append('Indoor furniture/workstation context and source boundaries broadly consistent in bounded assistant contact-sheet diagnostics; not independently verified ground truth.')
  if any('border_contact_proxy' in a['reasons'] for a in r['annotation_diagnostics'].values()):notes.append('At least one source object reaches the image border. Border contact alone is not a rejection or proof of annotation error.')
  r['assistant_diagnostic']=dict(method='All66 bounded contact thumbnails with unchanged LVIS polygon/box overlays;15 targeted originals/overlays for uncertain units',
   targeted_original_inspected=iid in TARGETED,checked_utc=now(),reviewer_kind='assistant',human_reviewed=False,observations=notes)
  r['human_verified']=False;r['scene_verified']=False;r['semantic_unit_verified']=False
  r['source_annotation_alignment']='broadly_consistent_at_assistant_diagnostic_resolution' if r['status']=='provisional_source_candidate' else 'unresolved'
  r['independently_verified_registration']=False
 eligible=[r for r in rows if r['status']=='provisional_source_candidate'];by_part={p:counts([r for r in eligible if r['proposed_split']==p]) for p in ('train','development')}
 commit(OUT/'screening-v2.json',dict(source,images=rows,status_counts=dict(C.Counter(r['status'] for r in rows)),eligible_partition_counts=dict(C.Counter(r['proposed_split'] for r in eligible)),eligible_category_counts=by_part,
  semantic_policy='Quarantine unresolved native object units and severe visibility/context limitations, do not correct labels. Partial/border objects are not automatically invalid; identifiable workstations remain provisional source positives.',
  human_review_performed=False,annotation_geometry_modified=False,independent_accuracy_established=False))
 commit(OUT/'CURRENT-SCREENING.json',dict(file='screening-v2.json',sha256=sha(OUT/'screening-v2.json')))
 commit(OUT/'eligible-candidates-v1.json',dict(images=eligible,source_manifest_sha256=PIN,screening_sha256=sha(OUT/'screening-v2.json'),training_approved=False,
  status='Provisional source-native positives after assistant diagnostics; not a frozen training lock or human-verified dataset',category_counts=by_part))
 print('REFINED',dict(C.Counter(r['status'] for r in rows)),dict(C.Counter(r['proposed_split'] for r in eligible)),flush=True)

if __name__=='__main__':run()
