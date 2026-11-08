"""Assistant scene/ambiguity diagnostics, explicitly not human ground truth."""
import collections as C
import copy
from common import *
from screen import xyxy,iou

NOTES={
 280490:('quarantined','foreground_food_closeup_limits_furniture_scope','Food/tea dominate; only a close partial upholstered back and heavily covered tabletop remain. Broader furniture unit/bounds remain uncertain.'),
 577718:('quarantined','foreground_still_life_hides_furniture','Bottle/glass dominate the frame; background seating is severely clipped/obscured, unsuitable for establishing the furniture annotation units.'),
 208516:('quarantined','foreground_laptop_obscures_tiny_furniture','Laptop occupies most of the capture; source bed/sofa regions are small, clipped background fragments.'),
 369567:('rejected','multi_panel_composite_not_single_room_capture','Full-size image consists of three visibly separated panels: fridge, portrait and hallway. It is not a single coherent room photograph for this pilot.'),
 235624:('quarantined','mirror_portrait_and_obscured_furniture_scope','Mirror portrait dominates; furniture is mostly tiny partial background/foreground regions. Source chair/bed boxes also have near-coincident extents.'),
 89062:('quarantined','foreground_glass_and_out_of_focus_background','Glass dominates a low-angle still life; furniture is tiny/partial or blurred in the far background.'),
 45463:('quarantined','bed_only_thin_clipped_strip','Source bed annotation covers only a thin bottom-edge strip in a bike/appliance-dominated capture; bed identity/bounds cannot be reliably established for this pilot.')}

def run():
 first=read(OUT/'screening-v1.json');rows=copy.deepcopy(first['images'])
 for r in rows:
  anns=r['original_annotations'];conflicts=[]
  for i,a in enumerate(anns):
   for b in anns[i+1:]:
    score=iou(xyxy(a),xyxy(b))
    if score>=.8 and a['category_id']!=b['category_id'] and a['category_id'] in (62,63,65) and b['category_id'] in (62,63,65):conflicts.append(dict(a=a['id'],b=b['id'],categories=[a['category_id'],b['category_id']],iou=score))
  r['near_coincident_exclusive_furniture_categories']=conflicts
  if conflicts:r['flags'].append('near_coincident_chair_couch_bed_category_ambiguity');r['status']='quarantined'
  if r['coco_image_id'] in NOTES:
   status,reason,evidence=NOTES[r['coco_image_id']];r['status']=status;r['flags'].append(reason)
   r['assistant_inspection']=dict(reviewer_kind='assistant',evidence=evidence,decision=status,checked_utc=now(),human_reviewed=False)
  else:
   if r['status']=='pending_scene_and_annotation_inspection':r['status']='provisionally_accepted'
   r['assistant_inspection']=dict(reviewer_kind='assistant',method='Bounded full72-image contact-sheet inspection with native-box overlays; geometry ambiguity gate',scene_assessment='Indoor furniture context supported; occlusion/truncation and native boundaries remain source-label limitations',checked_utc=now(),human_reviewed=False)
  r['human_verified']=False;r['scene_verified']=False
  r['visual_visibility_caveats']=dict(border_contact_regions=len(r['annotation_diagnostics']['border_contact_ids']),box_overlap_pairs=len(r['annotation_diagnostics']['box_overlap_pairs']),small_regions=len(r['annotation_diagnostics']['small_region_ids']),
   interpretation='Measured geometry proxies; no fabricated occlusion, truncation or amodal/whole-object labels')
 accepted=[r for r in rows if r['status']=='provisionally_accepted'];groups=C.defaultdict(set)
 for r in accepted:groups[r['identity_group']].add(r['proposed_split'])
 if any(len(parts)>1 for parts in groups.values()):raise ValueError('Current eligible source group spans partitions')
 commit(OUT/'screening-v2.json',dict(images=rows,duplicate_pairs=first['duplicate_pairs'],counts=dict(C.Counter(r['status'] for r in rows)),original_screening_sha256=sha(OUT/'screening-v1.json'),
  inspection_scope='All72 new contact thumbnails inspected; seven targeted full-resolution checks. Assistant quality diagnostics, no human label or boundary confirmation.',
  category_conflict_threshold=.8,original_annotations_unchanged=True,model_inference=False,human_review_invented=False))
 commit(OUT/'CURRENT-SCREENING.json',dict(path='screening-v2.json',sha256=sha(OUT/'screening-v2.json'),training_approved=False))
 commit(OUT/'eligible-candidates-v1.json',dict(images=accepted,source_manifest_sha256=PIN,final_screening_sha256=sha(OUT/'screening-v2.json'),training_approved=False,final_training_split_created=False,source_weak_labels=True))
 print('FINAL SCREEN',dict(C.Counter(r['status'] for r in rows)),dict(C.Counter(r['proposed_split'] for r in accepted)),flush=True)

if __name__=='__main__':run()
