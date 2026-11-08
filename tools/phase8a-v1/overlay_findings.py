"""Explicit assistant findings from ten actual photos; no corrected labels or activation."""
from prep_core import *

def hint(iid,a):
 cid=a['category_id'];x,y,w,h=a['bbox']
 if cid==181:return 'built_in_cabinet_front_or_bay_unit_unresolved','Individual outlined fronts/bays may be door/panel or cabinet units; not verified whole cabinets or one Open Images Cabinetry run.'
 if cid==361:
  if iid==68162:return 'writing_workstation_visually_supported','Visible writing/workstation furniture is broadly consistent with the native Desk mask; independent shared-class unit signoff still absent.'
  if iid==384375:return 'hotel_writing_vanity_table_candidate','Small hotel writing/vanity unit is partly occluded by a chair; native Desk is plausible but shared Desk/Table subtype policy remains open.'
  return 'occluded_rear_work_storage_unit_unresolved','Rear furniture behind sofas has limited unit visibility; Desk versus storage/side-table boundaries remain unresolved.'
 if iid==131352:return 'wall_mounted_bedside_light','Wall-mounted fixture; not a floor lamp. Source Lamp626 preserved.'
 if iid==165742:return 'distant_lighting_type_unresolved','Tiny background lighting/reflection regions; source-mask geometry is valid, but subtype and physical unit cannot be confirmed confidently.'
 if iid==199688:
  if x>540:return 'floor_standing_lamp_visually_supported','Visible tall freestanding lamp at room edge; source Lamp626 preserved.'
  if x>350:return 'wall_mounted_light','Wall-attached fixture; not a floor lamp.'
  return 'hanging_pendant_light','Pendant fixture over the bar; not a floor lamp.'
 if iid==279122:
  if h>3*w:return 'floor_standing_candidate_base_occluded','Tall light behind seating; base is occluded and full unit semantics remain provisional.'
  return 'ceiling_fan_light','Light attached to the ceiling fan; not a floor or table lamp.'
 if iid==295693:return 'table_lamp_visually_supported','Shade/base visibly rest on a storage unit; source Lamp626, not a floor-lamp claim.'
 if iid==468162:return 'standing_lamp_type_unresolved','Lamp next to sofa; support/base is obscured, so table-versus-floor subtype is not established.'
 if iid==504586:return 'wall_mounted_reading_fixture','Bedhead/wall reading fixture; not a floor lamp.'
 return 'unresolved','Source native semantics remain unresolved.'

def run():
 data=read(OUT/'existing33-registration-v1.json');findings=[]
 for photo in data['records']:
  for item in photo['annotations']:
   a=item['annotation'];subtype,note=hint(photo['coco_image_id'],a)
   item['visual_semantic_assessment']='assistant_inspected_not_independent_ground_truth';item['assistant_visual_subtype_hint']=subtype;item['observations']=note
   item['active']=False;item['independent_semantic_verification_complete']=False;item['human_reviewed']=False
   findings.append(dict(coco_image_id=photo['coco_image_id'],partition=photo['original_partition'],annotation_id=a['id'],native_category_id=a['category_id'],
    assistant_subtype_hint=subtype,source_box_polygon_alignment='Broadly consistent at inspected pixel resolution; no independent reference-mask truth',
    coordinate_metadata_verified=item['coordinate_metadata_consistent'],unit_policy_complete=False,independent_semantic_verification_complete=False,active=False,
    note=note,subtype_hint_is_not_a_relabel=True))
 data['annotations_activated']=0;data['assistant_visual_photos_inspected']=10;data['findings']=findings
 commit(OUT/'existing33-registration-v2.json',data)
 commit(OUT/'approval-B-existing-label-activation.json',dict(approvable_now=False,activation_requested=False,independently_verified_activation_annotation_ids=[],annotations_kept_inactive=33,
  source_assessment_sha256=sha(OUT/'existing33-registration-v2.json'),potential_native_counts=dict(cabinet=15,desk=4,lamp=14),
  blockers=['Approve/verify whole cabinet versus front/bay/run units','Define portable/table/floor versus wall/ceiling Lamp scope','Resolve two occluded Desk units and Desk/Table subtypes','Independent semantic/registration confirmation for any exact activation list','Preserve development labels for evaluation only; no hyperparameter selection on reserved sets'],
  hypothetical_development_benefit='At most two existing fresh-development photos could gain native Lamp supervision; no new photos and no Cabinet/Bed development images are added.',
  downloads=0,installations=0,training=False))
 print('All33 visually inspected source overlays remain inactive; no verified activation list invented',flush=True)

if __name__=='__main__':run()
