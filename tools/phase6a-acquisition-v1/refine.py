"""Conservative assistant visual diagnostics; never human review or relabeling."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *

# Evidence from the three bounded contact sheets and targeted full-size photos.
# Uncertainty quarantines images; no category/box is repaired or human flag set.
NOTES={
 '8b1cc95385a22519':dict(action='quarantine',reason='staged_catalogue_like_scene_with_embedded_depiction',evidence='Dining furniture displayed with a framed/inset copy of the same furniture. Native chair boxes include the embedded depiction; real-room and depiction supervision are unresolved.'),
 'e07312c926231410':dict(action='quarantine',reason='table_annotation_scope_and_capture_quality_unresolved',evidence='Dark, visibly soft kitchen capture. Native Table boxes cover surfaces at a chest/worktop and partially visible upholstered-looking foreground object; conventional table identity and unit are not established.'),
 'b5bfc85ebf0c0da5':dict(action='quarantine',reason='table_region_too_partial_to_establish_object_identity',evidence='Photo dominated by a wood-burning stove. Native Table box is a small clipped region at the left edge; no clearly identifiable whole table is visible in that region.'),
 'ef87eba394783972':dict(action='quarantine',reason='visible_motion_blur_not_caught_by_laplacian_threshold',evidence='Full-size bedroom capture has visible motion smear across bed and room edges despite passing the simple Laplacian variance threshold.'),
 'e7fe54073a9a84bc':dict(action='flag_only',reason='partial_bed_close_view',evidence='Close bed/window view; complete bed extent is not visible. Keep source truncation flags and provisional partial-object status; do not infer missing box edges.'),
 '50c91362e9bc580f':dict(action='flag_only',reason='partial_bed_close_view',evidence='Bed/pillow close view with limited room context; full object extent is not established. Source bounds retained.'),
 'df0179678dcd6762':dict(action='quarantine',reason='commercial_kitchen_work_surface_scope',evidence='Commercial kitchen prep surfaces are the native Table targets; source-table versus application furniture scope remains unresolved. Preserve boxes; do not optimize this ambiguity.'),
 'aa880abb6d6d5e70':dict(action='quarantine',reason='commercial_kitchen_work_surface_scope',evidence='Native Table target is a commercial kitchen work surface. Preserve source meaning; scope is unresolved for furniture optimization.')}

def run():
 first=read(DEST/'screening-v1.json');rows=copy.deepcopy(first['images'])
 for r in rows:
  if r['image_id'] in NOTES:
   note=NOTES[r['image_id']]
   r['assistant_visual_diagnostic']=dict(**note,reviewer_kind='assistant',human_reviewed=False,inspected_utc=now())
   if note['action']=='quarantine':r['status']='quarantined';r['flags'].append(note['reason'])
  r['human_verified']=False;r['scene_verified']=False
 result=dict(images=rows,duplicates=first['duplicates'],training_authorized=False,earlier_screening_sha256=sha(DEST/'screening-v1.json'),
  scope='Assistant visual quality/ambiguity diagnostics on all 41 new contact thumbnails, with four targeted full-size inspections. No human evidence, box edits, new categories, partition changes or independent ground-truth verification.',
  contacts=[str(p.relative_to(DEST)) for p in sorted((DEST/'contact-sheets-v1').glob('*.jpg'))],notes=NOTES)
 result['supersedes_assistant_audit_sha256']=sha(DEST/'screening-v2.json')
 write_new(DEST/'screening-v3.json',result)
 print('FINAL PROVISIONAL',sum(r['status']=='provisional_automated_candidate' for r in rows),'QUARANTINED',sum(r['status']=='quarantined' for r in rows))

if __name__=='__main__':run()
