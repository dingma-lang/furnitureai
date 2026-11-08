"""Preserve legacy OI eligibility_status as well as newer status fields in a versioned overlay."""
import collections as C
from core import *
from cross_reference import preserved_status

def run():
 old=read(OUT/'existing-photo-cross-reference.json');all_status={}
 for source,path in [('openimages','data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json'),
                     ('openimages','data/open-images/image-pilots/phase6a-v1/screening-v3.json'),
                     ('coco2017','data/coco/image-pilots/phase6d-v1/screening-v2.json')]:
  for r in read(ROOT/path)['images']:
   key=r['image_id'];key=key if key.startswith(source+':') else source+':'+key
   all_status[key]=preserved_status(r,'unresolved_missing_status')
 changes=[]
 for r in old['records']:
  actual=all_status[r['historical_source']]
  if r['historical_status']!=actual:changes.append(dict(coco_image_id=r['coco_image_id'],previous_overlay_status=r['historical_status'],preserved_source_status=actual))
  r['historical_status']=actual
 old['legacy_status_resolution']=dict(derived_overlay_changes=changes,historical_files_modified=False,earlier_overlay_superseded=True)
 commit(OUT/'existing-photo-cross-reference-v2.json',old)
 summary=read(OUT/'annotation-overlay-summary.json');index={r['coco_image_id']:r for r in old['records']}
 for r in summary['new_core_class_supervision']:
  r['historical_status']=index[r['coco_image_id']]['historical_status'];r['preserved_quarantine_or_unresolved']=r['historical_status'] not in ('provisionally_accepted','provisional_candidate');r.pop('preserved_quarantine',None)
 usable=[r for r in summary['new_core_class_supervision'] if r['historical_status']=='provisionally_accepted']
 summary['new_core_class_on_previously_accepted_COCO_photos']=dict(images=len(usable),instances=sum(sum(c['lvis_instances'] for c in r['new_core_class_annotations']) for r in usable),
   categories=dict(C.Counter({str(cid):sum(c['lvis_instances'] for r in usable for c in r['new_core_class_annotations'] if c['native_lvis_category_id']==cid) for cid in (181,361,626)})),
   automatic_annotation_eligibility_granted=False,remaining_checks='Native unit semantics, partial-label policy and mask registration; no training approved')
 summary['legacy_status_resolution']='Authoritative cross-reference is existing-photo-cross-reference-v2.json. Source status and eligibility_status fields preserved, no historical record changed.'
 commit(OUT/'annotation-overlay-summary-v2.json',summary)
 print('Resolved',len(changes),'legacy OI overlay status fields; no source modification',flush=True)

if __name__=='__main__':run()
