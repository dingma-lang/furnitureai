"""Read-only metadata/hash/coordinate assessment of ten already-local photos; no image decoding."""
import collections as C
import sqlite3
from audit_core import *

def mask_bounds(annotation):
 mask=annotation.get('segmentation')
 if not isinstance(mask,list) or not mask:return None
 xs=[v for polygon in mask for v in polygon[::2]];ys=[v for polygon in mask for v in polygon[1::2]]
 return [min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)] if xs and ys else None

def run():
 opportunity=read(BASE/'annotation-overlay-summary-v2.json');rows=[r for r in opportunity['new_core_class_supervision'] if r['historical_status']=='provisionally_accepted']
 if len(rows)!=10:raise ValueError('Prior ten-photo opportunity changed')
 source=read(BASE/'existing-image-lvis-overlays.json')['records'];byid=C.defaultdict(list)
 for r in source:
  a=r['lvis_annotation']
  if a['category_id'] in (181,361,626):byid[a['image_id']].append(a)
 photo_base=ROOT/'data/coco/image-pilots/phase6d-v1';acquired={r['coco_image_id']:r for r in read(photo_base/'acquisition-complete.json')['records']}
 rights={r['coco_image_id']:r for r in read(photo_base/'license-and-provenance-ledger.json')['records']}
 db=sqlite3.connect(f'file:{BASE}/furniture-index.sqlite?mode=ro',uri=True);records=[]
 for r in rows:
  iid=r['coco_image_id'];old=acquired[iid];annotations=byid[iid];path=photo_base/old['file']
  if sha(path)!=old['sha256']:raise ValueError('Existing photograph SHA changed')
  im=db.execute('SELECT width,height,url,neg,nex FROM images WHERE id=?',(iid,)).fetchone()
  if (old['width'],old['height'])!=(im[0],im[1]) or old['exif_orientation']!=1:raise ValueError('Image-coordinate metadata is inconsistent')
  diagnostics=[]
  for a in annotations:
   maskbox=mask_bounds(a);deltas=[abs(x-y) for x,y in zip(a['bbox'],maskbox)] if maskbox else None
   diagnostics.append(dict(native_annotation=a,polygon_extent_xywh=maskbox,bbox_polygon_extent_max_delta_pixels=max(deltas) if deltas else None,
    interpreted_category=CORE[a['category_id']],native_unit_semantics_verified=False,
    coordinate_frame='Native top-left pixel xywh/polygons at exact COCO image dimensions; no transform applied'))
  lic=rights[iid]
  records.append(dict(coco_image_id=iid,lvis_image_id=iid,lvis_split=r['lvis_split'],existing_partition=r['historical_partition'],
   existing_photo_file=str(path.relative_to(ROOT)),existing_photo_sha256=old['sha256'],existing_photo_width=old['width'],existing_photo_height=old['height'],exif_orientation=old['exif_orientation'],
   metadata_identity_dimensions_consistent=True,source_image_url=im[2],photo_rights_record_sha256=lic['rights_record_sha256'],photo_license=lic['photo_license'],
   creator=lic['creator'],creator_url=lic['creator_url'],native_negative_ids=read_scope(im[3]),native_not_exhaustive_ids=read_scope(im[4]),
   annotation_diagnostics=diagnostics,image_to_mask_visual_registration_verified=False,semantic_unit_verified=False,
   annotation_training_eligible=False,dataset_merge_performed=False,human_reviewed=False,
   outstanding=['Confirm actual visible-mask/image alignment and source annotation unit; dimensions alone are insufficient',
                'Keep each original native category and partial/exhaustive scope; do not convert unknown regions into background',
                'Preserve existing development membership and creator/scene identity grouping','Separate approval needed before dataset activation or training']))
 if sum(len(r['annotation_diagnostics']) for r in records)!=33:raise ValueError('Expected33 annotations changed')
 db.close();commit(OUT/'existing-ten-photo-registration-assessment.json',dict(records=records,photos=10,native_annotations=33,
  metadata_identity_dimensions_pass=10,photo_hashes_verified=10,photograph_downloads=0,photograph_decoding_or_visual_screening_performed=False,
  semantic_and_visual_registration_passes=0,annotations_merged_into_training=0,
  status='Metadata-compatible coordinate candidates; semantic/visual registration unresolved'))
 print('Ten existing photo hashes/dimensions and33 original annotations assessed; no activation',flush=True)

def read_scope(text):return json.loads(text)

if __name__=='__main__':run()
