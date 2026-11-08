"""Model-free native LVIS boxes/polygons, photo signatures and isolated eligibility."""
import collections as C
import math
from PIL import Image,ImageDraw
import numpy as np
from pilot_core import *
from signatures import signature,duplicate,xyxy,iou

def annotation_check(a,width,height):
 b=a.get('bbox');flags=[]
 if not isinstance(b,list) or len(b)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in b):return dict(valid=False,reasons=['invalid_bbox'])
 x,y,w,h=b
 if not (0<=x<width and 0<=y<height and w>0 and h>0 and x+w<=width+.02 and y+h<=height+.02):return dict(valid=False,reasons=['bbox_outside_image'])
 seg=a.get('segmentation')
 if not isinstance(seg,list) or not seg:return dict(valid=False,reasons=['unsupported_or_missing_mask'])
 canvas=Image.new('1',(width,height));draw=ImageDraw.Draw(canvas);xs=[];ys=[]
 for polygon in seg:
  if not isinstance(polygon,list) or len(polygon)<6 or len(polygon)%2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in polygon):return dict(valid=False,reasons=['invalid_polygon'])
  if any(not 0<=v<=(width if i%2==0 else height) for i,v in enumerate(polygon)):return dict(valid=False,reasons=['polygon_outside_image'])
  xs+=polygon[::2];ys+=polygon[1::2];draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=1)
 bounds=[min(xs),min(ys),max(xs),max(ys)];boxends=[x,y,x+w,y+h];delta=max(abs(p-q) for p,q in zip(bounds,boxends));area=int(np.asarray(canvas).sum())
 if delta>2:flags.append('bbox_polygon_extent_discrepancy')
 if area<16:flags.append('tiny_visible_mask')
 if min(x,y,width-x-w,height-y-h)<=1:flags.append('border_contact_proxy')
 return dict(valid=True,reasons=flags,raster_area_pixels=area,source_area=a.get('area'),polygon_extent_xyxy=bounds,bbox_polygon_max_delta_pixels=delta,
  core_class=a['category_id'] in CLASSES,visual_registration_verified=False,semantic_unit_verified=False,
  caveat='Pillow rasterization is diagnostic and may differ from COCO rasterization; source polygons are unchanged. Bounds do not prove image alignment.')

def run():
 rows=[];sources={r['coco_image_id']:r for r in manifest()['images']}
 for photo in read(OUT/'acquisition-complete.json')['records']:
  r=sources[photo['coco_image_id']];path=OUT/photo['file']
  if sha(path)!=photo['sha256']:raise ValueError('Acquired photo changed')
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:im.load();features=signature(im);fmt=im.format
  ann={str(a['id']):annotation_check(a,photo['width'],photo['height']) for a in r['original_furniture_annotations']}
  flags=[]
  if fmt!='JPEG' or photo['exif_orientation']!=1:flags.append('format_orientation_uncertain')
  if min(photo['width'],photo['height'])<200:flags.append('unusually_small_photo')
  if features['gray_std']<8 or features['white_fraction']>.98:flags.append('blank_or_nearly_uniform_heuristic')
  if features['laplacian_variance']<10:flags.append('blur_heuristic')
  if any(not d['valid'] or 'bbox_polygon_extent_discrepancy' in d['reasons'] for d in ann.values()):flags.append('source_box_mask_geometry_unresolved')
  valid_core=[a['id'] for a in r['original_furniture_annotations'] if a['category_id'] in CLASSES and ann[str(a['id'])]['valid'] and 'tiny_visible_mask' not in ann[str(a['id'])]['reasons']]
  if not valid_core:flags.append('no_large_valid_core_annotation')
  overlaps=[]
  for i,a in enumerate(r['original_furniture_annotations']):
   for b in r['original_furniture_annotations'][i+1:]:
    overlap=iou(xyxy(a),xyxy(b))
    if overlap>=.8 and a['category_id'] in CLASSES and b['category_id'] in CLASSES and a['category_id']!=b['category_id']:overlaps.append(dict(a=a['id'],b=b['id'],iou=overlap,interpretation='Geometry suspicion, not proven category error'))
  if overlaps:flags.append('near_coincident_different_core_categories')
  rows.append(dict(**photo,**features,format=fmt,flags=flags,status='quarantined' if flags else 'pending_scene_and_registration_diagnostics',
   original_annotations=r['original_furniture_annotations'],source_image_scope=r['source_image_scope'],annotation_diagnostics=ann,valid_core_annotation_ids=valid_core,
   near_coincident_core_boxes=overlaps,human_verified=False,scene_verified=False,semantic_unit_verified=False,
   native_categories=sorted({a['category_id'] for a in r['original_furniture_annotations']})))
 history=[]
 for relative in ['data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json','data/open-images/image-pilots/phase6a-v1/screening-v3.json','data/coco/image-pilots/phase6d-v1/screening-v2.json']:
  for r in read(ROOT/relative)['images']:history.append(dict(r,historical_origin=relative))
 pairs=[]
 for i,a in enumerate(rows):
  for b in rows[i+1:]+history:
   kind=duplicate(a,b)
   if kind:
    old=bool(b.get('historical_origin'));pairs.append(dict(a=a['image_id'],b=b['image_id'],kind=kind,historical=old,historical_origin=b.get('historical_origin'),cross_partition=a['proposed_split']!=b.get('proposed_split')))
    a['flags'].append(kind+'_image_overlap');a['status']='quarantined'
    if not old:b['flags'].append(kind+'_image_overlap');b['status']='quarantined'
 abo=[]
 for path in [ROOT/'data/abo/image-pilots/phase1b-v1/final-screening.json',ROOT/'data/abo/image-pilots/phase3-expansion-v1/audit-v3/screening.json']:abo+=read(path)['images']
 for a in rows:
  for b in abo:
   if a['sha256']==b.get('sha256') or a['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'):
    pairs.append(dict(a=a['image_id'],b=b['image_id'],kind='exact',historical=True,historical_origin='ABO stored hashes only',cross_partition=True));a['flags'].append('historical_ABO_hash_overlap');a['status']='quarantined'
 commit(OUT/'screening-v1.json',dict(images=rows,duplicate_pairs=pairs,historical_signature_records=len(history),ABO_stored_hash_records=len(abo),historical_image_pixels_opened=False,
  mask_geometry_policy='Finite original boxes/polygons, in-bounds vertices, positive raster area and extent tolerance2px. No source geometry edits.',
  perceptual_policy='dHash Hamming<=4 or pHash63 Hamming<=6 flags suspicion; no proof of physical identity.',model_inference=False))
 space();print('SCREEN',len(rows),dict(C.Counter(r['status'] for r in rows)),'duplicate/similarity pairs',len(pairs),flush=True)

if __name__=='__main__':run()
