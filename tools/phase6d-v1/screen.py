"""Model-free actual-image and unchanged-annotation screening."""
import collections as C
import math
from common import *

def signature(image):
 import numpy as np
 from PIL import ImageOps
 rgb=ImageOps.exif_transpose(image).convert('RGB');gray=np.asarray(rgb.convert('L'),dtype=float);small=np.asarray(rgb.convert('L').resize((9,8)),dtype=float)
 dh=sum(int(b)<<i for i,b in enumerate((small[:,1:]>small[:,:-1]).ravel()))
 a=np.asarray(rgb.convert('L').resize((32,32)),dtype=float);cos=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64);coeff=(cos@a@cos.T).ravel()[1:];ph=sum(int(b)<<i for i,b in enumerate(coeff>np.median(coeff)))
 lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
 return dict(decoded_pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),dhash64=f'{dh:016x}',phash63=f'{ph:016x}',gray_std=float(gray.std()),laplacian_variance=float(lap.var()),white_fraction=float(np.mean(np.all(np.asarray(rgb)>245,axis=2))))

def duplicate(a,b):
 if a.get('sha256') and a['sha256']==b.get('sha256') or a.get('decoded_pixel_sha256') and a['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'):return 'exact'
 if any(a.get(k) and b.get(k) and (int(a[k],16)^int(b[k],16)).bit_count()<=threshold for k,threshold in [('dhash64',4),('phash63',6)]):return 'suspected_perceptual'
 return None

def xyxy(a):
 x,y,w,h=a['bbox'];return [x,y,x+w,y+h]
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/union if union else 0.

def annotation_diagnostics(annotations,width,height):
 valid=[];invalid=[];border=[];small=[];crowd=[];overlap=[];category_conflicts=[]
 for a in annotations:
  b=xyxy(a);good=all(math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=width+1e-6 and 0<=b[1]<b[3]<=height+1e-6 and a['category_id'] in CLASSES
  if not good:invalid.append(a['id']);continue
  if a['iscrowd']:crowd.append(a['id'])
  else:valid.append(a['id'])
  if min(b[0],b[1],width-b[2],height-b[3])<=1:border.append(a['id'])
  if (b[2]-b[0])*(b[3]-b[1])/width/height<.005:small.append(a['id'])
 for i,a in enumerate(annotations):
  for b in annotations[i+1:]:
   v=iou(xyxy(a),xyxy(b))
   if v>=.1:overlap.append(dict(a=a['id'],b=b['id'],iou=v,heuristic='Box overlap, not a confirmed occlusion or duplicate'))
   if v>=.8 and {a['category_id'],b['category_id']}=={63,65}:category_conflicts.append(dict(a=a['id'],b=b['id'],iou=v,reason='Near-coincident native couch/bed source regions require disambiguation'))
 return dict(valid_noncrowd_annotation_ids=valid,invalid_annotation_ids=invalid,crowd_annotation_ids=crowd,border_contact_ids=border,small_region_ids=small,box_overlap_pairs=overlap,category_conflicts=category_conflicts,
  has_multiple_furniture_objects=len(valid)>1,occlusion_flags='Not provided by COCO; overlap/border are geometry proxies only',visual_box_completeness='Not established by geometry',unlabelled_furniture_is_not_confirmed_background=True)

def run():
 from PIL import Image
 m=manifest();lookup={r['image_id']:r for r in m['images']};rows=[]
 for photo in read(OUT/'acquisition-complete.json')['records']:
  p=OUT/photo['file'];source=lookup[photo['image_id']]
  if sha(p)!=photo['sha256']:raise ValueError('Acquired image changed')
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:im.load();s=signature(im)
  d=annotation_diagnostics(source['original_furniture_annotations'],photo['width'],photo['height']);flags=[]
  if min(photo['width'],photo['height'])<200:flags.append('small_photo')
  if s['gray_std']<8 or s['white_fraction']>.98:flags.append('blank_or_catalogue_like')
  if s['laplacian_variance']<10:flags.append('blur_heuristic')
  if photo['exif_orientation']!=1:flags.append('orientation_unresolved')
  if d['invalid_annotation_ids']:flags.append('invalid_source_annotation')
  if d['category_conflicts']:flags.append('native_couch_bed_category_conflict')
  rows.append(dict(**photo,**s,annotation_diagnostics=d,original_annotations=source['original_furniture_annotations'],flags=flags,status='quarantined' if flags else 'pending_scene_and_annotation_inspection',human_verified=False,scene_verified=False,
   rights_record=source['rights_record'],rights_record_sha256=source['rights_record_sha256']))
 historical=[]
 for file in [ROOT/'data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json',ROOT/'data/open-images/image-pilots/phase6a-v1/screening-v3.json']:
  historical.extend(dict(r,historical_origin=str(file.relative_to(ROOT))) for r in read(file)['images'])
 pairs=[]
 for i,a in enumerate(rows):
  for b in rows[i+1:]+historical:
   kind=duplicate(a,b)
   if kind:
    pair=dict(a=a['image_id'],b=b['image_id'],kind=kind,cross_partition=a['proposed_split']!=b.get('proposed_split'),historical=bool(b.get('historical_origin')));pairs.append(pair)
    # No inferred merging or movement. All suspects stay out until resolved.
    a['flags'].append(kind+'_image_overlap');a['status']='quarantined'
    if not pair['historical']:b['flags'].append(kind+'_image_overlap');b['status']='quarantined'
 # ABO comparisons use existing hashes only, never old/frozen image decoding.
 abo=[]
 for file in [ROOT/'data/abo/image-pilots/phase1b-v1/final-screening.json',ROOT/'data/abo/image-pilots/phase3-expansion-v1/audit-v3/screening.json']:abo.extend(read(file)['images'])
 for a in rows:
  for b in abo:
   if a['sha256']==b.get('sha256') or a['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'):
    pairs.append(dict(a=a['image_id'],b=b['image_id'],kind='exact',historical=True,cross_partition=True));a['status']='quarantined';a['flags'].append('historical_abo_hash_overlap')
 space(8*2**20)
 commit(OUT/'screening-v1.json',dict(images=rows,duplicate_pairs=pairs,counts=dict(C.Counter(r['status'] for r in rows)),abo_metadata_hash_records_compared=len(abo),historical_image_pixels_opened=False,model_inference=False,
  limitations='Perceptual thresholds flag suspicion rather than prove identity. Old Open Images stored signatures are compared; ABO is existing-hash only. Final eligibility requires scene/source-annotation inspection; no human review inferred.'))
 print('AUTOMATIC SCREEN',len(rows),dict(C.Counter(r['status'] for r in rows)),'duplicate pairs',len(pairs),flush=True)

if __name__=='__main__':run()
