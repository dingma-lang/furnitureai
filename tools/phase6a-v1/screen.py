"""Future model-free photograph/duplicate gate; no fabricated human labels."""
import collections as C
from core import *

def signature(image):
 import numpy as np
 from PIL import ImageOps
 rgb=ImageOps.exif_transpose(image).convert('RGB');gray=np.asarray(rgb.convert('L'),dtype=float)
 small=np.asarray(rgb.convert('L').resize((9,8)),dtype=float);bits=(small[:,1:]>small[:,:-1]).ravel()
 a=np.asarray(rgb.convert('L').resize((32,32)),dtype=float);cos=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64)
 coeff=(cos@a@cos.T).ravel()[1:];ph=sum(int(b)<<i for i,b in enumerate(coeff>np.median(coeff)))
 lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
 return dict(decoded_pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),dhash64=f'{sum(int(b)<<i for i,b in enumerate(bits)):016x}',phash63=f'{ph:016x}',gray_std=float(gray.std()),laplacian_variance=float(lap.var()),white_fraction=float(np.mean(np.all(np.asarray(rgb)>245,axis=2))))

def duplicate(a,b):
 exact=bool(a.get('sha256') and a['sha256']==b.get('sha256')) or bool(a.get('decoded_pixel_sha256') and a['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'))
 near=any(a.get(k) and b.get(k) and (int(a[k],16)^int(b[k],16)).bit_count()<=threshold for k,threshold in [('dhash64',4),('phash63',6)])
 return 'exact' if exact else 'suspected_perceptual' if near else None

def screen():
 from PIL import Image
 m=read(MANIFEST);rows=[];protected=set(read(OUT/'protected-identities.json'))
 scope={r['image_id']:r for r in read(OUT/'annotation-scope-audit.json')['records']}
 for r in read(DEST/'acquisition-complete.json')['records']:
  path=DEST/r['file']
  if sha(path)!=r['sha256']:raise ValueError('Changed photo')
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:im.load();sig=signature(im)
  flags=[]
  if min(r['width'],r['height'])<200:flags.append('small_image')
  if sig['gray_std']<8 or sig['white_fraction']>.98:flags.append('blank_or_catalogue_like')
  if sig['laplacian_variance']<10:flags.append('possibly_blurred_heuristic')
  if r['sha256'] in protected or sig['decoded_pixel_sha256'] in protected:flags.append('historical_holdout_overlap')
  source=r['source'];scope_risks=source['taxonomy_risks']
  # Ambiguous native Desk/Cabinetry supervision remains unknown; retain their
  # boxes/provenance as evidence, but mask these classes until scope reviewed.
  mask=set(scope[r['image_id']]['optimization_classes_to_mask'])
  coverage={c:value and c not in mask for c,value in source['coverage'].items()}
  annotations=[dict(g,ignore=True) if g['label'] in mask else g for g in source['annotations']]
  if not any(not g['ignore'] and coverage[g['label']] for g in annotations):flags.append('no_unambiguous_training_box')
  rows.append(dict(**{k:r[k] for k in ('image_id','file','sha256','width','height')},**sig,proposed_split=source['proposed_split'],identity_group=source['identity_group'],source_identity=source['source_identity'],photographer_identity=source['photographer_identity'],metadata_original_md5=source['metadata_original_md5'],flags=flags,original_annotations=source['annotations'],annotations=annotations,coverage=coverage,scope_risks=scope_risks,scene_verified=False,human_verified=False,status='quarantined' if flags else 'provisional_automated_candidate'))
 old=read(OLD/'screening-final-v1/screening.json')['images']
 pairs=[]
 for i,a in enumerate(rows):
  for b in rows[i+1:]+old:
   result=duplicate(a,b)
   if result:
    cross=b.get('proposed_split')!=a['proposed_split'];historical=b in old
    pairs.append(dict(a=a['image_id'],b=b['image_id'],kind=result,cross_partition=cross,historical=historical))
    # No automatic family merge or membership change. Quarantine both new
    # members for cross-partition similarities and every exact duplicate.
    if cross or historical or result=='exact':
     a['status']='quarantined';a['flags'].append(result+'_overlap')
     if b in rows:b['status']='quarantined';b['flags'].append(result+'_overlap')
 isolation([r for r in rows if r['status']=='provisional_automated_candidate'],protected)
 result=dict(images=rows,duplicates=pairs,counts=dict(C.Counter(r['status'] for r in rows)),training_authorized=False,indoor_suitability='metadata-supported, not visually confirmed; automated blur/blank and duplicate checks are heuristics')
 reserve=2*len(json.dumps(result).encode())+1048576
 if allocated(DEST)+allocated(OUT)+allocated(HERE)+reserve>STORAGE:raise RuntimeError('Insufficient total storage for screening artifacts')
 write_new(DEST/'screening-v1.json',result)
 print('SCREENED',len(rows),'No training or final training manifest created')

if __name__=='__main__':screen()
