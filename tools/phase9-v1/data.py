"""Explicit allowed image membership; final36 pixels gated on a sealed selection."""
import importlib.util
import sys
from core import *

def exposed_rows():
 lock=read(OLD_LOCK);rows=[r for r in lock['rows'] if r['split']=='development']
 if len(rows)!=18 or sum(not a['ignore'] for r in rows for a in r['annotations'])!=41:raise ValueError('Historical development reference changed')
 return [dict(r,phase9_partition='exposed',path=str(ROOT/'data/open-images/image-pilots/phase5-v1'/r['file']),fp_coverage=dict(r['coverage'])) for r in rows]

def final_rows_metadata():
 raw=read(READINESS)['source_rows'];out=[]
 for source in raw:
  if source['split']!='development':continue
  targets=source['resolved_targets'];scope=source['source_native_scope'];annotations=[];coverage={c:False for c in CLASSES};fp={c:False for c in CLASSES}
  if source['source']=='phase6a_openimages':base=ROOT/'data/open-images/image-pilots/phase6a-v1';screen=read(base/'screening-v3.json')['images']
  elif source['source']=='phase6d_coco':base=ROOT/'data/coco/image-pilots/phase6d-v1';screen=read(base/'screening-v2.json')['images']
  elif source['source']=='phase7_LVIS':base=ROOT/'data/lvis/image-pilots/phase7-v1';screen=read(base/'screening-v2.json')['images']
  elif source['source']=='phase8c_LVIS':base=ROOT/'data/lvis/image-pilots/phase8b-v1';screen=read(base/'screening-v2.json')['images']
  else:raise ValueError('Unknown fresh source')
  original=next(r for r in screen if r['image_id']==source['image_id']);width=original['width'];height=original['height']
  for t in targets:
   label=t['comparison_group'];a=t['original_annotation'];coverage[label]=True
   if t['native_dataset']=='OpenImages':box=list(a['box']);flags=a['flags']
   else:
    x,y,w,h=a['bbox'];box=[x/width,y/height,(x+w)/width,(y+h)/height];flags={'IsTruncated':int(t['border_or_truncation']),'IsOccluded':0}
   annotations.append(dict(label=label,box=box,ignore=bool(t.get('group_of')),source_mid=str(t['native_id']),flags=flags))
   same_native=[a for a in original['original_annotations'] if a.get('category_id',a.get('source_mid'))==t['native_id']]
   all_retained=len([q for q in targets if q['native_id']==t['native_id']])==len(same_native)
   # Only exact native-class source coverage with all native targets retained
   # can supply unmatched FPs. COCO scope stays unverified, LVIS NEX unknown.
   if t['native_dataset']=='LVIS':fp[label]=all_retained and t['native_id'] not in scope.get('not_exhaustive_category_ids',[])
   elif t['native_dataset']=='OpenImages':fp[label]=all_retained and scope.get('native_image_level_labels',{}).get(t['native_id'])==1
  if 'openimages' in source['source']:
   from_native={'/m/02crq1':'sofa','/m/01mzpv':'chair','/m/04bcr3':'table','/m/03ssj5':'bed','/m/01y9k5':'desk','/m/01s105':'cabinet','/m/0dtln':'lamp'}
   for native,val in scope.get('native_image_level_labels',{}).items():
    if val==0 and native in from_native:coverage[from_native[native]]=True;fp[from_native[native]]=True
  out.append(dict(image_id=source['image_id'],phase9_partition='final',split='development',sha256=source['sha256'],
   width=width,height=height,path=str(base/original['file']),annotations=annotations,coverage=coverage,fp_coverage=fp,
   identity_group=source['known_source_component'],source=source['source'],source_native_scope=scope,
   evaluation_scope='Native-source partial annotation agreement; not whole-project accuracy',human_verified=False))
 if len(out)!=36 or sum(not a['ignore'] for r in out for a in r['annotations'])!=126:raise ValueError('Fresh36 membership changed')
 exposed=exposed_rows()
 if {r['image_id'] for r in out}&{r['image_id'] for r in exposed} or {r['sha256'] for r in out}&{r['sha256'] for r in exposed}:raise ValueError('Fresh/exposed overlap')
 return out

def image(row):
 from PIL import Image,ImageOps
 allowed=exposed_rows() if row['phase9_partition']=='exposed' else final_rows_metadata() if row['phase9_partition']=='final' else []
 if row not in allowed:raise ValueError('Image not in exact approved nonfrozen membership')
 if row['phase9_partition']=='final':
  selected=checked(OUT/'selection.json')
  if not selected or selected['code_hashes']!=code_hashes() or selected['fresh_manifest_sha256']!=sha(OUT/'fresh-metadata-lock.json'):raise PermissionError('Final pixels require immutable development selection and identical code')
 p=Path(row['path']).resolve()
 if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Source path/hash mismatch')
 with Image.open(p) as im:
  im=ImageOps.exif_transpose(im).convert('RGB')
  if im.size!=(row['width'],row['height']):raise ValueError('Annotation/image coordinate frame mismatch')
  return im
