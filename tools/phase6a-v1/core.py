"""Isolated Phase 6A contracts. No network, training, or mutable historical output."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/open-images/preparation/phase6a-v1'
META=ROOT/'data/open-images/metadata/phase5-v1'
OLD=ROOT/'data/open-images/image-pilots/phase5-v1'
DEST=ROOT/'data/open-images/image-pilots/phase6a-v1'
MANIFEST=OUT/'acquisition-manifest-v2.json'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
MIDS=dict(zip(CLASSES,('/m/02crq1','/m/01mzpv','/m/04bcr3','/m/03ssj5','/m/01y9k5','/m/01s105','/m/0dtln')))
COCO={62:('chair','chair'),63:('couch','sofa'),65:('bed','bed'),67:('dining table','table')}
SEED='phase6a-commercial-detection-v1-20261105'
TRANSFER=2**30
STORAGE=2**31
IMAGE_MAX=3*2**20
SELECTED_SHA='d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def atomic(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def allocated(root):return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file())
def box(value):
 if len(value)!=4 or not all(math.isfinite(x) for x in value) or not 0<=value[0]<value[2]<=1 or not 0<=value[1]<value[3]<=1:raise ValueError('Invalid normalized xyxy box')
 return value
def coco_box(value,width,height):
 if width<=0 or height<=0 or len(value)!=4:raise ValueError('Invalid COCO dimensions')
 x,y,w,h=map(float,value)
 return box([x/width,y/height,(x+w)/width,(y+h)/height])
def license_url(url):
 u=urlparse(url)
 if u.scheme not in ('https','http') or u.hostname!='creativecommons.org' or u.query or u.fragment or u.port or u.username:raise ValueError('Unsupported licence URL')
 path=u.path.rstrip('/')
 if not (path=='/publicdomain/zero/1.0' or re.fullmatch(r'/licenses/by/(1\.0|2\.0|2\.5|3\.0|4\.0)',path)):raise ValueError('Restricted or unapproved licence')
 return 'https://creativecommons.org'+path+'/'
def photo_id(url):
 u=urlparse(url)
 if u.scheme!='https' or u.hostname not in ('flickr.com','www.flickr.com') or u.query or u.fragment or u.username or u.port:return None
 m=re.fullmatch(r'/photos/([^/]+)/(\d+)/?',u.path)
 return m[2] if m else None
def image_url(row):
 ident=row['image_id'];part=row['source_partition']
 if not re.fullmatch('[0-9a-f]{16}',ident) or part!='validation':raise ValueError('Only exact eligible validation source objects permitted')
 return f'https://open-images-dataset.s3.amazonaws.com/{part}/{ident}.jpg'
def isolation(rows,protected=()):
 owners={k:{} for k in ('image_id','source_identity','identity_group','metadata_original_md5','photographer_identity','sha256','decoded_pixel_sha256')}
 for r in rows:
  if r['proposed_split'] not in ('train','development'):raise ValueError('Frozen or invalid partition')
  for k,seen in owners.items():
   v=r.get(k)
   if not v:continue
   if v in protected:raise ValueError('Protected identity '+k)
   if v in seen and (k in ('image_id','source_identity','sha256','decoded_pixel_sha256') or seen[v]!=r['proposed_split']):raise ValueError('Duplicate/leaked '+k)
   seen[v]=r['proposed_split']
 return True

def parse_coco(document):
 """Strict official instances schema; missing unsupported classes stay unknown."""
 categories={r['id']:r['name'] for r in document['categories']}
 if len(categories)!=len(document['categories']):raise ValueError('Duplicate COCO category')
 for ident,(source,_) in COCO.items():
  if ident in categories and categories[ident]!=source:raise ValueError('COCO category schema conflict')
 images={r['id']:r for r in document['images']};seen=set();rows=[]
 if len(images)!=len(document['images']):raise ValueError('Duplicate COCO image')
 for ann in document['annotations']:
  if ann['id'] in seen or ann['image_id'] not in images or ann['category_id'] not in categories:raise ValueError('Invalid COCO references')
  seen.add(ann['id']);im=images[ann['image_id']]
  b=coco_box(ann['bbox'],im['width'],im['height'])
  if ann['category_id'] not in COCO:continue
  if ann.get('iscrowd',0) not in (0,1):raise ValueError('Invalid COCO crowd flag')
  rows.append(dict(image_id=ann['image_id'],source_category_id=ann['category_id'],label=COCO[ann['category_id']][1],box=b,ignore=bool(ann.get('iscrowd',0)),original_annotation=ann,
   taxonomy_note='dining table is a strict subtype of broad table' if ann['category_id']==67 else 'source native class; cross-dataset equivalence requires separate stratum',annotation_license='https://creativecommons.org/licenses/by/4.0/'))
 return rows
