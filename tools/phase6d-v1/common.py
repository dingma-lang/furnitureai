"""Phase6D exact72-photo acquisition/screening contracts; no model interface."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RIGHTS=ROOT/'data/coco/rights-audits/phase6c-v1'
MANIFEST=RIGHTS/'proposed-image-acquisition-v1.json'
PIN='bc6722b2b603e88e437961c2a8c7b8f6d5dfe16cd07141ee8b478d5ef8390725'
OUT=ROOT/'data/coco/image-pilots/phase6d-v1'
CAP=100000000
DISK=536870912
MAX_IMAGE=3145728
CLASSES={62:'chair',63:'sofa',65:'bed',67:'dining_table'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def commit(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending-'+uuid.uuid4().hex)
 with tmp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p)
 finally:tmp.unlink()
def atomic(p,value):
 p=Path(p);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def allocated(root=OUT):return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file())
def space(extra=0):
 if allocated()+allocated(HERE)+extra+16*2**20>DISK:raise RuntimeError('Approved512MiB storage limit')
def manifest():
 if sha(MANIFEST)!=PIN:raise ValueError('Frozen72-photo manifest changed')
 m=read(MANIFEST);rows=m['images']
 if len(rows)!=72 or len({r['image_id'] for r in rows})!=72 or sum(r['proposed_split']=='train' for r in rows)!=60 or sum(r['proposed_split']=='development' for r in rows)!=12:raise ValueError('Approved membership/allocation changed')
 for r in rows:
  if not r['license_verified'] or r['rights_status']!='verified_commercially_compatible' or r['current_license_url']!='https://creativecommons.org/licenses/by/2.0/':raise ValueError('Unverified licensing')
  if r['remote_object']['url']!=f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{r['coco_image_id']:012d}.jpg" or not 0<r['remote_object']['bytes']<=MAX_IMAGE:raise ValueError('Unapproved source object')
 if sum(r['remote_object']['bytes'] for r in rows)!=10853122:raise ValueError('Approved expected resource budget changed')
 return m
