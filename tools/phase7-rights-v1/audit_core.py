"""Phase7 rights-only boundaries; no model, photograph GET or split-mutation interface."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'data/lvis/metadata/phase7-v1'
OUT=ROOT/'data/lvis/rights-audits/phase7-v1'
MANIFEST=BASE/'proposed-photo-rights-audit-v1.json'
PIN='15340ef1f78c1f34c6cb9be10fd3b7d6e5d07528bc393f514f6f4a5ef89e014f'
CAP=128000000
DISK=536870912
PAGE=1048576
DECODED=4194304
CORE={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}
STATUSES=('verified_commercially_compatible','potentially_compatible_but_not_independently_verified','restricted_or_incompatible','temporarily_unavailable_or_rate_limited','conflicting_or_insufficient_evidence')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,value):
 p=Path(p);temp=p.with_name(p.name+'.pending')
 with temp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def commit(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def allocated():
 return sum(p.lstat().st_blocks*512 for base in (OUT,HERE) for p in [base,*base.rglob('*')] if p.exists())
def space(extra=0):
 if allocated()+extra+16*2**20>DISK:raise ResourceLimit('512MiB allocated storage limit/reserve')
 if os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize<extra+16*2**20:raise ResourceLimit('Insufficient disk space')
class ResourceLimit(RuntimeError):pass
class HostSuspended(RuntimeError):pass

def manifest():
 if sha(MANIFEST)!=PIN:raise ValueError('Approved400 source manifest changed')
 m=read(MANIFEST);p=read(BASE/'next-approval-request.json')
 if p['manifest_sha256']!=PIN or p['max_cumulative_response_body_bytes']!=CAP or p['max_new_allocated_storage']!=DISK or m['proposed_allocation']!={'train':320,'development':80}:raise ValueError('Approved scope mismatch')
 if len(m['images'])!=400 or len({r['coco_image_id'] for r in m['images']})!=400 or len({r['source_photo_id'] for r in m['images']})!=400:raise ValueError('Repeated/missing source IDs')
 for r in m['images']:
  if not r['source_photo_id'].startswith('flickr-photo:') or not r['source_photo_id'].split(':')[1].isdigit():raise ValueError('Invalid original photo identity')
  if r['source_page_resolver']!='https://www.flickr.com/photo.gne?id='+r['source_photo_id'].split(':')[1]:raise ValueError('Unapproved resolver')
  if r['official_image_object_url']!=f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{r['coco_image_id']:012d}.jpg":raise ValueError('Unapproved photograph object')
 return m
