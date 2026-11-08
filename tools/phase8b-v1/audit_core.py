"""Exact Phase8B approval boundaries; metadata evidence only, no photograph GET."""
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'data/lvis/metadata/phase7-v1'
OUT=ROOT/'data/lvis/rights-audits/phase8a-v1'
MANIFEST=ROOT/'reports/phase8a-v1/targeted-rights-candidates-v1.json'
PIN='023a7860affd18270d12e931528148ad32a029bd5787ec1fdc916c09ee212d29'
CAP=32000000
DISK=134217728
PAGE=1048576
DECODED=4194304
CORE={181:'cabinet',626:'lamp',77:'bed',361:'desk',982:'sofa',232:'chair',1050:'table'}
STATUSES=('verified_commercially_compatible','potentially_compatible_but_not_independently_verified','restricted_or_incompatible','temporarily_unavailable_or_rate_limited','conflicting_or_insufficient_evidence')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def allocated():return sum(p.lstat().st_blocks*512 for base in (OUT,HERE) for p in [base,*base.rglob('*')] if p.exists())
class ResourceLimit(RuntimeError):pass
class HostSuspended(RuntimeError):pass
def space(extra=0):
 if allocated()+extra+16*2**20>DISK:raise ResourceLimit('128MiB allocated storage limit/reserve')
 if os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize<extra+16*2**20:raise ResourceLimit('Insufficient free disk')
def atomic(p,value):
 space(2*PAGE);p=Path(p);temp=p.with_name(p.name+'.pending')
 with temp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def commit(p,value):
 space(2*PAGE)
 with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def manifest():
 if sha(MANIFEST)!=PIN:raise ValueError('Approved44 source manifest changed')
 m=read(MANIFEST);approval=read(ROOT/'reports/phase8a-v1/approval-A-targeted-rights-audit.json')
 if approval['manifest_sha256']!=PIN or approval['max_cumulative_response_body_bytes']!=CAP or approval['max_new_allocated_storage_bytes']!=DISK:raise ValueError('Approved limits changed')
 if len(m['images'])!=44 or len({r['coco_image_id'] for r in m['images']})!=44 or len({r['source_photo_id'] for r in m['images']})!=44:raise ValueError('44 unique IDs required')
 if dict(collections.Counter(r['proposed_partition'] for r in m['images']))!={'train':8,'development':36}:raise ValueError('Frozen8/36 allocation changed')
 prior=read(BASE/'proposed-photo-rights-audit-v1.json')['images']
 if {r['coco_image_id'] for r in prior}&{r['coco_image_id'] for r in m['images']}:raise ValueError('Previous400 or paused296 overlap')
 for r in m['images']:
  photo=r['source_photo_id'].removeprefix('flickr-photo:')
  if not photo.isdigit() or r['source_page_resolver']!='https://www.flickr.com/photo.gne?id='+photo:raise ValueError('Unapproved source ID/resolver')
  if r['official_image_object_url']!=f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{r['coco_image_id']:012d}.jpg":raise ValueError('Unapproved image object')
  # In-memory adapter only: immutable approved file and native records stay unchanged.
  r.update(metadata_license_url=r['metadata_license_claim'],original_source_asset_url=r['original_flickr_asset_url'],
   annotation_license='LVIS v1 annotations CC BY4.0; underlying photo licence separate',
   annotation_archive_sha256=m['source_archive_sha256'],original_lvis_furniture_annotations=r['original_LVIS_core_annotations'],
   commercial_rights_status='potentially_compatible_but_individually_unverified')
 return m

def history_graph():
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase8b_historical_graph',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 uf,protected,history,_=m.identity_graph();anchors=set(protected)
 for r in read(ROOT/'data/lvis/rights-audits/phase7-v1/rights-audit.json')['records']:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
 for r in read(BASE/'proposed-photo-rights-audit-v1.json')['images']:anchors.add(r['source_photo_id'])
 return uf,{uf.find(a) for a in anchors},history
