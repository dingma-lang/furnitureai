"""Phase6C exact-ID, metadata-only authorization contracts."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse,parse_qs
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'data/coco/metadata/phase6b-v1'
MANIFEST=BASE/'proposed-image-candidates-v1.json'
PROPOSAL=BASE/'next-rights-audit-proposal.json'
PIN='4dd2925357784f18f3aa5cb9ee2e3847235fd7bf7e32bc9fbac9063010bb1b0a'
OUT=ROOT/'data/coco/rights-audits/phase6c-v1'
CAP=256000000
DISK=536870912
PAGE=1048576
DECODED=4194304
PHOTO_MAX=3145728
STATUSES=('verified_commercially_compatible','potentially_compatible_but_unverified','restricted_or_incompatible','source_unavailable','conflicting_licensing_evidence')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
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
def manifest():
 if sha(MANIFEST)!=PIN:raise ValueError('Frozen400-ID manifest changed')
 m=read(MANIFEST);p=read(PROPOSAL)
 if len(m['images'])!=400 or p['candidate_manifest_sha256']!=PIN or p['max_cumulative_body_transfer']!=CAP or p['max_new_allocated_storage']!=DISK:raise ValueError('Approved scope changed')
 if len({r['source_identity'] for r in m['images']})!=400:raise ValueError('Repeated source identity')
 if sum(r['proposed_split']=='train' for r in m['images'])!=320 or sum(r['proposed_split']=='development' for r in m['images'])!=80:raise ValueError('Allocation changed')
 for r in m['images']:
  photo=r['source_identity'].removeprefix('flickr-photo:')
  if not photo.isdigit() or r['proposed_source_page_resolver']!='https://www.flickr.com/photo.gne?id='+photo:raise ValueError('Unapproved source ID')
  if r['proposed_secure_image_url']!=f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{r['coco_image_id']:012d}.jpg":raise ValueError('Unapproved image object')
 return m
def account(url):
 u=urlparse(url);parts=u.path.strip('/').split('/')
 if u.scheme not in ('http','https') or u.hostname not in ('flickr.com','www.flickr.com') or len(parts)<2 or parts[0] not in ('photos','people'):return None
 return 'flickr-account:'+parts[1].casefold()
def canonical_id(url):
 u=urlparse(url)
 if u.scheme!='https' or u.hostname not in ('flickr.com','www.flickr.com') or u.port or u.username or u.query or u.fragment:return None
 m=re.fullmatch(r'/photos/[^/]+/(\d+)/?',u.path)
 return m[1] if m else None
def allowed_page(url,photo):
 u=urlparse(url)
 if canonical_id(url)==photo:return True
 query=parse_qs(u.query)
 resolver_query=query=={'id':[photo]} or query=={'id':[photo],'rb':['1']}
 return u.scheme=='https' and u.hostname in ('flickr.com','www.flickr.com') and u.path=='/photo.gne' and resolver_query and not (u.port or u.username or u.fragment)
def license_kind(url):
 u=urlparse(url)
 if u.scheme not in ('http','https') or u.hostname!='creativecommons.org' or u.port or u.username or u.query or u.fragment:return ('unknown',None)
 p=u.path.rstrip('/')
 if p=='/publicdomain/zero/1.0':return ('allowed','https://creativecommons.org'+p+'/')
 m=re.fullmatch(r'/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(1\.0|2\.0|2\.5|3\.0|4\.0)',p)
 if not m:return ('unknown',None)
 return ('allowed' if m[1]=='by' else 'outside_policy','https://creativecommons.org'+p+'/')
