"""Approved27 LVIS-associated photographs only. No model/rights-retry interface."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RIGHTS=ROOT/'data/lvis/rights-audits/phase8a-v1'
META=ROOT/'data/lvis/metadata/phase7-v1'
MANIFEST=RIGHTS/'proposed-image-acquisition-v1.json'
PIN='b92ca092b47e025b477892a7fc69358b54ab30b520ae3af15a3bf1e2af3ad46b'
OUT=ROOT/'data/lvis/image-pilots/phase8b-v1'
CAP=100000000
DISK=536870912
MAX_IMAGE=3145728
CLASSES={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def commit(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending-'+uuid.uuid4().hex)
 with tmp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p)
 finally:tmp.unlink()
def atomic(p,value):
 p=Path(p);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def allocated(base=OUT):return sum(p.lstat().st_blocks*512 for p in [base,*base.rglob('*')] if p.exists())
def space(extra=0):
 if allocated()+allocated(HERE)+extra+16*2**20>DISK:raise RuntimeError('512MiB allocation cap/reserve')
 if OUT.exists() and os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize<extra+16*2**20:raise RuntimeError('Insufficient disk space')
def manifest():
 if sha(MANIFEST)!=PIN:raise ValueError('Approved27 manifest SHA changed')
 m=read(MANIFEST);rows=m['images']
 if len(rows)!=27 or len({r['coco_image_id'] for r in rows})!=27 or sum(r['proposed_partition']=='train' for r in rows)!=7 or sum(r['proposed_partition']=='development' for r in rows)!=20:raise ValueError('Approved IDs/allocation changed')
 normalized=[]
 for original in rows:
  r=dict(original);proof=r['verified_photo_rights'];obj=r['remote_object'];iid=r['coco_image_id']
  if r['current_rights_status']!='verified_commercially_compatible' or proof['license_url']!='https://creativecommons.org/licenses/by/2.0/' or r['identity_decision']['identity_flags']:raise ValueError('Unverified or identity-quarantined row')
  if obj['url']!=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{iid:012d}.jpg' or not 0<obj['bytes']<=MAX_IMAGE:raise ValueError('Unapproved image object')
  r.update(image_id=f'coco2017:{iid:012d}',source_identity=r['source_photo_id'],identity_group=r['identity_decision']['known_source_group'],proposed_split=r['proposed_partition'],
   rights_record=proof['rights_record'],rights_record_sha256=proof['rights_record_sha256'],source_document=proof['evidence_file'],source_document_sha256=proof['evidence_sha256'],rights_checked_utc=proof['evidence_checked_utc'],
   original_furniture_annotations=r['original_lvis_furniture_annotations'],source_image_scope=r['native_scope'])
  normalized.append(r)
 if sum(r['remote_object']['bytes'] for r in rows)!=3913471:raise ValueError('Approved size sum changed')
 return dict(m,images=normalized)
