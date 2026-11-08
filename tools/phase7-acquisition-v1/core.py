"""Isolated Phase7 LVIS metadata acquisition contracts; no photograph client."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/lvis/metadata/phase7-v1'
PROPOSAL=ROOT/'reports/phase7-data-strategy-v1/next-approval-request.json'
PIN='5a60c03bb32250a4c5feec3b873b625eeb54b886655260ed993e7b7ddc68503c'
CAP=850000000
DISK=1073741824
HERE=Path(__file__).resolve().parent
CORE={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}
SUBTYPES={19:'armchair',170:'bunk_bed',285:'coffee_table',301:'sofa_bed',346:'cupboard',358:'deck_chair',367:'dining_table',438:'file_cabinet',464:'folding_chair',558:'highchair',610:'kitchen_table',627:'lamppost',628:'lampshade',731:'oil_lamp',889:'rocking_chair',1051:'table_lamp'}
RELEVANT=set(CORE)|set(SUBTYPES)

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,value):
 p=Path(p);tmp=p.with_name(p.name+'.pending')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def commit(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def allocated():
 return sum(p.lstat().st_blocks*512 for base in (OUT,HERE) for p in [base,*base.rglob('*')] if p.exists())
def space(extra=0):
 if allocated()+extra+32*2**20>DISK:raise RuntimeError('1GiB allocation cap/reserve')
 if os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize<extra+32*2**20:raise RuntimeError('Insufficient disk space')
def proposal():
 if sha(PROPOSAL)!=PIN:raise ValueError('Frozen proposal SHA changed')
 p=read(PROPOSAL)
 if p['expected_body_bytes']!=414291789 or p['max_cumulative_body_bytes']!=CAP or p['max_new_allocated_storage']!=DISK or len(p['objects'])!=2:raise ValueError('Approved proposal changed')
 for r,split,size in zip(p['objects'],('train','val'),(350264821,64026968)):
  if r['url']!=f'https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{split}.json.zip' or r['bytes']!=size or r['expected_local_filename']!=f'lvis_v1_{split}.json.zip':raise ValueError('Unapproved object')
 return p
