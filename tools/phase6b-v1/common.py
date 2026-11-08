"""Phase 6B isolated metadata-only contracts."""
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/coco/metadata/phase6b-v1'
PROPOSAL=ROOT/'data/open-images/image-pilots/phase6a-v1/expansion-proposal-v1/coco-metadata-acquisition-proposal.json'
URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
SIZE=252907541
SHA='113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ETAG='"f4bbac642086de4f52a3fdda2de5fa2c"'
TRANSFER=520000000
STORAGE=1073741824
EXTRACT_MAX=650000000
DERIVED_MAX=100000000
MEMBERS=('annotations/instances_train2017.json','annotations/instances_val2017.json')
MAP={62:('chair','chair'),63:('couch','sofa'),65:('bed','bed'),67:('dining table','table')}
SEED='phase6b-coco-2017-metadata-v1-20261105'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def atomic(p,v):
 p=Path(p);temp=p.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 sync_dir(p.parent)
def sync_dir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def commit(p,v):
 """Publish a complete exclusive marker atomically; never replace old records."""
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.pending-'+uuid.uuid4().hex)
 with temp.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 try:os.link(temp,p);sync_dir(p.parent)
 finally:temp.unlink()
 sync_dir(p.parent)
def allocated(root=OUT):return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file())
def space(extra=0,root=OUT):
 # Include implementation and task temporary files through a conservative
 # 16MiB allowance outside the metadata destination.
 if allocated(root)+extra+16*2**20>STORAGE:raise RuntimeError('1GiB allocated-storage cap')
def approved():
 m=read(PROPOSAL)
 if m['destination']!=str(OUT) or m['max_cumulative_transfer']!=TRANSFER or m['max_new_allocated_storage']!=STORAGE or m['extract_only']!=list(MEMBERS):raise ValueError('Approved resource/member scope changed')
 if len(m['objects'])!=1 or any(m['objects'][0][k]!=v for k,v in dict(url=URL,bytes=SIZE,etag=ETAG,sha256=SHA).items()):raise ValueError('Approved source changed')
 return m
