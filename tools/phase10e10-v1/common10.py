"""Bounded metadata-only Phase10E.10; no image or model client."""
import datetime,hashlib,json,os,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/rights-discovery/phase10e10-v1';REPORT=ROOT/'reports/phase10e10-v1'
CAP=32_000_000;DISK=128*2**20;MAX_CANDIDATES=200
UA='FurnitureAI-MetadataDiscovery/10E.10 (https://github.com/dingma-lang/furnitureai; metadata only)'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def allocated():
    seen=set();n=0
    for root in (OUT,REPORT,HERE):
        for p in ([root,*root.rglob('*')] if root.exists() else []):
            s=p.stat();k=(s.st_dev,s.st_ino)
            if k not in seen:seen.add(k);n+=s.st_blocks*512
    return n
def atomic(p,b):
    p=Path(p)
    if not any(p.resolve().is_relative_to(r) for r in (OUT,REPORT)):raise PermissionError('New E10 metadata roots only')
    if p.exists():raise FileExistsError(p)
    if allocated()+2*len(b)+8*2**20>DISK:raise RuntimeError('128MiB cap/reserve')
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.link(tmp,p) # Retain same-inode pending link; never remove historical files.
def emit(p,x):atomic(p,(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
