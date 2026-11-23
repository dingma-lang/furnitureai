"""Exact64IDmetadata/rights authorization;no model or photo acquisition entry point."""
import datetime as dt,hashlib,io,json,os,sys,threading,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/coco/rights-audits/phase10e8-v1'
REPORT=ROOT/'reports/phase10e9-v1'
MANIFEST=ROOT/'data/open-images/preparation/phase10e8-v1/licensed-training-rights-candidates-v1.json'
PIN='50e29c7b64b4ca67b695df5c941e4bb3c930ffe483e25d0de731ac9e22c86466'
PARENT='fa9c22e7fbea0559dd57726a161c4d6e1eaf5689'
CAP=32000000;DISK=128*2**20;PAGE=2**20;DECODED=4*2**20
TLS_REQUEST_CAP=PAGE+256*1024
TRANSACTION_OVERHEAD_RESERVE=65536
UA='FurnitureAI-RightsAudit/10E.9 (user-authorized source-license metadata verification; no image downloads)'
STATUS_A='A_rights_supported';STATUS_B='B_unresolved_rights';STATUS_C='C_unsuitable'
sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'))
from commonE6 import read,sha,digest,CLASSES,BASE,BASE_SHA
LOCK=threading.RLock()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def allocated():
    seen=set();total=0
    for root in (OUT,REPORT,HERE):
        if not root.exists():continue
        for p in [root,*root.rglob('*')]:
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total
def space(extra=0):
    if allocated()+extra+8*2**20>DISK:raise RuntimeError('128MiB new allocated storage limit/reserve')
def atomic(path,data):
    path=Path(path)
    with LOCK:
        if path.exists() or not any(path.resolve().is_relative_to(p) for p in (OUT,REPORT)):raise PermissionError('Fresh E9 rights/report artifacts only')
        space(2*len(data)+65536);path.parent.mkdir(parents=True,exist_ok=True)
        p=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(p,path)
def emit(path,value):atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def manifest():
    if sha(MANIFEST)!=PIN:raise ValueError('Frozen64IDmanifest changed')
    m=read(MANIFEST);r=m['records']
    if len(r)!=64 or len({x['image_id'] for x in r})!=64 or len({x['source_photo_id'] for x in r})!=64:raise ValueError('Exactly64unique approved IDs required')
    for x in r:
        fid=x['source_photo_id'].removeprefix('flickr-photo:');cid=x['image_id'].removeprefix('coco2017:')
        if not fid.isdigit() or len(cid)!=12 or not cid.isdigit() or x['original_source_page_resolver']!='https://www.flickr.com/photo.gne?id='+fid:
            raise ValueError('Unapproved source/photo identity')
        if x['official_image_url']!='https://s3.amazonaws.com/images.cocodataset.org/train2017/'+cid+'.jpg':raise ValueError('Unapproved official image URL')
    return m
