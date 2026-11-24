"""Isolated exact-seven photo inspection authorization, CPU only."""
import datetime,hashlib,json,os,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/licensed-rooms/image-pilots/phase10e11-wellcome-v1';REPORT=ROOT/'reports/phase10e11-v1'
APPROVAL=ROOT/'reports/phase10e10-v1/next-photo-approval-request.json'
MANIFEST=ROOT/'reports/phase10e10-v1/potential-training-acquisition.json'
MANIFEST_SHA='53ce3a31a7d85dae17b374746f7705fd814f968cea3774834d63600b4c824be6'
IDS=('efqgz854','bpxzpgg7','x6v3kwb7','y4n8bmat','vugqdsr7','mhnzfe6b','mqxh737p')
CAP=8_000_000;DISK=64*2**20;IMAGE_CAP=512000
UA='FurnitureAI-SuitabilityPilot/10E.11 (https://github.com/dingma-lang/furnitureai; approved seven-image inspection)'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rows():
    if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Approved seven-photo manifest changed')
    a=read(APPROVAL)
    if tuple(a['exact_ids'])!=IDS or a['manifest_SHA256']!=MANIFEST_SHA:raise ValueError('Approval identity mismatch')
    r=read(MANIFEST)['records'];supported={r['source_image_id']:r for r in read(ROOT/'reports/phase10e10-v1/rights-supported-image-inventory.json')['records']}
    if tuple(x['source_image_id'] for x in r)!=IDS or len(supported)!=76:raise ValueError('Exact approved seven and supported76 required')
    for x in r:
        if x['source_image_id'] not in supported or x['rights_status']!='A_rights_supported' or supported[x['source_image_id']]['image_metadata_URL']!=x['image_metadata_URL']:raise ValueError('Unsupported source')
    return r
def allocated():
    seen=set();total=0
    for root in (OUT,REPORT,HERE):
        for p in ([root,*root.rglob('*')] if root.exists() else []):
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total
def space(extra=0):
    if allocated()+extra+4*2**20>DISK:raise RuntimeError('64MiB allocation cap/reserve')
def atomic(p,b):
    p=Path(p)
    if p.exists() or not any(p.resolve().is_relative_to(root) for root in (OUT,REPORT)):raise PermissionError('New E11 files only, no replacement')
    space(2*len(b)+65536);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.pending-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.link(tmp,p)
def emit(p,x):atomic(p,(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def metadata_url(id):
    if id not in IDS:raise ValueError('Unapproved image ID')
    return 'https://api.wellcomecollection.org/catalogue/v2/images/'+id+'?include=source.contributors%2Csource.genres%2Csource.subjects'
def rights_validate(current,old):
    if current.get('id')!=old['source_image_id'] or current.get('source',{}).get('id')!=old['source_work_id']:raise ValueError('Changed photograph/work identity')
    locations=current.get('locations',[])
    if not locations:raise ValueError('Missing source licence')
    for l in locations:
        licence=l.get('license',{}).get('url','').replace('http://','https://').rstrip('/')
        if licence!='https://creativecommons.org/licenses/by/4.0':raise ValueError('Changed/missing licence')
        if l.get('url')!=old['image_metadata_URL'] or l.get('credit') not in old['credit']:raise ValueError('Changed source or credit')
        if any(a.get('terms') or a.get('status',{}).get('id') not in ('open',None) for a in l.get('accessConditions',[])):raise ValueError('Access or use restrictions unresolved')
    prior=[x['name'] for x in old['declared_contributors']];names=[c.get('agent',{}).get('label') for c in current.get('source',{}).get('contributors',[])]
    if sorted(prior)!=sorted(names):raise ValueError('Creator evidence changed; do not auto-upgrade')
    return dict(status='source_supported_CC_BY4_conditional',creator=old['photographer'],rights_holder=old['rights_holder'],credit=old['credit'],
       licence_URL='https://creativecommons.org/licenses/by/4.0/',source_work=old['canonical_source_url'],ownership_and_noncopyright_rights_not_warranted=True,checked_UTC=now())
