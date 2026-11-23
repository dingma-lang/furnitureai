"""CPU-only E6 roots;no model/inference/downloader entry point."""
import hashlib,io,json,os,subprocess,sys,threading,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/open-images/preparation/phase10e6-v1'
REPORT=ROOT/'reports/phase10e6-v1'
sys.path.insert(0,str(ROOT/'tools/phase10e1-v1'))
from commonE import PREP,BASE,BASE_SHA,CLASSES,PROMPTS,read,sha,digest
CAP=512*2**20
LOCK=threading.RLock()

def atomic(path,data):
    path=Path(path)
    with LOCK:
        if path.exists() or not any(path.resolve().is_relative_to(r) for r in (DATA,REPORT)):raise PermissionError('New E6 artifacts only')
        used=sum(p.stat().st_blocks*512 for r in (DATA,REPORT) if r.exists() for p in [r,*r.rglob('*')])
        if used+2*len(data)+65536>CAP:raise RuntimeError('512MiB self-imposed CPU output limit')
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
def emit(path,value):atomic(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())

def context():
    c=read(PREP/'training-context.json');train={e['row']['image_id']:e['row'] for e in c['entries']};dev=c['development']
    if len(train)!=59 or len(dev)!=19:raise ValueError('Expected59optimization/19internal rows')
    for k in ('image_id','sha256','identity_group'):
        if {r[k] for r in train.values()}&{r[k] for r in dev}:raise ValueError('Partition/group overlap')
    return c,train,dev

def require_train(ident,train):
    row=train.get(ident)
    if row is None or row['historical_partition']!='train' or row['internal_partition']!='optimization':raise PermissionError('Only existing optimization images may be mined')
    return row

def registered(boxes,w,h):
    import numpy as np
    b=np.asarray(boxes,dtype=np.float64)
    return np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)*np.asarray([max(w,h)/w,max(w,h)/h]*2)

def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def overlap(a,b):return intersection(a,b)>0
def iou(a,b):
    hit=intersection(a,b);union=area(a)+area(b)-hit
    return hit/union if union else 0.

def semantic(ann):
    s=ann['native_label'].lower().replace('_',' ')
    if any(v in s for v in ('chair','stool','seat','ottoman','bench','beanbag')):return 'seating_or_ambiguous_seating'
    if ann['part_or_semantics_unresolved'] or any(v in s for v in ('cushion','drawer','headboard','mattress','lampshade','lamp shade','tablecloth','crossbar','knob','hinge')):return 'furniture_fragment_or_ambiguous_part'
    if ann['protect_furniture'] or any(v in s for v in ('sofa','couch','bed','table','desk','cabinet','cupboard','wardrobe','chandelier','countertop','bookcase','bookshelf','lamp','furniture')):return 'other_furniture'
    if any(v in s for v in ('human','person','woman','man','face','hair','arm','mouth','nose','leg','hand','body')) and s not in ('handbag',):return 'person_or_body_part'
    if any(v in s for v in ('plant','flower','tree','vase','decoration','wreath','candle','picture','mirror','tapestry')):return 'plant_or_decoration'
    if any(v in s for v in ('television','tv','screen','monitor','laptop','computer','appliance','stove','toaster','microwave','refrigerator','fan','heater','conditioner')):return 'appliance_or_screen'
    return 'other_native_object'
