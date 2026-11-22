"""CPU-only preparation roots. No implicit model construction or execution."""
import hashlib,io,json,os,sys,time,uuid,threading,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'data/open-images/preparation/phase10e1-v1'
BINDING=PREP/'input-binding-v2.json'
REPORT=ROOT/'reports/phase10e1-v1'
FUTURE=ROOT/'data/open-images/experiments/phase10e-distillation-v1'
SMOKE=ROOT/'data/open-images/experiments/phase10e-gradient-smoke-v1'
BASE=ROOT/'models/owlv2-original-baseline-20261103'
BASE_SHA='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
PARENT='f0e7d5714bde7a84dd2b57022249dae374ef44e1'
BRANCH='experiment/phase10e1-selective-distillation-preflight'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
PROMPTS=['a photo of a '+c for c in CLASSES]
MAX_PREP=128*2**20
LOCK=threading.RLock()
sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
from c10 import read,sha

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def allocated():return sum(p.stat().st_blocks*512 for root in (PREP,REPORT) if root.exists() for p in [root,*root.rglob('*')])
def atomic(path,data):
    path=Path(path)
    with LOCK:
        if not any(path.resolve().is_relative_to(root) for root in (PREP,REPORT)):raise PermissionError('Outside E1 preparation')
        if path.exists():raise FileExistsError('Immutable E1 output exists')
        if allocated()+2*((len(data)+4095)//4096)*4096+8192>MAX_PREP:raise RuntimeError('128MiB preparation bound')
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path)
def emit(path,value):atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def source_context():
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from common21 import context21,SOURCE
    from common6 import candidates
    from masks6 import validate_bundle
    from review21 import semantic_protected
    from g5_common import overlap
    old,prepared,train,dev,reg,_,_=context21()
    native={p['image_id']:p for p in read(ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json')['photos']}
    frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
    bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,observations)
    records=[]
    for row in train:
        ident=row['image_id'];selected=[e for e in bundle['records'] if e['image_id']==ident]
        legacy_protected=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        extra=[a['box'] for a in native[ident]['annotations'] if semantic_protected(a) or a['native_ignored']]
        distill_protected=[a['box'] for a in row['annotations']]+legacy_protected+extra
        admitted=[];excluded=[]
        for e in selected:
            if any(overlap(e['box'],b) for b in distill_protected):excluded.append(dict(region=e['candidate_id'],reason='Conservative native furniture/part/ignored overlap;not a declaration that old evidence was wrong'))
            else:admitted.append(e)
        records.append(dict(row=row,source_regions=selected,legacy_protected=legacy_protected,distill_regions=admitted,distill_protected=distill_protected,excluded_distill_regions=excluded))
    return old,prepared,records,dev,bundle
