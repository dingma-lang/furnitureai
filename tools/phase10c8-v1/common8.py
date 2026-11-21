"""Exactly two matched arms; source evidence and partial-label protocol pinned."""
from pathlib import Path
import sys,os,uuid,hashlib,random
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from common6 import read,save,sha,digest,manifest,optimizer_member,CLASSES,ADMISSION,TAXONOMY,candidates,registry,overlap
RUN=ROOT/'data/open-images/experiments/phase10c8-ab-v2'
REPORT=ROOT/'reports/phase10c8-v2'
BASE=ROOT/'models/owlv2-original-baseline-20261103'
SOURCE=ROOT/'data/open-images/preparation/phase10c6-v1'
ARMS=('A-control','B-corrected')
MAX_VRAM=int(6.5*2**30)
PROMPTS=['a photo of a '+c for c in CLASSES]
PROBE_EPOCHS=(0,1,3,5)
EVAL_EPOCHS=(1,3,5)

def grouped_epoch(rows,epoch):
    order=list(rows);random.Random(6106+epoch).shuffle(order)
    return [order[i:i+4] for i in range(0,len(order),4)]

def subsets(prepared=None):
    p=manifest() if prepared is None else prepared;train=[r for r in p['rows'] if optimizer_member(r)];dev=[r for r in p['rows'] if r['historical_partition']=='train' and r['internal_partition']=='inner_development']
    if len(train)!=59 or len(dev)!=19 or len({r['identity_group'] for r in train})!=56 or sum(not a['ignore'] for r in train for a in r['annotations'])!=182:raise ValueError('Pinned data counts changed')
    for key in ('image_id','sha256','identity_group'):
        if {r[key] for r in train}&{r[key] for r in dev}:raise ValueError('Train/development leakage')
    return train,dev

def image_path(row,part,prepared):
    index={r['image_id']:r for r in prepared['rows']}
    if row!=index.get(row['image_id']) or row['historical_partition']!='train' or row['internal_partition']!=part or part not in ('optimization','inner_development'):raise PermissionError('Not an authorized exact training-derived image')
    p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Image bytes/path changed')
    return p

def verify_binding(b):
    if b.get('training_authorized') is not True or b.get('arms')!=list(ARMS) or b.get('max_updates_per_arm')!=75 or b.get('max_total_updates')!=150 or b.get('max_epochs')!=5 or b.get('max_queue_seconds')!=3600 or b.get('max_vram_bytes')!=MAX_VRAM:raise PermissionError('Specific Phase10C.8 authorization/budget required')
    for p,h in b['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('Pinned source/code/model changed: '+p)
    prepared=manifest();subsets(prepared);return prepared

def parameter_hash(model,frozen):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if (not p.requires_grad)==frozen:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def save_npz(path,arrays):
    import numpy as np
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite raw outputs')
    tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        if tmp.stat().st_size>600000:raise ValueError('Raw image-output bound')
        os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def load_npz(path):
    import numpy as np,zipfile
    if path.stat().st_size>600000:raise ValueError('Oversized raw output')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) not in (2,5) or sum(i.file_size for i in z.infolist())>600000:raise ValueError('Unsafe raw output archive')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if set(a) not in ({'logits','square_cxcywh'},{'logits','square_cxcywh','scores','labels','boxes'}) or a['logits'].shape!=(3600,7) or a['square_cxcywh'].shape!=(3600,4) or any(v.dtype!=np.dtype('int64' if k=='labels' else 'float32') or not np.isfinite(v).all() for k,v in a.items()):raise ValueError('Raw output shape/dtype mismatch')
    if 'scores' in a and (a['scores'].shape!=(3600,) or a['labels'].shape!=(3600,) or a['boxes'].shape!=(3600,4) or (a['labels']<0).any() or (a['labels']>6).any()):raise ValueError('Evaluation postprocessing shape mismatch')
    return a
