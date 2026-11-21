"""Read-only C8 inputs; C9 has no model/optimizer execution entry point."""
from pathlib import Path
import sys,os,json,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase10c8-v1'))
from common8 import read,save,sha,digest,subsets,manifest,load_npz,CLASSES,SOURCE,ARMS
C8=ROOT/'data/open-images/experiments/phase10c8-ab-v2'
OUT=ROOT/'data/open-images/diagnostics/phase10c9-v1'
REPORT=ROOT/'reports/phase10c9-v1'
MODELS={'original':('A-control',0),'control':('A-control',5),'corrected':('B-corrected',5)}
THRESHOLDS=(.15,.5,.8)
TAXONOMY={
 'A':'Genuine non-furniture false positive',
 'B':'Another furniture category misclassified as Chair',
 'C':'Real Chair or seating stool missing from annotations',
 'D':'Duplicate detection of an annotated Chair',
 'E':'Fragmented or poorly localized furniture detection',
 'F':'Uncertain due to incomplete annotations or ambiguous content',
 'G':'Other verified failure mechanism'}

def metric_module():
    import importlib.util
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('c9_frozen_metric',ROOT/'tools/phase10c-v1/metric_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def quantiles(values):
    import numpy as np
    return {str(q):float(np.quantile(values,q)) for q in (0,.1,.5,.9,.99,1)} if len(values) else {}

def source_path(row):
    p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Source-photo identity/hash changed')
    return p

def registered_box(square_cxcywh,w,h,clip=True):
    """Exact inverse of the recorded bottom/right square-padding transform."""
    import numpy as np
    b=np.asarray(square_cxcywh,dtype=np.float64);xy=np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),axis=-1)
    xy=xy*np.asarray([max(w,h)/w,max(w,h)/h]*2)
    return np.clip(xy,0,1) if clip else xy

def geometry(box,w,h):
    width=(box[2]-box[0])*w;height=(box[3]-box[1])*h;area=(box[2]-box[0])*(box[3]-box[1])
    return dict(width_pixels=width,height_pixels=height,area_fraction=area,
        size_bin='<1%' if area<.01 else '1-5%' if area<.05 else '5-20%' if area<.2 else '>=20%',
        tiny_heuristic=min(width,height)<32 or area<.01,border_touching_heuristic=min(box[:2])<=.005 or max(box[2:])>=.995)

def verify_seal(path,root=ROOT):
    n=0
    for line in path.read_text().splitlines():
        expected,relative=line.split('  ',1);p=(root/relative).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=expected:raise ValueError('Historical artifact integrity failure: '+relative)
        n+=1
    return n

def assert_mining_scope(row,train,dev):
    index={r['image_id']:r for r in train}
    if row!=index.get(row['image_id']) or row['historical_partition']!='train' or row['internal_partition']!='optimization':raise PermissionError('Only exact optimization-training rows may be mined')
    for key in ('image_id','sha256','identity_group'):
        if row[key] in {r[key] for r in dev}:raise PermissionError('Development/evaluation leakage')
    return True
