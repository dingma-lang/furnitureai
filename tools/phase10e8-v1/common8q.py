"""CPU-only E8 qualification roots;historical checkpoints/data are read only."""
import hashlib,io,json,os,sys,threading,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/open-images/preparation/phase10e8-v1'
REPORT=ROOT/'reports/phase10e8-v1'
E7=ROOT/'data/open-images/experiments/phase10e6-training-query-cache-v1'
PARENT='ecafe99dc8fd288da151f51dcc00ee3f724889ff'
sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'))
from commonE6 import PREP,BASE,BASE_SHA,CLASSES,PROMPTS,read,sha,digest,context,require_train,registered,area,intersection,overlap,iou,semantic
CAP=256*2**20
LOCK=threading.RLock()
POLICY=dict(version='phase10e8-source-visible-Chair-negative-v1',minimum_side_pixels=32,boundary_margin_pixels=2,
    fully_contained_in_one_simple_visible_source_polygon=True,zero_protected_overlap=True,
    real_query_full_containment_required_for_hard_negative=True,source_and_actual_visual_support_required=True,
    Chair_includes_seating_and_bar_stools=True,footstools_ottomans_stepstools_separate_or_protected=True,
    other_six_classes_unknown=True,missing_annotations_are_not_negatives=True,training_authorized=False)

def allocated():
    return sum(p.stat().st_blocks*512 for root in (DATA,REPORT,HERE) if root.exists() for p in [root,*root.rglob('*')])

def atomic(path,value):
    path=Path(path)
    with LOCK:
        if path.exists() or not any(path.resolve().is_relative_to(root) for root in (DATA,REPORT)):
            raise PermissionError('Fresh E8 outputs only;no replacement')
        if allocated()+2*((len(value)+4095)//4096)*4096+65536>CAP:raise RuntimeError('256MiB bounded CPU output cap')
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with tmp.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def emit(path,value):atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def exact_candidates():
    m=read(E7/'analysis/candidate-manifest.json');wanted=set(m['B_automatic_unverified'])
    rows=[r for r in read(E7/'analysis/inspection-samples.json')['records'] if r['candidate_id'] in wanted]
    if len(rows)!=21 or len({r['candidate_id'] for r in rows})!=21 or {r['candidate_id'] for r in rows}!=wanted or len({r['source_group'] for r in rows})!=10:
        raise ValueError('Exact21frozen E7candidates/10sourcegroups required')
    return sorted(rows,key=lambda r:r['sample_id'])
