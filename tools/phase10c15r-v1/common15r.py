"""One separately approved C15 recovery;historical attempt remains immutable."""
import sys,os,json,time,uuid,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c15-v1'))
from common15 import *
PRIOR_RUN=RUN15
PRIOR_REPORT=REPORT15
HERER=Path(__file__).resolve().parent
RUNR=ROOT/'data/open-images/experiments/phase10c15-negative-gradient-recovery-v1'
REPORTR=ROOT/'reports/phase10c15r-v1'
PARENT_R='f77d77403a65391f88369d9e17925d6127d68666'
BRANCH_R='experiment/phase10c15-negative-gradient-smoke'
WRITE_LOCK=threading.RLock()


def allocated_r():
    return sum(p.stat().st_blocks*512 for folder in (RUNR,REPORTR) if folder.exists() for p in folder.rglob('*'))


def atomic_r(path,payload):
    with WRITE_LOCK:
        path=Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (RUNR,REPORTR)):raise PermissionError('Outside approved recovery outputs')
        if path.exists():raise FileExistsError('Immutable recovery record already exists')
        if allocated_r()+2*((len(payload)+4095)//4096)*4096+8192>MAX_BYTES:raise RuntimeError('128MiB new diagnostic output boundary')
        temp=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with temp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.link(temp,path) # Retain pending link;no overwriting/cleanup of any artifact.


def emit_r(path,value):atomic_r(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def validate_indices(indices):
    if indices!=list(range(5,25)) or any(type(i) is not int for i in indices):raise PermissionError('Only exact photo indices5through24 authorized')


def verify_prior():
    public=verify_seal(PRIOR_REPORT/'SHA256SUMS');meta=read(PRIOR_REPORT/'artifact-seal.json');manifest=ROOT/meta['private_manifest']
    if sha(manifest)!=meta['private_manifest_sha256']:raise ValueError('Prior private manifest changed')
    sealed=read(manifest)
    for relative,expected in sealed['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Prior attempt artifact changed: '+relative)
    for relative,expected in meta['new_code_sha256'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Sealed C15 repair changed: '+relative)
    if read(PRIOR_RUN/'failure.json')['forwards']!=5 or (PRIOR_RUN/'complete.json').exists():raise ValueError('Prior consumed attempt status changed')
    prior=read(PRIOR_RUN/'authorized-config.json');records=[]
    for i,ident in enumerate(prior['photo_ids'][:4],1):
        p=PRIOR_RUN/(str(i).zfill(2)+'.json');r=read(p);raw=PRIOR_RUN/r['raw_file']
        if r['image_id']!=ident or sha(raw)!=r['raw_sha256'] or r['actual_forward']!=i or not r['parameter_grads_all_none'] or r['counts']['protected_violations']:raise ValueError('Prior completed record invalid')
        records.append(dict(index=i,image_id=ident,file=str(p.relative_to(ROOT)),sha256=sha(p),raw_file=str(raw.relative_to(ROOT)),raw_sha256=sha(raw)))
    return prior,records,dict(public_files=public,private_files=len(sealed['files']))


def inputs_r():
    cfg=read(RUNR/'authorized-config.json');validate_indices(cfg['photo_indices'])
    if cfg.get('recovery_authorized') is not True or cfg.get('training_authorized') is not False or cfg['limits']!=dict(new_forwards=20,cumulative_forwards=25,seconds=180,VRAM_bytes=MAX_VRAM,output_bytes=MAX_BYTES,optimizer_updates=0):raise PermissionError('Specific one-attempt recovery approval required')
    for relative,expected in cfg['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Recovery pinned input/code changed: '+relative)
    prepared,train,dev,_,reg=training_context();byid={r['image_id']:r for r in train};rows=[byid[cfg['photo_ids'][i-1]] for i in cfg['photo_indices']]
    if len(rows)!=20 or len({r['identity_group'] for r in rows})!=20:raise ValueError('Exact20 source-isolated optimization photos required')
    for row in rows:require_training_row(row,train,dev)
    return cfg,prepared,rows,reg
