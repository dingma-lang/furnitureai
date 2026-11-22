"""One authorized zero-update diagnostic; no optimizer/training entry point."""
import json,os,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c14-v1'))
from common14 import *
HERE15=Path(__file__).resolve().parent
RUN15=ROOT/'data/open-images/experiments/phase10c15-source-negative-gradient-smoke-v1'
REPORT15=ROOT/'reports/phase10c15-v1'
MAX_BYTES=128*2**20
MAX_VRAM=13*2**29
PARENT='109275b31cdc26ce0e2793ca6464971d08f35b6a'
BRANCH='experiment/phase10c15-negative-gradient-smoke'


def allocated():
    return sum(p.stat().st_blocks*512 for folder in (RUN15,REPORT15) if folder.exists() for p in folder.rglob('*'))


def atomic_bytes(path,payload):
    path=Path(path)
    if path.exists():raise FileExistsError('Immutable diagnostic record already exists: '+str(path))
    if not path.resolve().is_relative_to(RUN15) and not path.resolve().is_relative_to(REPORT15):raise PermissionError('Diagnostic output path outside approved roots')
    # Pending file is deliberately retained: no cleanup/deletion, no overwrite.
    if allocated()+2*((len(payload)+4095)//4096)*4096+8192>MAX_BYTES:raise RuntimeError('128MiB diagnostic allocation boundary')
    temp=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
    with temp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.link(temp,path)


def emit(path,value):atomic_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def inputs15():
    cfg=read(RUN15/'authorized-config.json')
    if cfg.get('smoke_authorized') is not True or cfg.get('training_authorized') is not False or cfg['limits']!=dict(forwards=24,seconds=180,VRAM_bytes=MAX_VRAM,output_bytes=MAX_BYTES,optimizer_updates=0):raise PermissionError('Specific zero-update smoke approval required')
    for relative,expected in cfg['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Pinned smoke input/code changed: '+relative)
    prepared,train,dev,native,registry=training_context();byid={r['image_id']:r for r in train}
    rows=[byid[i] for i in cfg['photo_ids']]
    if len(rows)!=24 or len(set(cfg['photo_ids']))!=24 or len({r['identity_group'] for r in rows})!=24:raise ValueError('Exact24 training probes/groups required')
    for row in rows:require_training_row(row,train,dev)
    return cfg,prepared,rows,registry
