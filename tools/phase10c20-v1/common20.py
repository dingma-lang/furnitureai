"""One C20 recovery attempt; original C18/C19 inputs remain immutable."""
import sys,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c19-v1'))
from common19 import *
HERE20=Path(__file__).resolve().parent
RUN20=ROOT/'data/open-images/experiments/phase10c19-recovery-v1'
REPORT20=ROOT/'reports/phase10c20-v1'
PARENT20='d34d73634bdb2d4c8a738029803bbe23c3d2d8c9'
BRANCH20='experiment/phase10c20-source-balanced-recovery'
MAX_OUTPUT20=int(2.5*2**30)
LOCK20=threading.RLock()


def allocation20():
    return sum(p.stat().st_blocks*512 for folder in (RUN20,REPORT20) if folder.exists() for p in folder.rglob('*'))


def atomic20(path,data):
    with LOCK20:
        path=Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (RUN20,REPORT20)):raise PermissionError('Outside C20 output roots')
        if path.exists():raise FileExistsError('Immutable C20 output exists')
        if allocation20()+2*((len(data)+4095)//4096)*4096+8192>MAX_OUTPUT20:raise RuntimeError('C20 2.5GiB allocation bound')
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path)


def emit20(path,value):atomic20(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def verify19_seals():
    count=verify_seal(REPORT19/'SHA256SUMS');s=read(REPORT19/'artifact-seal.json');p=ROOT/s['private_manifest']
    if sha(p)!=s['private_manifest_sha256']:raise ValueError('C19 private seal changed')
    for rel,h in {**read(p)['files'],**s['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('C19 artifact changed: '+rel)
    return dict(public_files=count,private_files=s['private_files'])


def inputs20():
    cfg=read(RUN20/'binding.json')
    if cfg.get('recovery_authorized') is not True or cfg['limits']!=dict(additional_updates=47,final_update=75,forwards=270,cumulative_forwards=452,training_VJPs=552,probe_VJPs=144,seconds=515,VRAM_bytes=MAX_VRAM17,output_bytes=MAX_OUTPUT20):raise PermissionError('Exact C20 recovery authorization required')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Bound C20 input changed: '+rel)
    old,prepared,train,dev,registry=inputs18()
    if cfg['unchanged_C8_settings']!=old['unchanged_C8_settings'] or cfg['orders']!=old['orders'] or cfg['evaluation']!=old['evaluation']:raise ValueError('Objective/order/protocol drift')
    return cfg,prepared,train,dev,registry


def write_npz20(path,arrays):
    import numpy as np
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite raw outputs')
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic20(path,b.getvalue())


def numpy_random_calls(paths):
    found=[]
    for path in paths:
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute) and node.attr=='random' and isinstance(node.value,ast.Name) and node.value.id in ('np','numpy'):
                found.append(dict(file=str(path.relative_to(ROOT)),line=node.lineno))
    return found


def prepare20():
    if RUN20.exists() or REPORT20.exists():raise FileExistsError('Fresh recovery destination required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT20 or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH20:raise ValueError('C20 parent/branch mismatch')
    c19=verify19_seals();c18=verify18_seals();old,prepared,train,dev,reg=inputs18();proposal=read(REPORT19/'recovery-disabled-v2.json')
    from recovery19 import validate_payload,remaining_sequence
    import torch
    payload=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True);ref=read(next((RUN18/'checkpoints').glob('update-000-*.receipt.json')));initial=torch.load(RUN18/ref['file'],map_location='cpu',weights_only=True)
    validate_payload(payload,old,initial);sequence=remaining_sequence(old,payload)
    paths=[ROOT/('tools/'+rel) for rel in ('phase10c18-v1/worker18.py','phase10c16-v1/balanced16.py','phase10c15-v1/loss15.py','phase10c8-v1/loss8.py','phase10c8-v1/common8.py','phase10a-v1/expanded.py')]
    if numpy_random_calls(paths):raise ValueError('Unrecorded NumPy training RNG requires additional investigation')
    files=dict(old['files'])
    for p in [*HERE20.glob('*.py'),*HERE19.glob('*.py'),REPORT19/'recovery-disabled-v2.json',DATA19/'remaining-sequence.json',CHECKPOINT28]:files[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(old,schema='phase10c20-one-authorized-recovery-v1',parent=PARENT20,branch=BRANCH20,recovery_authorized=True,training_authorized=True,Phase10D_authorized=False,files=files,limits=dict(additional_updates=47,final_update=75,forwards=270,cumulative_forwards=452,training_VJPs=552,probe_VJPs=144,seconds=515,VRAM_bytes=MAX_VRAM17,output_bytes=MAX_OUTPUT20),proposal_sha256=sha(REPORT19/'recovery-disabled-v2.json'),source_checkpoint=proposal['source_checkpoint'],source_checkpoint_sha256=CHECKPOINT28_SHA256,remaining_sequence=sequence,prior_forwards=182,prior_optimizer_updates=28,prior_allocated_bytes=allocation18(),initial_cursor=dict(epoch_zero_based=1,next_group_zero_based=13),numpy_rng_note='Not recorded by C18 and no NumPy RNG calls found in its numerical/data path;no invented restoration. Future snapshots record it without consuming it.',bitwise_uninterrupted_replay_certified=False,mechanical_changes=['Repaired C19 direct guard only;no objective changes','Restore checkpoint28 model/AdamW/scaler/cursor and recorded RNG LAST','Failure telemetry and unique recovery output paths','Epoch2 source-vector diagnostics cover resumed groups14–15 only;previous signed vectors unavailable'],automatic_retry=False)
    RUN20.mkdir(parents=True);REPORT20.mkdir(parents=True)
    for name in ('checkpoints','updates','probes','evaluation','telemetry','failures'):(RUN20/name).mkdir()
    emit20(RUN20/'binding.json',cfg);emit20(REPORT20/'authorized-config.json',{k:v for k,v in cfg.items() if k!='files'})
    emit20(REPORT20/'preflight.json',dict(status='PASS',C19_seal=c19,C18_seal=c18,checkpoint28_sha256=CHECKPOINT28_SHA256,next_update=29,remaining_updates=47,remaining_training_photos=184,numpy_RNG_calls_found=[],numpy_state_recorded_in_C18=False,numpy_missing_state_critical=False,numpy_note=cfg['numpy_rng_note'],GPU_calls=0,uninterrupted_bitwise_equivalence_not_claimed=True))
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
    print('C20 CPU input preflight PASS;exact checkpoint28;one47update recovery',flush=True)


if __name__=='__main__':prepare20()
