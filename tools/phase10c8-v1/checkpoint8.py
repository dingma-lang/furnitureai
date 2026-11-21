"""Atomic unique recovery snapshots and conservative optimizer intent records."""
from common8 import *

def save_snapshot(folder,payload):
    import torch
    out=folder/'checkpoints';out.mkdir(exist_ok=True);p=out/('update-'+str(payload['updates']).zfill(3)+'-'+uuid.uuid4().hex[:12]+'.pt');tmp=p.with_suffix('.pending')
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    try:os.link(tmp,p)
    finally:tmp.unlink()
    rec=dict(file=str(p.relative_to(RUN)),sha256=sha(p),updates=payload['updates'],epoch=payload['epoch'],bytes=p.stat().st_size)
    # Independent reload catches storage/schema corruption before the next update.
    state=torch.load(p,map_location='cpu',weights_only=True)
    if state['updates']!=payload['updates'] or state['binding_sha256']!=payload['binding_sha256'] or set(state['weights'])!=set(payload['weights']) or any(not torch.equal(state['weights'][k],payload['weights'][k]) for k in payload['weights']):raise ValueError('Checkpoint replay verification failed')
    save(out/(p.stem+'.receipt.json'),rec);return rec
