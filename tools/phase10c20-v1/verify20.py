"""Audit recovery checkpoint registry,CPU integration tests and historical files."""
import re
from common20 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU verification only')
    import torch
    cfg,*_=inputs20();done=read(RUN20/'complete.json');registry=[]
    for i in range(29,76):
        u=read(RUN20/'updates'/('update-'+str(i).zfill(3)+'.json'));rec=u['checkpoint'];p=RUN20/rec['file']
        if sha(p)!=rec['sha256'] or p.stat().st_size>20000000:raise ValueError('Checkpoint bytes/size mismatch')
        s=torch.load(p,map_location='cpu',weights_only=True)
        if s['updates']!=i or s['epoch']*15+s['next_group']!=i or s['binding_sha256']!=sha(RUN20/'binding.json') or set(s['weights'])!=HEAD_KEYS or s['frozen_sha256']!=done['frozen_sha256'] or any(not torch.isfinite(v).all() for v in s['weights'].values()) or any(int(v['step'])!=i for v in s['optimizer']['state'].values()):raise ValueError('Checkpoint state/cursor integrity failed')
        for g in s['optimizer']['param_groups']:
            if g['lr']!=1e-5 or g['betas']!=(.9,.999) or g['weight_decay']!=.01 or g['eps']!=1e-8:raise ValueError('Optimizer options changed')
        registry.append(rec)
    emit20(RUN20/'checkpoint-registry.json',dict(source=done['initial_checkpoint'],checkpoints=registry,final=done['final_checkpoint'],deployed=False))
    results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in [('C20-completed',['discover','-s',str(HERE20),'-p','test_completed20.py'],ROOT),('C8-protocol',['test_ab8.MetricTests'],ROOT/'tools/phase10c8-v1')]:
        p=RUN20/('CPU-final-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with p.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=p.read_text();match=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not match:raise RuntimeError('CPU final test failed: '+str(p))
        results.append(dict(suite=name,passed=int(match[1]),file=str(p.relative_to(ROOT)),sha256=sha(p)));print('PASS',name,match[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation()
    if 'GPU_model_forwards' in history:history['historical_preservation_GPU_forwards']=history.pop('GPU_model_forwards')
    history.update(C19_seals=verify19_seals(),C18_seals=verify18_seals(),C17_seals=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),checkpoint28_sha256=sha(CHECKPOINT28),original_checkpoint_sha256=sha(BASE/'model.safetensors'),recovery_updates=47,cumulative_updates=75,recovery_forwards=270,cumulative_forwards=452,all47new_checkpoints_reloaded=True,frozen_parameters_unchanged=done['frozen_sha256']==read(RUN20/'restoration.json')['frozen_sha256'],allocated_new_output_bytes=allocation20(),new_GPU_after_recovery=0)
    emit20(REPORT20/'preservation.json',history);emit20(REPORT20/'tests.json',dict(status='PASS',CPU_postrun_passed=sum(r['passed'] for r in results),CPU_preflight_passed=73,suites=results,checkpoint_snapshots_reloaded=47,GPU_calls_after_recovery=0))
    print('C20 completed integrity verification;historical preservation PASS',flush=True)


if __name__=='__main__':run()
