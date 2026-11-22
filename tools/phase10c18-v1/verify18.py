"""Post-run CPU tests, checkpoint audit and historical read-only preservation."""
import re
from common18 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('Verification must be CPU-only')
    import torch
    cfg,*_=inputs18();failure=read(RUN18/'failure.json');done=failure['state'];records=[]
    initial=list((RUN18/'checkpoints').glob('update-000-*.receipt.json'))
    if len(initial)!=1 or done['updates']!=28 or done['optimizer_hook_calls']!=28:raise ValueError('Exact stopped single attempt required')
    for i in range(29):
        rec=read(initial[0]) if i==0 else read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json'))['checkpoint']
        p=RUN18/rec['file']
        if sha(p)!=rec['sha256']:raise ValueError('Checkpoint bytes changed')
        s=torch.load(p,map_location='cpu',weights_only=True)
        if s['updates']!=i or s['frozen_sha256']!=failure['original_frozen_fingerprint'] or s['binding_sha256']!=sha(RUN18/'binding.json') or set(s['weights'])!=HEAD_KEYS or any(not torch.isfinite(v).all() for v in s['weights'].values()) or any(int(v['step'])!=i for v in s['optimizer']['state'].values()):raise ValueError('Snapshot schema/state counter integrity failed')
        for g in s['optimizer']['param_groups']:
            if g['lr']!=1e-5 or g['betas']!=(.9,.999) or g['eps']!=1e-8 or g['weight_decay']!=.01:raise ValueError('Optimizer setting drift')
        records.append(rec)
    registry=dict(checkpoints=records,latest=records[-1],epoch5_checkpoint_exists=False,deployed=False)
    if (RUN18/'checkpoint-registry.json').exists():
        if read(RUN18/'checkpoint-registry.json')!=registry:raise ValueError('Registry changed')
    else:emit18(RUN18/'checkpoint-registry.json',registry)
    results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    suites=[('C18-completed',['discover','-s',str(HERE18),'-p','test_completed18.py'],ROOT),('C18-training',['discover','-s',str(HERE18),'-p','test_training18.py'],ROOT),('C8',['discover','-s','tools/phase10c8-v1','-p','test_ab8.py'],ROOT)]
    for name,args,cwd in suites:
        p=RUN18/('CPU-final-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with p.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=p.read_text();match=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not match:raise RuntimeError('CPU verification suite failed: '+str(p))
        results.append(dict(suite=name,passed=int(match[1]),file=str(p.relative_to(ROOT)),sha256=sha(p)));print('PASS',name,match[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation()
    if 'GPU_model_forwards' in history:history['historical_preservation_routine_GPU_forwards']=history.pop('GPU_model_forwards')
    history.update(C17_seals=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),actual_C18_forwards=done['forwards'],actual_C18_optimizer_updates=28,original_checkpoint_sha256=sha(BASE/'model.safetensors'),all29new_snapshots_reloaded=True,conservative_new_allocated_bytes=allocation18(),CPU_verification_forwards=0)
    for phase in (11,13,14,16):
        folder=ROOT/('reports/phase10c'+str(phase)+'-v1');seal=read(folder/'artifact-seal.json');private=ROOT/seal.get('private_manifest',seal.get('private_seal',''))
        expected=seal.get('private_manifest_sha256',seal.get('private_seal_sha256'))
        if sha(private)!=expected:raise ValueError('Historical private seal changed')
        if private.suffix=='.json':
            for rel,h in read(private)['files'].items():
                if sha(ROOT/rel)!=h:raise ValueError('Historical artifact changed')
        else:verify_seal(private)
        history['C'+str(phase)+'_public_seal_files']=verify_seal(folder/'SHA256SUMS')
    emit18(REPORT18/'preservation.json',history);emit18(REPORT18/'tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,pre_GPU_CPU_tests=56,new_GPU_after_single_arm=0,checkpoints_verified=29,experiment_status='STOPPED_SAFETY_GUARD'));print('C18 post-run tests and historical preservation PASS',flush=True)


if __name__=='__main__':run()
