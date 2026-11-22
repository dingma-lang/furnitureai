"""Read-only historical verification and final CPU suites; never a GPU runner."""
import re
from common19 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification required')
    import torch
    log=DATA19/'CPU-tests-attempt2.log';s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
    if not m or int(m[1])!=38 or not s.rstrip().endswith('OK'):raise ValueError('Final C19 CPU suite required')
    results=[dict(suite='C19',passed=38,file=str(log.relative_to(ROOT)),sha256=sha(log))]
    suites=[('C16',['discover','-s','tools/phase10c16-v1','-p','test_balanced16.py'],ROOT),('C8',['discover','-s','tools/phase10c8-v1','-p','test_ab8.py'],ROOT),('C15-loss',['test_smoke15.LossTests','test_smoke15.CachedRegressionTests'],HERE15)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in suites:
        p=DATA19/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with p.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=p.read_text();m=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not m:raise RuntimeError('CPU suite failed: '+str(p))
        results.append(dict(suite=name,passed=int(m[1]),file=str(p.relative_to(ROOT)),sha256=sha(p)));print('PASS',name,m[1],flush=True)
    cfg,*_=inputs18();prior=verify18_seals();c17=verify17_seals()
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation()
    if 'GPU_model_forwards' in history:history['historical_preservation_routine_GPU_forwards']=history.pop('GPU_model_forwards')
    for phase in (11,13,14,16):
        folder=ROOT/('reports/phase10c'+str(phase)+'-v1');seal=read(folder/'artifact-seal.json');private=ROOT/seal.get('private_manifest',seal.get('private_seal',''))
        if sha(private)!=seal.get('private_manifest_sha256',seal.get('private_seal_sha256')):raise ValueError('Historical seal changed')
        if private.suffix=='.json':
            for rel,h in read(private)['files'].items():
                if sha(ROOT/rel)!=h:raise ValueError('Historical private artifact changed')
        else:verify_seal(private)
        history['C'+str(phase)+'_public_files']=verify_seal(folder/'SHA256SUMS')
    history.update(C18_seals=prior,C17_seals=c17,C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),checkpoint28_sha256=sha(CHECKPOINT28),original_checkpoint_sha256=sha(BASE/'model.safetensors'),C19_GPU_forwards=0,C19_training_updates=0,C19_model_inference=0,CUDA_initialized=torch.cuda.is_initialized(),new_outputs_allocated_bytes=allocated19())
    if torch.cuda.is_initialized():raise RuntimeError('C19 may not initialize CUDA')
    proposal=read(ROOT/read(REPORT19/'recovery-proposal-status.json')['active_proposal'])
    if proposal['guard_sha256']!=sha(HERE19/'guard19.py') or proposal['telemetry_sha256']!=sha(HERE19/'telemetry19.py') or proposal['recovery_helpers_sha256']!=sha(HERE19/'recovery19.py'):raise ValueError('Disabled proposal code binding changed')
    emit19(REPORT19/'tests.json',dict(status='PASS',CPU_passed=sum(v['passed'] for v in results),suites=results,CUDA_initialized=False,GPU_forwards=0,optimizer_steps_on_real_models=0));emit19(REPORT19/'preservation.json',history)
    print('C19 CPU/historical preservation PASS;total',sum(v['passed'] for v in results),';noCUDA',flush=True)


if __name__=='__main__':run()
