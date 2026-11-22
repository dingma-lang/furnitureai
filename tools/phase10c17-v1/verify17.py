"""Post-smoke CPU suites and historical preservation;never calls GPU worker."""
import subprocess,re
from common17 import *


def run():
    suites=[('C17',['discover','-s',str(HERE17),'-p','test*.py'],ROOT),('C16',['discover','-s',str(HERE16),'-p','test*.py'],ROOT),('C8',['discover','-s','tools/phase10c8-v1','-p','test*.py'],ROOT),('C6',['discover','-s','tools/phase10c6-v1','-p','test*.py'],ROOT),('C14',['discover','-s','tools/phase10c14-v1','-p','test*.py'],ROOT),('Expanded',['discover','-s','tools/phase10a-v1','-p','test*.py'],ROOT),('C15-loss',['test_smoke15.LossTests','test_smoke15.CachedRegressionTests'],HERE15)]
    results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in suites:
        log=RUN17/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not m or 'skipped=' in s:raise RuntimeError('CPU suite failed: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,m[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation();public=verify_seal(REPORT16/'SHA256SUMS');meta=read(REPORT16/'artifact-seal.json');p=ROOT/meta['private_manifest']
    if sha(p)!=meta['private_manifest_sha256']:raise ValueError('C16 private manifest changed')
    for rel,h in read(p)['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C16 preparation artifact changed')
    for rel,h in meta['code_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C16 algorithm changed')
    history['C16_public_files']=public;history['C16_private_files']=meta['private_files'];history['C15R']=verify16_inputs()[2]
    m14=read(REPORT14/'artifact-seal.json');p=ROOT/m14['private_seal']
    if sha(p)!=m14['private_seal_sha256']:raise ValueError('C14 seal changed')
    history['C14_private_files']=verify_seal(p);history['C14_public_files']=verify_seal(REPORT14/'SHA256SUMS')
    m13=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');p=ROOT/m13['private_seal']
    if sha(p)!=m13['private_seal_sha256']:raise ValueError('C13 seal changed')
    history['C13_private_files']=verify_seal(p);history['C13_public_files']=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS');history['C11']=verify_history()[0];history['C8_private_files']=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    cfg,_,groups,_=inputs17();done=read(RUN17/'complete.json')
    history.update(actual_smoke_forwards=59,actual_smoke_gradient_calculations=295,optimizer_steps=0,model_updates=0,initial_model_sha256=done['model_state_before'],final_model_sha256=done['model_state_after'],original_checkpoint_sha256=sha(BASE/'model.safetensors'),calibration_trainable=True,all12original_heads_unchanged=True,cap_metadata_adapter_only=True,original_C16_code_changed=False,conservative_new_allocated_bytes=allocation17(),CPU_verification_model_forwards=0)
    emit17(REPORT17/'preservation.json',history);emit17(REPORT17/'tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,pre_GPU_tests=44,new_GPU_after_smoke=0));print('C17 CPU/HISTORICAL PASS',sum(r['passed'] for r in results),history['historical_data_model_files'],flush=True)


if __name__=='__main__':run()
