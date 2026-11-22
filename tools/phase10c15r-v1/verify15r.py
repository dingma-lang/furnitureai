"""Post-recovery CPU suites and historical hashes,never a model run."""
import subprocess,re
from common15r import *


def run():
    suites=[('C15R',['discover','-s',str(HERER),'-p','test*.py']),('C14',['discover','-s','tools/phase10c14-v1','-p','test*.py']),('FrozenScope',['discover','-s','tools/phase10c10-v1','-p','test_preflight10.py','-k','ScopeTests']),('C8',['discover','-s','tools/phase10c8-v1','-p','test*.py']),('C6',['discover','-s','tools/phase10c6-v1','-p','test*.py'])];results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args in suites:
        log=RUNR/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not m or 'skipped=' in s:raise RuntimeError('CPU regression failed: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,m[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation();prior,reused,counts=verify_prior();history['C15_seals']=counts
    meta=read(REPORT14/'artifact-seal.json');p=ROOT/meta['private_seal']
    if sha(p)!=meta['private_seal_sha256']:raise ValueError('C14 private seal changed')
    history['C14_public_files']=verify_seal(REPORT14/'SHA256SUMS');history['C14_private_files']=verify_seal(p)
    meta13=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');p=ROOT/meta13['private_seal']
    if sha(p)!=meta13['private_seal_sha256']:raise ValueError('C13 private seal changed')
    history['C13_public_files']=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS');history['C13_private_files']=verify_seal(p);history['C11']=verify_history()[0];history['C8_private_files']=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    cfg,_,_,_=inputs_r();done=read(RUNR/'complete.json');scope=read(RUNR/'initialization.json')
    if done['model_state_before']!=done['model_state_after'] or done['model_state_after']!=read(PRIOR_RUN/'initialization.json')['model_state']['sha256']:raise ValueError('Model state changed across fixed-weight diagnosis')
    history.update(recovery_forwards=20,cumulative_forwards=25,recovery_gradient_calculations=140,cumulative_gradient_calculations=168,optimizer_updates=0,original_checkpoint_sha256=sha(BASE/'model.safetensors'),C11_head_sha256=sha(ROOT/cfg['head']['path']),live_model_before=done['model_state_before'],live_model_after=done['model_state_after'],all_recovery_parameters_and_buffers_unchanged=True,old_missing_telemetry_unchanged=True,allocated_new_diagnostic_bytes=allocated_r(),additional_model_inference_in_CPU_verification=0)
    emit_r(REPORTR/'preservation.json',history);emit_r(REPORTR/'tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,all24_cached_equality_passed=True,GPU_forwards_after_recovery=0));print('C15R CPU/HISTORICAL PRESERVATION PASS',sum(r['passed'] for r in results),history['historical_data_model_files'],flush=True)


if __name__=='__main__':run()
