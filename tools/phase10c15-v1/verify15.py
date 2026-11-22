"""Post-stop CPU tests and immutable history checks;no GPU retry entry point."""
import re,subprocess
from common15 import *


def run():
    suites=[('C15',['discover','-s','tools/phase10c15-v1','-p','test*.py']),('C14',['discover','-s','tools/phase10c14-v1','-p','test*.py']),('FrozenScope',['discover','-s','tools/phase10c10-v1','-p','test_preflight10.py','-k','ScopeTests']),('C8',['discover','-s','tools/phase10c8-v1','-p','test*.py']),('C6',['discover','-s','tools/phase10c6-v1','-p','test*.py'])]
    results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args in suites:
        log=RUN15/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not m or 'skipped=' in s:raise RuntimeError('CPU test failure: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,m[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation();history['C14_public_files']=verify_seal(REPORT14/'SHA256SUMS');old=read(REPORT14/'artifact-seal.json');seal=ROOT/old['private_seal']
    if sha(seal)!=old['private_seal_sha256']:raise ValueError('C14 seal changed')
    history['C14_private_files']=verify_seal(seal)
    record13=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');history['C13_public_files']=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS')
    if sha(ROOT/record13['private_seal'])!=record13['private_seal_sha256']:raise ValueError('C13 seal changed')
    history['C13_private_files']=verify_seal(ROOT/record13['private_seal']);history['C11']=verify_history()[0];history['C8_private_files']=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    cfg=read(RUN15/'authorized-config.json');intentional=[]
    for relative,expected in cfg['files'].items():
        p=ROOT/relative
        if p.is_relative_to(HERE15):
            archived=RUN15/'executed-source-v1'/p.name
            if sha(archived)!=expected:raise ValueError('Executed source archive changed')
            if sha(p)!=expected:intentional.append(relative)
        elif sha(p)!=expected:raise ValueError('Pinned data,model,environment or historical source changed: '+relative)
    if set(intentional)!={str((HERE15/name).relative_to(ROOT)) for name in ('loss15.py','gpu15.py','test_smoke15.py')}:raise ValueError('Unexpected repaired-code drift')
    if (RUN15/'complete.json').exists() or not (RUN15/'failure.json').exists():raise ValueError('Failed run status changed')
    failure=read(RUN15/'failure.json');assert failure['forwards']==5 and failure['gradient_calculations']==28
    if allocated()>MAX_BYTES:raise ValueError('128MiB diagnostic storage limit')
    history.update(smoke_GPU_forwards=5,smoke_gradient_calculations=28,smoke_optimizer_updates=0,repaired_code_never_executed_on_GPU=True,executed_sources_preserved=True,intentionally_repaired_new_code=intentional,old_GPU_binding_rejects_current_repair=True,terminal_live_parameter_hash_not_available=True,allocated_diagnostic_bytes=allocated(),new_model_or_dataset_downloads=0)
    emit(REPORT15/'preservation.json',history);emit(REPORT15/'tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,all24_cached_CPU_box_loss_regression_passed=True,additional_GPU_forwards_after_failure=0))
    print('POST-STOP CPU/PRESERVATION PASS',sum(r['passed'] for r in results),history['historical_data_model_files'],flush=True)


if __name__=='__main__':run()
