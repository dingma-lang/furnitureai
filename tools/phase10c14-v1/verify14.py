"""Relevant CPU suites and read-only historical preservation; no model runner."""
import os,re,subprocess,sys,uuid
from common14 import *


def run():
    suites=[('C14',['discover','-s','tools/phase10c14-v1','-p','test*.py']),('C13',['discover','-s','tools/phase10c13-v1','-p','test*.py']),('C11',['discover','-s','tools/phase10c11-v1','-p','test*.py']),('FrozenScope',['discover','-s','tools/phase10c10-v1','-p','test_preflight10.py','-k','ScopeTests']),('C8',['discover','-s','tools/phase10c8-v1','-p','test*.py']),('C6',['discover','-s','tools/phase10c6-v1','-p','test*.py']),('ExpandedData',['discover','-s','tools/phase10a-v1','-p','test*.py'])]
    results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args in suites:
        path=RAW14/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with path.open('x') as f:r=subprocess.run([sys.executable,'-B','-m','unittest',*args,'-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        log=path.read_text();match=re.search(r'Ran (\d+) tests?',log)
        if r.returncode or not match or 'skipped=' in log:raise RuntimeError('CPU suite failed: '+str(path))
        results.append(dict(suite=name,passed=int(match[1]),log=str(path.relative_to(ROOT)),sha256=sha(path)));print('PASS',name,match[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    historical=preservation()
    public13=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS');record=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');private=ROOT/record['private_seal']
    if sha(private)!=record['private_seal_sha256']:raise ValueError('C13 private seal changed')
    private13=verify_seal(private);history11=verify_history()[0];c8=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    _,train,dev,photos,registry=training_context()
    for row in train:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Optimization source photo changed')
    historical.update(C13_public_files=public13,C13_private_files=private13,C11=history11,C8_private_files=c8,optimization_photos=59,canonical_training_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'))
    save(REPORT14/'preservation.json',historical)
    save(REPORT14/'tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,browser=read(REPORT14/'browser-tests.json'),live_browser=read(REPORT14/'live-browser-tests-v2.json'),new_model_forwards=0,optimizer_updates=0))
    print('CPU/HISTORICAL PASS',sum(r['passed'] for r in results),historical['historical_data_model_files'],flush=True)


if __name__=='__main__':run()
