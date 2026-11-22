"""CPU suites and read-only historical preservation,no model execution."""
import os,subprocess,re
from common16 import *


def run():
    suites=[('C16',['discover','-s',str(HERE16),'-p','test*.py'],ROOT),('C8',['discover','-s','tools/phase10c8-v1','-p','test*.py'],ROOT),('C6',['discover','-s','tools/phase10c6-v1','-p','test*.py'],ROOT),('C14',['discover','-s','tools/phase10c14-v1','-p','test*.py'],ROOT),('Expanded',['discover','-s','tools/phase10a-v1','-p','test*.py'],ROOT),('C15-loss',['test_smoke15.LossTests','test_smoke15.CachedRegressionTests'],HERE15)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');results=[]
    for name,args,cwd in suites:
        log=DATA16/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not m or 'skipped=' in s:raise RuntimeError('CPU suite failed: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,m[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation();_,_,confirmed=verify16_inputs();history['C15R']=confirmed
    meta=read(REPORT14/'artifact-seal.json');p=ROOT/meta['private_seal']
    if sha(p)!=meta['private_seal_sha256']:raise ValueError('C14 private seal changed')
    history['C14_public_files']=verify_seal(REPORT14/'SHA256SUMS');history['C14_private_files']=verify_seal(p)
    m=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');p=ROOT/m['private_seal']
    if sha(p)!=m['private_seal_sha256']:raise ValueError('C13 private seal changed')
    history['C13_private_files']=verify_seal(p);history['C13_public_files']=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS');history['C11']=verify_history()[0];history['C8_private_files']=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    from spec16 import validate_spec
    cfg=read(REPORT16/'future-experiment-disabled.json');validate_spec(cfg)
    for key in ('destination',):
        if (ROOT/cfg[key]).exists() or (ROOT/read(REPORT16/'next-approval-request.json')[key]).exists():raise ValueError('Unapproved GPU/training directory was created')
    history.update(real_model_forwards=0,GPU_training=0,new_negative_labels=0,policy_disabled_by_default=True,C8_calibration_trainability_preserved_in_proposal=True)
    def versioned(name,data):
        path=REPORT16/name;version=1
        while path.exists():version+=1;path=REPORT16/(Path(name).stem+'-v'+str(version)+Path(name).suffix)
        emit16(path,data)
    versioned('preservation.json',history);versioned('tests.json',dict(status='PASS',CPU_passed=sum(r['passed'] for r in results),suites=results,browser=read(REPORT16/'browser-tests.json'),new_model_forwards=0,training_updates=0));print('C16 CPU/HISTORICAL PASS',sum(r['passed'] for r in results),history['historical_data_model_files'],flush=True)


if __name__=='__main__':run()
