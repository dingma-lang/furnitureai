"""Fresh CPU logs only,never mutate historical test-fixture directories."""
import re
from common18 import *


def run():
    cfg,*_=inputs18();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');results=[]
    suites=[('C18',[str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERE18),'-p','test_training18.py','-v'],ROOT),('C16',[str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERE16),'-p','test_balanced16.py','-v'],ROOT),('C15-loss',[str(ROOT/'.venv/bin/python'),'-B','-m','unittest','test_smoke15.LossTests','test_smoke15.CachedRegressionTests','-v'],HERE15)]
    for name,command,cwd in suites:
        log=RUN18/('CPU-preflight-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();match=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or match is None:raise RuntimeError('CPU preflight failed: '+str(log))
        results.append(dict(suite=name,passed=int(match[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,match[1],flush=True)
    verify17_seals()
    emit18(REPORT18/'CPU-preflight-tests.json',dict(status='PASS',binding_sha256=sha(RUN18/'binding.json'),suites=results,CPU_passed=sum(r['passed'] for r in results),GPU_forwards=0))


if __name__=='__main__':run()
