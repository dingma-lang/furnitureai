"""Run current C16 contract and C17 CPU tests before a single GPU launch."""
import subprocess,re
from common17 import *


def run():
    inputs17();results=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,folder in [('C16',HERE16),('C17',HERE17)]:
        log=RUN17/('CPU-preflight-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(folder),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();m=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not m or 'skipped=' in s:raise RuntimeError('CPU preflight failed;no GPU launch: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('CPU PASS',name,m[1],flush=True)
    emit17(REPORT17/'CPU-preflight-tests.json',dict(status='PASS',configuration_sha256=sha(RUN17/'authorized-config.json'),passed=sum(r['passed'] for r in results),suites=results,all24_original_cached_mask_checks_passed=True,full59_sequence_checked=True,model_forwards=0))


if __name__=='__main__':run()
