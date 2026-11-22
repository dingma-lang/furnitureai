"""All24 saved-photo equality and recovery tests must pass before one launch."""
import subprocess,re
from common15r import *


def run():
    cfg,_,_,_=inputs_r();log=RUNR/('CPU-preflight-'+uuid.uuid4().hex[:8]+'.log');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERER),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    s=log.read_text();n=re.search(r'Ran (\d+) tests?',s);cached='test_all24_saved_C11_training_outputs_preserve_exact_executed_box_loss'
    if r.returncode or not n or cached not in s or 'skipped=' in s:raise RuntimeError('CPU preflight failed;no GPU launch;inspect '+str(log))
    emit_r(REPORTR/'CPU-preflight-tests.json',dict(status='PASS',passed=int(n[1]),all24_cached_equality_passed=True,configuration_sha256=sha(RUNR/'authorized-config.json'),log=str(log.relative_to(ROOT)),log_sha256=sha(log),new_GPU_forwards=0))
    print('CPU PREFLIGHT PASS',n[1],';all24 cached equality tests;no GPU forwards',flush=True)


if __name__=='__main__':run()
