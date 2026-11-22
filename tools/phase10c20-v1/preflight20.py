"""CPU tests and historical preservation before consuming the one recovery."""
import re
from common20 import *


def run():
    cfg,*_=inputs20();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');results=[]
    # C19's disabled-proposal test asserted the future destination was absent.
    # It is now explicitly authorized/created; retain all other C19 checks.
    import unittest
    from test_telemetry_recovery19 import RecoveryTests
    recovery_checks=['test_telemetry_recovery19.RecoveryTests.'+n for n in unittest.defaultTestLoader.getTestCaseNames(RecoveryTests) if n!='test_proposal_disabled_with_correct_budgets_and_no_executor']
    suites=[('C20',['discover','-s',str(HERE20),'-p','test_recovery20.py'],ROOT),('C19-guard',['test_guard19.GuardTests',*recovery_checks],HERE19),('C16',['discover','-s',str(HERE16),'-p','test_balanced16.py'],ROOT)]
    for name,args,cwd in suites:
        log=RUN20/('preflight-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();m=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not m:raise RuntimeError('CPU recovery preflight failed: '+str(log))
        results.append(dict(suite=name,passed=int(m[1]),file=str(log.relative_to(ROOT)),sha256=sha(log)));print('PASS',name,m[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation();history.update(C19_seals=verify19_seals(),C18_seals=verify18_seals(),CPU_only=True,checkpoint28_sha256=sha(CHECKPOINT28))
    emit20(REPORT20/'preservation-before-GPU.json',history);emit20(REPORT20/'CPU-preflight-tests.json',dict(status='PASS',binding_sha256=sha(RUN20/'binding.json'),CPU_passed=sum(v['passed'] for v in results),suites=results,GPU_forwards=0));print('C20 CPU preflight andhistoricalpreservationPASS',flush=True)


if __name__=='__main__':run()
