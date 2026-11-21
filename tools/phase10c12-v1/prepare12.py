"""Emit the one disabled config and run only CPU verification suites."""
import re,subprocess
from common12 import *
from config12 import specification,check_spec


def run():
    verify_preparation()
    if FUTURE.exists():raise FileExistsError('No C12 GPU execution authorized')
    cfg=specification();cfg['code_hashes']=dict(read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json')['code_hashes'])
    cfg['code_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in HERE12.glob('*.py')})
    check_spec(cfg);save(REPORT12/'experiment-disabled.json',cfg)
    commands=[('C12',HERE12,None),('C11',ROOT/'tools/phase10c11-v1',None),('frozen_scope',ROOT/'tools/phase10c10-v1','test_preflight10.ScopeTests'),('C8_loss',ROOT/'tools/phase10c8-v1',None),('C6_mask',ROOT/'tools/phase10c6-v1',None),('expanded_base',ROOT/'tools/phase10a-v1',None)]
    suites=[]
    for name,path,test in commands:
        args=[str(ROOT/'.venv/bin/python'),'-B','-m','unittest']+([test,'-v'] if test else ['discover','-s',str(path),'-p','test*.py','-v'])
        log=OUT12/('tests-'+name+'.log')
        with log.open('x') as f:r=subprocess.run(args,cwd=path if test else ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();n=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not n or not s.rstrip().endswith('OK'):raise RuntimeError('CPU suite failed; inspect '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('CPU PASS',name,n[1],flush=True)
    save(REPORT12/'tests.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,CPU_only=True,real_training_updates=0,GPU_forwards=0,C12_experiment_directory_created=FUTURE.exists(),old_C10_execution_absence_tests_not_applicable_after_C11=True))


if __name__=='__main__':run()
