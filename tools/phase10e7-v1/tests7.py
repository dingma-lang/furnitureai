"""Bounded CPU suites only;writes fresh passed/failed transcripts."""
import re
from common7 import *
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only tests')
    suites=[]
    plans=[('E7',['discover','-s',str(HERE),'-p','test_cache7.py'],ROOT),
           ('E6',['discover','-s',str(ROOT/'tools/phase10e6-v1'),'-p','test_auditE6.py'],ROOT),
           ('E1-mask-cache',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1')]
    for name,args,cwd in plans:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),capture_output=True,timeout=120)
        p=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log');atomic(p,result.stdout+result.stderr)
        text=p.read_text();m=re.search(r'Ran (\d+) tests?',text)
        if result.returncode or not m or '\nOK' not in text or 'skipped=' in text:raise RuntimeError('CPU test failure:'+str(p))
        suites.append(dict(name=name,passed=int(m[1]),log=str(p.relative_to(ROOT)),sha256=sha(p)));print('CPU PASS',name,m[1],flush=True)
    emit(REPORT/'CPU-tests.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,GPU_forwards=0))
if __name__=='__main__':run()
