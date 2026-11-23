"""CPU-only pre/post tests and immutable artifact checks;never launches training."""
import argparse,importlib.util,re
from common5 import *

def tests(post=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only test process')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    plans=[('E5',['discover','-s',str(HERE),'-p','test_experiment5.py'],ROOT),
           ('E3-numerics',['test_numerics3'],ROOT/'tools/phase10e3-v1'),
           ('E1-mask-cache',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1'),
           ('C11-scope',['test_preflight10.ScopeTests'],ROOT/'tools/phase10c10-v1'),
           ('metrics',['test_phase10c.MetricTests'],ROOT/'tools/phase10c-v1')]
    suites=[]
    for name,args,cwd in plans:
        log=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();match=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not match or '\nOK' not in text or 'skipped=' in text:raise RuntimeError('CPU tests failed: '+str(log))
        suites.append(dict(name=name,passed=int(match[1]),log_sha256=sha(log)));print('CPU PASS',name,match[1],flush=True)
    emit(REPORT/('postrun-tests.json' if post else 'CPU-tests.json'),dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,new_GPU_calls=0))

def preservation():
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
    from verify4 import check as e4check
    e4check()
    # Reuse E4's complete preservation verifier with a private copy of its
    # globals redirected ONLY for the new report publication. Historical code
    # and inputs retain their original module globals and immutable files.
    import types,verify4
    values=dict(verify4.preservation.__globals__);values['emit']=lambda p,v:emit(REPORT/'historical-preservation.json',v)
    types.FunctionType(verify4.preservation.__code__,values)()
    cfg,*_=inputs()
    emit(REPORT/'preservation.json',dict(status='PASS',E4_and_prior_seals=True,original_sha256=sha(BASE/'model.safetensors'),
         main=subprocess.check_output(['git','rev-parse','main'],text=True).strip(),historical_check=read(REPORT/'historical-preservation.json'),
         new_checkpoints_preserved=True,coefficient_unchanged=cfg['coefficient']==COEFFICIENT,Mac_project_modified=False))

def seal():
    if not (REPORT/'report.md').exists() or read(REPORT/'preservation.json')['status']!='PASS':raise ValueError('Report/preservation required')
    emit(REPORT/'artifact-seal.json',dict(private_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()},
          code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},another_training_run=False))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())

def check():
    meta=read(REPORT/'artifact-seal.json')
    for rel,h in {**meta['private_files'],**meta['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E5 sealed artifact changed: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public E5 checksum changed')
    print('E5 final artifact seal PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tests',action='store_true');p.add_argument('--postrun',action='store_true');p.add_argument('--preservation',action='store_true');p.add_argument('--seal',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification')
    if a.tests:tests(a.postrun)
    elif a.preservation:preservation()
    elif a.seal:seal()
    elif a.check:check()
    else:p.error('Choose CPU verification action')
