"""Isolated CPU tests and read-only historical checks. Never launches the worker."""
import argparse
import ast
import importlib.util
import re
from common4 import *


def tests(post=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only tests required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=[]
    plans=[('E4',['discover','-s',str(HERE),'-p','test_recovery4.py'],ROOT),
           ('E3',['discover','-s',str(ROOT/'tools/phase10e3-v1'),'-p','test_*3.py'],ROOT),
           ('E1',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1'),
           ('C11',['test_preflight10.ScopeTests'],ROOT/'tools/phase10c10-v1')]
    excluded=[]
    if post:
        # E3's preparation-only absence assertion expires when the user
        # authorizes E4. Keep every numerical,mask,routing,persistence check.
        tree=ast.parse((ROOT/'tools/phase10e3-v1/test_integration3.py').read_text())
        names=[n.name for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='IntegrationTests'
               for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
        omit='test_recovery_proposal_and_execution_disabled'
        args=['test_numerics3',*['test_integration3.IntegrationTests.'+n for n in names if n!=omit]]
        plans[1]=('E3',args,ROOT/'tools/phase10e3-v1')
        excluded=[dict(test=omit,reason='Passed pre-GPU;its recovery-directory-must-not-exist assertion is no longer applicable after authorized E4 creation. Original source unchanged.')]
    for name,args,cwd in plans:
        log=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:
            r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();match=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not match or '\nOK' not in text or 'skipped=' in text:raise RuntimeError('CPU tests failed: '+str(log))
        result.append(dict(suite=name,passed=int(match[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)));print('CPU PASS',name,match[1],flush=True)
    emit(REPORT/('postrun-tests.json' if post else 'CPU-tests.json'),dict(status='PASS',passed=sum(r['passed'] for r in result),suites=result,GPU_calls=0,real_model_updates=0,phase_lifecycle_exclusions=excluded))


def preservation():
    sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
    from verify3 import check as check3
    from verifyE import check as check1
    sys.path.insert(0,str(ROOT/'tools/phase10e2-v1'))
    from verify2 import check as check2
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from sealD import check as checkD
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from verify21 import check_seal
    from common21 import verify20_seals,verify19_seals,verify18_seals,verify17_seals,C8,verify_seal
    for check in (check1,check2,check3,checkD,check_seal):check()
    spec=importlib.util.spec_from_file_location('_old_integrity4',ROOT/'tools/phase10c-v1/verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    data=old.preservation();extra={}
    for phase in ('phase10c11-v1','phase10c13-v1'):
        folder=ROOT/'reports'/phase;meta=read(folder/'artifact-seal.json');private=ROOT/meta['private_seal']
        if sha(private)!=meta['private_seal_sha256']:raise ValueError('Historical seal modified')
        extra[phase]=dict(public=verify_seal(folder/'SHA256SUMS'),private=verify_seal(private))
    cfg,*_=inputs()
    data.update(C11_C13=extra,C20=verify20_seals(),C19=verify19_seals(),C18=verify18_seals(),C17=verify17_seals(),
                C8=verify_seal(C8/'ARTIFACT-SHA256SUMS'),E1=True,E2=True,E3=True,D=True,C21=True,
                original_sha256=sha(BASE/'model.safetensors'),adapted_sha256=sha(ROOT/cfg['reference_checkpoint']),
                main=subprocess.check_output(['git','rev-parse','main'],text=True).strip())
    if data['original_sha256']!=BASE_SHA or data['adapted_sha256']!=cfg['reference_sha256'] or data['main']!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Historical weights/main changed')
    # Historical utility's GPU counters describe its own read-only check, not E4.
    data['historical_check_only']=True;emit(REPORT/'preservation.json',data)
    print('Historical preservation PASS',data['historical_data_model_files'],flush=True)


def seal():
    if read(REPORT/'preservation.json')['status']!='PASS' or not (REPORT/'report.md').exists():raise ValueError('Final report/preservation required')
    private={str(p.relative_to(ROOT)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    emit(REPORT/'artifact-seal.json',dict(private_files=private,code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},
                                        GPU_training=False,optimizer_updates=0,conservative_allocated_bytes_before_seal=allocated()))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())


def check():
    seal=read(REPORT/'artifact-seal.json')
    for rel,h in {**seal['private_files'],**seal['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E4 sealed artifact changed: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('E4 public checksum mismatch')
    print('E4 final seal PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tests',action='store_true');p.add_argument('--postrun',action='store_true');p.add_argument('--preservation',action='store_true');p.add_argument('--seal',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification')
    if a.tests:tests(a.postrun)
    elif a.preservation:preservation()
    elif a.seal:seal()
    elif a.check:check()
    else:p.error('Choose CPU verification action')
