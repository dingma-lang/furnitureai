"""CPU preflight/postrun integrity;no GPU execution or historical output changes."""
import argparse,re
from common2 import *

def tests(post=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only tests')
    cfg,*_=inputs();suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in [('E2',['discover','-s',str(HERE),'-p','test_gradient2.py'],ROOT),('E1loss-cache',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1'),('C11scope',['test_preflight10.ScopeTests'],ROOT/'tools/phase10c10-v1')]:
        p=RUN/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with p.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        match=re.search(r'Ran (\d+) tests?',p.read_text())
        if r.returncode or not match or '\nOK' not in p.read_text() or 'skipped=' in p.read_text():raise RuntimeError('CPU test failure: '+str(p))
        suites.append(dict(suite=name,passed=int(match[1]),log=str(p.relative_to(ROOT)),sha256=sha(p)));print('CPU PASS',name,match[1],flush=True)
    emit(REPORT/('postrun-tests.json' if post else 'CPU-tests.json'),dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,binding_sha256=sha(RUN/'binding.json'),new_GPU_calls=0))

def preservation():
    cfg,*_=inputs()
    from verifyE import check as checkE1
    checkE1()
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from sealD import check as checkD
    checkD()
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from verify21 import check_seal
    from common21 import verify20_seals,verify19_seals,verify18_seals,verify17_seals,C8,verify_seal
    from verify import preservation as old_preservation
    check_seal();p=old_preservation()
    extra={}
    for phase in ('phase10c11-v1','phase10c13-v1'):
        folder=ROOT/'reports'/phase;s=read(folder/'artifact-seal.json');private=ROOT/s['private_seal']
        if sha(private)!=s['private_seal_sha256']:raise ValueError('Historical seal changed')
        extra[phase]=dict(public=verify_seal(folder/'SHA256SUMS'),private=verify_seal(private))
    p.update(C11_C13=extra,C20=verify20_seals(),C19=verify19_seals(),C18=verify18_seals(),C17=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),E1_verified=True,D_verified=True,C21_verified=True,original_sha256=sha(BASE/'model.safetensors'),C11_sha256=sha(ROOT/cfg['reference_checkpoint']),model_weight_updates=0,optimizer_steps=0,main_unchanged=subprocess.check_output(['git','rev-parse','main'],text=True).strip())
    if p['original_sha256']!=BASE_SHA or p['C11_sha256']!=cfg['reference_sha256'] or p['main_unchanged']!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Original/reference/main preservation failure')
    if (RUN/'complete.json').exists():
        done=read(RUN/'complete.json')
        if done['forwards']!=28 or done['actual_forwards']!=28 or done['gradient_calculations']>48:raise ValueError('Execution count mismatch')
        for stage in ('A','B'):
            if read(RUN/f'state-{stage}-before.json')!=read(RUN/f'state-{stage}-after.json'):raise ValueError('Stage model changed')
        if len(list((RUN/'forward-intents').glob('*.json')))!=28 or len(list((RUN/'gradient-intents').glob('*.json')))!=done['gradient_calculations']:raise ValueError('Durable operation-intent count mismatch')
        p.update(actual_forwards=28,actual_gradient_calculations=done['gradient_calculations'],stage_A_unchanged=True,stage_B_unchanged=True)
    emit(REPORT/'preservation.json',p);print('E2 preservation PASS',p['historical_data_model_files'],flush=True)

def seal():
    if not (REPORT/'report.md').exists() or read(REPORT/'preservation.json')['status']!='PASS':raise ValueError('Final report/preservation required')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()};path=RUN/'artifact-sha256.json';emit(path,dict(files=files))
    emit(REPORT/'artifact-seal.json',dict(private_manifest=str(path.relative_to(ROOT)),private_manifest_sha256=sha(path),private_files=len(files),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},conservative_allocated_bytes_before_final_seal=allocated(),limits=LIMITS,optimizer_steps=0,training_started=False))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.')]
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode());print('E2 artifact seal written',flush=True)

def check():
    s=read(REPORT/'artifact-seal.json');p=ROOT/s['private_manifest']
    if sha(p)!=s['private_manifest_sha256']:raise ValueError('Private manifest changed')
    for rel,h in {**read(p)['files'],**s['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('Sealed E2 artifact changed: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public seal changed')
    print('E2 final seal PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--postrun',action='store_true');p.add_argument('--seal',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    elif a.seal:seal()
    else:
        tests(a.postrun)
        if a.postrun:preservation()
