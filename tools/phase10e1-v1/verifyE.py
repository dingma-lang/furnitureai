"""CPU tests,historical integrity,reporting and immutable E1 provenance seal."""
import argparse,re
from commonE import *

def verify():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU verification only')
    import torch
    for rel,h in read(BINDING)['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('E1 input changed: '+rel)
    suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in [('E1',['discover','-s',str(HERE),'-p','test_distillationE.py'],ROOT),('C11scope',['test_preflight10.ScopeTests'],ROOT/'tools/phase10c10-v1'),('positive-box-base',['test_ab8.LossTests'],ROOT/'tools/phase10c8-v1'),('source-protections',['test_auto6.GeometryTests','test_auto6.EvidenceTests','test_auto6.MaskTests'],ROOT/'tools/phase10c6-v1')]:
        p=PREP/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with p.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        match=re.search(r'Ran (\d+) tests?',p.read_text())
        if r.returncode or not match or '\nOK' not in p.read_text() or 'skipped=' in p.read_text():raise RuntimeError('CPU test failed: '+str(p))
        suites.append(dict(suite=name,passed=int(match[1]),log=str(p.relative_to(ROOT)),sha256=sha(p)));print('CPU PASS',name,match[1],flush=True)
    emit(REPORT/'tests-v2.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,actual_model_forwards=0,GPU_calls=0,saved_model_updates=0))
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from sealD import check as checkD
    checkD()
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from common21 import verify20_seals,verify19_seals,verify18_seals,verify17_seals,C8,verify_seal
    from verify21 import check_seal
    from verify import preservation
    check_seal();history=preservation();extra={}
    for name in ('phase10c11-v1','phase10c13-v1'):
        folder=ROOT/'reports'/name;s=read(folder/'artifact-seal.json');private=ROOT/s['private_seal']
        if sha(private)!=s['private_seal_sha256']:raise ValueError('Historical private seal changed: '+name)
        extra[name]=dict(public=verify_seal(folder/'SHA256SUMS'),private=verify_seal(private),final_checkpoint=s['final_checkpoint'])
    history.update(C11_C13=extra,C20=verify20_seals(),C19=verify19_seals(),C18=verify18_seals(),C17=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),original_sha256=sha(BASE/'model.safetensors'),D_seal_verified=True,C21_seal_verified=True,training_started=False,GPU_calls=0,model_forwards=0,downloads=0,coefficient=None,training_directory_created=FUTURE.exists(),smoke_directory_created=SMOKE.exists(),main_sha256=subprocess.check_output(['git','rev-parse','main'],text=True).strip())
    if history['original_sha256']!=BASE_SHA or FUTURE.exists() or SMOKE.exists() or torch.cuda.is_initialized():raise ValueError('E1 no-execution/original preservation failure')
    if history['main_sha256']!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Main changed')
    emit(REPORT/'preservation-v2.json',history);print('E1 preservation PASS',history['historical_data_model_files'],flush=True)

def seal():
    if read(REPORT/'tests-v2.json')['status']!='PASS' or read(REPORT/'preservation-v2.json')['status']!='PASS' or not (REPORT/'report.md').exists():raise ValueError('Passing verification and report required')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(PREP.rglob('*')) if p.is_file()};p=PREP/'artifact-sha256.json';emit(p,dict(files=files))
    emit(REPORT/'artifact-seal.json',dict(private_manifest=str(p.relative_to(ROOT)),private_manifest_sha256=sha(p),private_files=len(files),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},conservative_allocated_bytes_before_seal=allocated(),maximum_allocated_bytes=MAX_PREP,model_forwards=0,updates=0,coefficient=None))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.')]
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode());print('E1 sealed',len(public),'reports',flush=True)

def check():
    s=read(REPORT/'artifact-seal.json');p=ROOT/s['private_manifest']
    if sha(p)!=s['private_manifest_sha256']:raise ValueError('E1 private manifest changed')
    for rel,h in {**read(p)['files'],**s['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E1 artifact changed: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('E1 public seal changed: '+rel)
    print('E1 sealed preservation PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.seal:seal()
    elif a.check:check()
    else:verify()
