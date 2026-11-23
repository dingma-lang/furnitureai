"""CPU tests,preservation,isolated artifacts and read-only final checks."""
import argparse,importlib.util,re,subprocess,types
from common8q import *

def tests():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    plans=[('E8',['discover','-s',str(HERE),'-p','test_qualification8.py'],ROOT),
       ('E7-cache',['discover','-s',str(ROOT/'tools/phase10e7-v1'),'-p','test_analysis7.py'],ROOT),
       ('E6-evidence',['discover','-s',str(ROOT/'tools/phase10e6-v1'),'-p','test_auditE6.py'],ROOT),
       ('E1-mask-cache',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1')]
    suites=[]
    for name,args,cwd in plans:
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,capture_output=True,timeout=180)
        log=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log');atomic(log,r.stdout+r.stderr);m=re.search(r'Ran (\d+) tests?',log.read_text())
        if r.returncode or not m or '\nOK' not in log.read_text() or 'skipped=' in log.read_text():raise RuntimeError('CPU suite failed:'+str(log))
        suites.append(dict(name=name,passed=int(m[1]),path=str(log.relative_to(ROOT)),sha256=sha(log)));print('CPU PASS',name,m[1],flush=True)
    emit(REPORT/'CPU-tests.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,GPU_forwards=0))

def preservation():
    for rel,h in read(DATA/'input-binding.json')['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned source changed:'+rel)
    sys.path.insert(0,str(ROOT/'tools/phase10e7-v1'));import verify7
    verify7.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'));import verifyE6
    verifyE6.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e5-v1'));import verify5
    verify5.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'));import verify4
    verify4.check()
    values=dict(verify4.preservation.__globals__);values['emit']=lambda p,v:emit(REPORT/'historical-preservation.json',v)
    types.FunctionType(verify4.preservation.__code__,values)()
    from serve8 import load_dashboard
    payload,sources=load_dashboard()
    if payload['summary']['decisions']!={'unverified_candidate':5,'protected_unsuitable':16} or len(sources)!=10:raise ValueError('Final decision/source integrity mismatch')
    emit(REPORT/'preservation.json',dict(status='PASS',E7_E6_E5_E4_seals_verified=True,historical=read(REPORT/'historical-preservation.json'),
        original_sha256=sha(BASE/'model.safetensors'),E5_sha256=sha(ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt'),
        bound_sources_and_masks_unchanged=True,source_annotations_or_partitions_modified=False,
        evaluation_photographs_decoded_or_mined=False,read_only_historical_file_checksum_checks=True,
        new_GPU_forwards=0,model_construction=0,backward_calls=0,optimizer_updates=0,downloads=0,Mac_application_accessed=False,
        main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()))

def seal():
    if read(REPORT/'CPU-tests.json')['status']!='PASS' or read(REPORT/'preservation.json')['status']!='PASS' or not (REPORT/'report.md').exists():raise ValueError('Passing tests,preservation and report required')
    for component in read(DATA/'dataset-index-v1.json')['components'].values():
        if sha(ROOT/component['path'])!=component['sha256']:raise ValueError('Composed dataset component changed')
    emit(REPORT/'artifact-seal.json',dict(private_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(DATA.rglob('*')) if p.is_file() and not p.name.startswith('.')},
        code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},training_authorized=False,GPU_forwards=0))
    paths=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in paths).encode())

def check():
    s=read(REPORT/'artifact-seal.json')
    for rel,h in {**s['private_files'],**s['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E8sealed artifact changed:'+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public E8report changed:'+rel)
    if allocated()>CAP:raise ValueError('Bounded CPU output cap exceeded')
    print('E8artifact seal PASS;CPU-only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','preservation','seal','check']);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verifier')
    globals()[a.action]()
