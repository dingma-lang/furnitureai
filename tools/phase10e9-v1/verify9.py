"""CPU regression,preservation and sealed metadata,never network or models."""
import argparse,re,subprocess,types
from common9 import *
def tests():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    plans=[('completed',['discover','-s',str(HERE),'-p','test_completed9.py'],ROOT),
           ('E8-geometry',['test_qualification8.GeometryTests','test_qualification8.DecisionTests'],ROOT/'tools/phase10e8-v1')]
    suites=[]
    for name,args,cwd in plans:
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,capture_output=True,timeout=120)
        p=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log');atomic(p,r.stdout+r.stderr);m=re.search(r'Ran (\d+) tests?',p.read_text())
        if r.returncode or not m or '\nOK' not in p.read_text():raise RuntimeError('CPU post-audit failure:'+str(p))
        suites.append(dict(name=name,passed=int(m[1]),path=str(p.relative_to(ROOT)),sha256=sha(p)));print('CPU PASS',name,m[1],flush=True)
    output=REPORT/('post-audit-tests-v2.json' if (REPORT/'post-audit-tests.json').exists() else 'post-audit-tests.json')
    emit(output,dict(status='PASS',passed=sum(s['passed'] for s in suites),pre_network_passed=read(REPORT/'CPU-tests.json')['passed'],
          total_passed=sum(s['passed'] for s in suites)+read(REPORT/'CPU-tests.json')['passed'],suites=suites,new_network_requests=0,GPU_calls=0))
def preservation():
    if sha(MANIFEST)!=PIN:raise ValueError('Frozen manifest changed')
    source=read(OUT/'local-candidate-audit.json')
    if sha(ROOT/source['source'])!=source['source_sha256']:raise ValueError('Original native COCO metadata changed')
    sys.path.insert(0,str(ROOT/'tools/phase10e8-v1'));import verify8
    verify8.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e7-v1'));import verify7
    verify7.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'));import verifyE6
    verifyE6.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e5-v1'));import verify5
    verify5.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'));import verify4
    verify4.check()
    # Older phases also have a module namedcommon9. Run the deep historical
    # checker in a fresh CPU process so this phase's same-named module cannot
    # shadow legacy imports. The child publisher captures data in RAM only.
    script="""import json,sys,types
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'tools/phase10e4-v1'))
import verify4
values=[];scope=dict(verify4.preservation.__globals__)
scope['emit']=lambda p,v:values.append(v)
types.FunctionType(verify4.preservation.__code__,scope)()
if len(values)!=1:raise ValueError('Expected one historical preservation result')
print('E9_HISTORY_JSON='+json.dumps(values[0],sort_keys=True))
"""
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    child=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c',script],cwd=ROOT,env=env,capture_output=True,timeout=180)
    log=REPORT/('CPU-historical-'+uuid.uuid4().hex[:8]+'.log');atomic(log,child.stdout+child.stderr)
    lines=[line for line in child.stdout.decode().splitlines() if line.startswith('E9_HISTORY_JSON=')]
    if child.returncode or len(lines)!=1:raise RuntimeError('Historical CPU integrity subprocess failed:'+str(log))
    emit(REPORT/'historical-preservation.json',json.loads(lines[0].split('=',1)[1]))
    emit(REPORT/'preservation.json',dict(status='PASS',E8_E7_E6_E5_E4_seals_verified=True,historical=read(REPORT/'historical-preservation.json'),
          exact64IDmanifest_unchanged=True,native_annotations_unchanged=True,partitions_unchanged=True,
          original_OWLv2_sha256=sha(BASE/'model.safetensors'),E5_sha256=sha(ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt'),
          new_GPU_forwards=0,model_construction=0,new_labels=0,photograph_downloads=0,photographs_decoded=0,
          read_only_historical_checksum_verification=True,Mac_application_accessed=False,
          main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()))
def seal():
    latest=REPORT/('post-audit-tests-v2.json' if (REPORT/'post-audit-tests-v2.json').exists() else 'post-audit-tests.json')
    if read(REPORT/'preservation.json')['status']!='PASS' or read(latest)['status']!='PASS' or not (REPORT/'report.md').exists():raise ValueError('Report/tests/preservation required')
    emit(REPORT/'artifact-seal.json',dict(private_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and not p.name.startswith('.')},
          code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},photographs=0,GPU_calls=0))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())
def check():
    s=read(REPORT/'artifact-seal.json')
    for rel,h in {**s['private_files'],**s['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E9sealed artifact changed:'+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public E9metadata changed:'+rel)
    if allocated()>DISK:raise ValueError('128MiB storage limit')
    print('E9artifact seal PASS;no network or GPU work')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','preservation','seal','check']);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification')
    globals()[a.action]()
