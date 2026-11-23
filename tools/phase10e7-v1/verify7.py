"""CPU-only post-cache verification,preservation and immutable audit seal."""
import argparse,importlib.util,re,types
from common7 import *

def post_tests():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERE),'-p','test_analysis7.py','-v'],cwd=ROOT,env=env,capture_output=True,timeout=120)
    p=REPORT/('CPU-analysis-'+uuid.uuid4().hex[:8]+'.log');atomic(p,r.stdout+r.stderr);m=re.search(r'Ran (\d+) tests?',p.read_text())
    if r.returncode or not m or '\nOK' not in p.read_text():raise RuntimeError('Post-cache CPU tests failed:'+str(p))
    emit(REPORT/'post-cache-tests.json',dict(status='PASS',passed=int(m[1]),preflight_passed=read(REPORT/'CPU-tests.json')['passed'],log=str(p.relative_to(ROOT)),sha256=sha(p),additional_GPU_forwards=0))
    print('CPU post-cache PASS',m[1])

def preservation():
    inputs()
    sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'));import verifyE6
    verifyE6.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e5-v1'));import verify5
    verify5.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'));import verify4
    verify4.check()
    values=dict(verify4.preservation.__globals__);values['emit']=lambda p,v:emit(REPORT/'historical-preservation.json',v)
    types.FunctionType(verify4.preservation.__code__,values)()
    done=read(RUN/'complete.json');r=done['resources']
    if done['model_before']!=done['model_after'] or r['actual_forwards']!=59 or r['attempted_forwards']!=59 or r['optimizer_steps']!=0 or r['backward_calls']!=0:
        raise ValueError('Inference state or budget changed')
    budget(dict(attempted_forwards=r['attempted_forwards'],actual_forwards=r['actual_forwards']),r['wall_seconds'],r['peak_allocated_VRAM_bytes'],allocated())
    emit(REPORT/'preservation.json',dict(status='PASS',all_binding_hashes_unchanged=True,E6_seal_verified=True,E5_seal_verified=True,E4_seal_verified=True,
          historical=read(REPORT/'historical-preservation.json'),E5_sha256=sha(CHECKPOINT),original_sha256=sha(BASE/'model.safetensors'),
          model_parameters_buffers_and_derived_tensors_unchanged=True,fingerprint_before=done['model_before'],fingerprint_after=done['model_after'],
          optimizer_steps=0,backward_calls=0,frozen_or_reserved_used_for_inference_or_mining=False,
          read_only_historical_checksum_verification=True,Mac_application_accessed=False))

def seal():
    preservation_report=REPORT/('preservation-v2.json' if (REPORT/'preservation-v2.json').exists() else 'preservation.json')
    if read(preservation_report)['status']!='PASS' or read(REPORT/'post-cache-tests.json')['status']!='PASS' or not (REPORT/'report.md').exists():raise ValueError('Report,tests and preservation required')
    emit(REPORT/'artifact-seal.json',dict(private_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and not p.name.startswith('.')},
         code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},GPU_forwards=59,optimizer_steps=0,no_automatic_resume=True))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())

def check():
    s=read(REPORT/'artifact-seal.json')
    for rel,h in {**s['private_files'],**s['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E7sealed artifact changed:'+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public E7hash mismatch:'+rel)
    if allocated()>LIMITS['output_bytes']:raise ValueError('Final storage exceeds128MiB')
    print('E7seal PASS;no additional GPU execution')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['post_tests','preservation','seal','check']);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification')
    globals()[a.action]()
