"""CPU tests,preservation and immutable E6 seals. No GPU execution path."""
import argparse,importlib.util,re,types
from commonE6 import *

def tests():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    plans=[('E6',['discover','-s',str(HERE),'-p','test_*.py'],ROOT),
           ('E5',['discover','-s',str(ROOT/'tools/phase10e5-v1'),'-p','test_experiment5.py'],ROOT),
           ('E1-mask-cache',['test_distillationE.MaskLossTests','test_distillationE.CacheTests'],ROOT/'tools/phase10e1-v1'),
           ('C11-scope',['test_preflight10.ScopeTests'],ROOT/'tools/phase10c10-v1'),
           ('metrics',['test_phase10c.MetricTests'],ROOT/'tools/phase10c-v1')]
    suites=[]
    for name,args,cwd in plans:
        log=REPORT/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,capture_output=True,timeout=180)
        atomic(log,result.stdout+result.stderr);text=log.read_text();m=re.search(r'Ran (\d+) tests?',text)
        if result.returncode or not m or '\nOK' not in text or 'skipped=' in text:raise RuntimeError('CPU tests failed:'+str(log))
        suites.append(dict(name=name,passed=int(m[1]),log=str(log.relative_to(ROOT)),sha256=sha(log)))
        print('PASS',name,m[1],flush=True)
    summary=REPORT/'CPU-tests.json';version=1
    while summary.exists():
        version+=1;summary=REPORT/f'CPU-tests-v{version}.json'
    emit(summary,dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,GPU_calls=0))

def preservation():
    sys.path.insert(0,str(ROOT/'tools/phase10e5-v1'))
    import verify5
    verify5.check()
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
    import verify4
    verify4.check()
    # Only the report publisher is redirected;all historical sources and
    # hashes are read through their unchanged original verification functions.
    values=dict(verify4.preservation.__globals__)
    def publish_unchanged(path,value):
        if path.exists():
            if read(path)!=value:raise ValueError('Completed verification record changed:'+str(path))
        else:emit(path,value)
    values['emit']=lambda path,value:publish_unchanged(REPORT/'historical-preservation.json',value)
    types.FunctionType(verify4.preservation.__code__,values)()
    from validateE6 import check
    validation=check()
    publish_unchanged(REPORT/'dataset-integrity.json',validation)
    emit(REPORT/'preservation.json',dict(status='PASS',historical=read(REPORT/'historical-preservation.json'),
        E4_seal_verified=True,E5_seal_verified=True,new_dataset_integrity=validation,
        original_sha256=sha(BASE/'model.safetensors'),unchanged_main=subprocess.check_output(['git','rev-parse','main'],text=True).strip(),
        downloads=0,model_inference=0,training_updates=0,Mac_application_accessed=False))

def seal():
    test_summary=max(REPORT.glob('CPU-tests*.json'),key=lambda p:p.stat().st_mtime_ns)
    if read(REPORT/'preservation.json')['status']!='PASS' or read(test_summary)['status']!='PASS' or not (REPORT/'report.md').exists():
        raise ValueError('Final report,preservation and passing tests required')
    private={str(p.relative_to(ROOT)):sha(p) for p in sorted(DATA.rglob('*')) if p.is_file() and not p.name.startswith('.')}
    emit(REPORT/'artifact-seal.json',dict(private_files=private,
          code={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},
          new_downloads=0,new_model_inference=0,optimizer_updates=0))
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS']
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())

def check():
    sealed=read(REPORT/'artifact-seal.json')
    for rel,h in {**sealed['private_files'],**sealed['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E6 sealed artifact changed:'+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public E6 report changed:'+rel)
    print('E6 seal PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','preservation','seal','check']);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only execution required')
    globals()[a.action]()
