"""CPU-only post-discovery verification, no transport entry point or model loading."""
import argparse,subprocess
from common10 import *
def tests():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERE),'-p','test_discovery10.py','-v'],cwd=ROOT,env=env,capture_output=True,timeout=60)
    atomic(REPORT/'CPU-tests.log',r.stdout+r.stderr)
    import re
    n=re.search(r'Ran (\d+) tests',r.stderr.decode())
    if r.returncode or not n:raise RuntimeError('CPU tests failed; inspect new report log')
    emit(REPORT/'tests.json',dict(status='PASS',passed=int(n[1]),network_in_tests=0,model_loads=0,GPU_calls=0,log_sha256=sha(REPORT/'CPU-tests.log')))
def seal():
    if read(REPORT/'tests.json')['status']!='PASS':raise ValueError('Tests required')
    emit(REPORT/'artifact-seal.json',dict(code={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')},
      private={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('.')},
      preserved_E9_SHA256SUMS=sha(ROOT/'reports/phase10e9-v1/SHA256SUMS'),
      model_constructed=False,new_labels=0,photo_bodies=0,closed_network=True))
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS').encode())
def check():
    s=read(REPORT/'artifact-seal.json')
    for rel,h in {**s['code'],**s['private']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('Changed E10artifact: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Changed public export: '+rel)
    if sha(ROOT/'reports/phase10e9-v1/SHA256SUMS')!=s['preserved_E9_SHA256SUMS']:raise ValueError('Previous phase changed')
    if allocated()>DISK:raise ValueError('Storage cap')
    subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'tools/phase10e9-v1/verify9.py'),'check'],check=True,timeout=30)
    print('E10 sealed metadata/code PASS; E9 immutable; allocated new bytes:',allocated())
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','seal','check']);a=p.parse_args();globals()[a.action]()
