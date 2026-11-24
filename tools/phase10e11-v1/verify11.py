"""Read-only CPU seal verification, never downloads or model operations."""
import argparse,subprocess
from common11 import *
def seal():
    if read(REPORT/'tests.json')['passed']!=30:raise ValueError('Completed CPU tests required')
    emit(REPORT/'artifact-seal.json',dict(private={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('.')},
      code={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.is_file()},previous_E10_SHA256SUMS=sha(ROOT/'reports/phase10e10-v1/SHA256SUMS'),
      exact_seven_images=True,training_active=False,model_forward_calls=0))
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.') and p.name!='SHA256SUMS').encode())
def check():
    s=read(REPORT/'artifact-seal.json')
    for rel,h in {**s['private'],**s['code']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('E11 artifact changed:'+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(ROOT/rel)!=h:raise ValueError('Public report changed:'+rel)
    if sha(ROOT/'reports/phase10e10-v1/SHA256SUMS')!=s['previous_E10_SHA256SUMS']:raise ValueError('Previous phase changed')
    for r in read(OUT/'acquisition-closed.json')['records']:
        if sha(ROOT/r['file'])!=r['sha256']:raise ValueError('Photograph bytes changed')
    if allocated()>DISK:raise ValueError('64MiB cap')
    subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'tools/phase10e10-v1/verify10.py'),'check'],check=True,timeout=30)
    print('E11 photograph/metadata/code seal PASS; allocated new bytes:',allocated())
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check']);a=p.parse_args();globals()[a.action]()
