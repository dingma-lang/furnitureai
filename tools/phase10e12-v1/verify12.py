"""Read-only E12 verification; optional full rehash never loads images or models."""
import argparse,json,os,subprocess,sys
from audit12 import ROOT,OUT,digest,emit
CHANGED=('data','dataset_quality','evaluation','geometry','postprocessing')
MAIN='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6'
def seal():
    if json.load(open(OUT/'preservation-comparison.json'))['status']!='PASS':raise ValueError('Preservation required')
    files=[*sorted((ROOT/'tools/phase10e12-v1').glob('*.py')),ROOT/'tests/test_audit_regressions.py',*[ROOT/'src/furniture_bench'/(n+'.py') for n in CHANGED]]
    private=[p for p in (OUT/'private').iterdir() if p.is_file()]
    emit(OUT/'artifact-seal.json',dict(code={str(p.relative_to(ROOT)):digest(p) for p in files},private_audit_indexes={str(p.relative_to(ROOT)):digest(p) for p in private},main=MAIN,
         original_commit='5818ab333dfd239d581129e02af93238bc85e191',model_or_dataset_changes=0))
    public=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in public:f.write(digest(p)+'  '+str(p.relative_to(ROOT))+'\n')
def check(full=False):
    s=json.load(open(OUT/'artifact-seal.json'))
    for rel,h in {**s['code'],**s['private_audit_indexes']}.items():
        if digest(ROOT/rel)!=h:raise ValueError('Changed E12 artifact:'+rel)
    for line in (OUT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1)
        if digest(ROOT/rel)!=h:raise ValueError('Changed public report:'+rel)
    if subprocess.check_output(['git','rev-parse','main'],text=True).strip()!=MAIN:raise ValueError('Main changed')
    if full:
        rows=json.load(open(OUT/'private/protected-after.json'))['records']
        for r in rows:
            if digest(ROOT/r['path'])!=r['sha256']:raise ValueError('Protected content changed:'+r['path'])
        print('Full protected rehash PASS:',len(rows),'files')
    print('E12 code/report/index seals PASS;main unchanged; no model operations')
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only verification')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check']);p.add_argument('--full',action='store_true');a=p.parse_args()
    if a.action=='seal':seal()
    else:check(a.full)
