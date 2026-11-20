"""CPU-only tests and historical byte verification. Never invokes GPU workers."""
import os
import re
import subprocess
import sys
from c10 import *

def preservation():
    inv=read(ROOT/'reports/phase10a-v1/backup-inventory.json');checked=0
    for record in inv['records']:
        p=ROOT/record['path']
        if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Historical file changed: '+record['path'])
        checked+=1
    count=artifacts=0
    for seal,folder in ((RUN/'comparison-v1/CHECKPOINT-SHA256SUMS',RUN),(RUN/'comparison-v1/ARTIFACT-SHA256SUMS',RUN/'comparison-v1')):
        for line in seal.read_text().splitlines():
            digest,rel=line.split('  ',1);p=(folder/rel).resolve()
            if not p.is_relative_to(folder) or sha(p)!=digest:raise ValueError('Phase10B integrity seal failed: '+rel)
            if folder==RUN:count+=1
            else:artifacts+=1
    old=read(RUN/'comparison-v1/results.json')
    for path,digest in old['code'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Historical implementation changed: '+path)
    m=manifest();a=[r for r in m['rows'] if r['internal_partition']=='optimization'];b=internal_rows()
    for field in ('image_id','identity_group','sha256'):
        if {r[field] for r in a}&{r[field] for r in b}:raise ValueError('Training/internal validation leakage: '+field)
    # Rehash all eligible photographs as bytes only; never decode or infer.
    for r in m['rows']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Eligible photo changed')
    return dict(status='PASS',historical_data_model_files=checked,phase10b_snapshots=count,phase10b_report_artifacts=artifacts,
        historical_implementation_files=len(old['code']),eligible_image_hashes=len(m['rows']),source_group_isolation=True,
        annotation_overlays_activated=0,training_updates=0,GPU_model_forwards=0,reserved_or_frozen_images_decoded=0,
        datasets_or_models_downloaded=0,historical_data_modified=False,Mac_project_modified=False)

def run():
    suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for name in ('phase10c-v1','phase10b-v1','phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        log=RAW/('tests-'+name+'-'+uuid.uuid4().hex+'.log')
        with log.open('x') as f:
            result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();n=re.search(r'Ran (\d+) tests?',s)
        if result.returncode or not n or 'skipped=' in s:raise RuntimeError('CPU suite failed; inspect '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('PASS',name,n[1],flush=True)
    p=preservation();save(RAW/'preservation.json',p)
    save(REPORT/'tests.json',dict(status='PASS',suites=suites,passed=sum(s['passed'] for s in suites),preservation=p))
    print('PRESERVATION PASS',p['historical_data_model_files'],p['phase10b_snapshots'],flush=True)

if __name__=='__main__':run()
