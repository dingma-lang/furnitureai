"""CPU regression suites, hash preservation and empty-overlay preflight."""
import importlib.util
import os,re,subprocess,uuid
from g5_common import *

def run():
    suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for name in ('phase10c5-v1','phase10c4-v1','phase10c3-v1','phase10c2-v1','chair-fp-audit-v1','phase10c-v1','phase10b-v1','phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        log=RAW/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        body=log.read_text();n=re.search(r'Ran (\d+) tests?',body)
        if r.returncode or not n or 'skipped=' in body:raise RuntimeError('CPU suite failed: '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('PASS',name,n[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('g5_previous_verify',ROOT/'tools/phase10c-v1/verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=mod.preservation()
    for rel,seal in (('data/open-images/experiments/phase10c-recovery-v1','SHA256SUMS'),('reports/chair-fp-investigation-v1','SHA256SUMS'),
        ('data/open-images/preparation/phase10c2-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1/submissions/manual-v1','SHA256SUMS'),
        ('data/open-images/preparation/phase10c3-v1','SHA256SUMS'),('data/open-images/preparation/phase10c4-v1','INITIAL-SHA256SUMS')):
        folder=ROOT/rel;count=0
        for line in (folder/seal).read_text().splitlines():
            value,path=line.split('  ',1);p=(folder/path).resolve()
            if not p.is_relative_to(folder) or sha(p)!=value:raise ValueError('Historical seal changed: '+rel+'/'+path)
            count+=1
        result[rel+'_sealed_files']=count
    from g5_review import registry,validate,compare_masks
    reg=registry();empty=read(RAW/'empty-review.json');v=validate(empty,reg);m=compare_masks([])
    if v['accepted_hard_negatives'] or m['changed_mask_terms']:raise ValueError('Unreviewed data acquired training negatives')
    save_versioned(RAW/'initial-mask-comparison.json',m)
    save_versioned(REPORT/'initial-mask-summary.json',{k:v for k,v in m.items() if k!='records'})
    save_versioned(REPORT/'preservation-summary.json',result)
    save_versioned(REPORT/'tests.json',dict(status='PASS',CPU_tests=sum(x['passed'] for x in suites),suites=suites,GPU_forwards=0,optimizer_updates=0,real_human_reviews_created=0))
    print('PRESERVATION PASS',result['historical_data_model_files'],result['phase10b_snapshots'],';empty overlay changes0terms',flush=True)

if __name__=='__main__':run()
