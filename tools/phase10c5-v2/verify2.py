"""CPU regressions and read-only preservation; write reports only in v2."""
import importlib.util,os,re,subprocess
from review2 import *
from dataset2 import compile_dataset,category_masks

def run():
    suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for name in ('phase10c5-v2','phase10c5-v1','phase10c4-v1','phase10c3-v1','phase10c2-v1','chair-fp-audit-v1','phase10c-v1','phase10b-v1','phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        log=RAW/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        n=re.search(r'Ran (\d+) tests?',log.read_text())
        if r.returncode or not n or 'skipped=' in log.read_text():raise RuntimeError('CPU test failed: '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('PASS',name,n[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('v2_prior_verifier',ROOT/'tools/phase10c-v1/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);preserved=m.preservation()
    for rel,seal in (('data/open-images/experiments/phase10c-recovery-v1','SHA256SUMS'),('reports/chair-fp-investigation-v1','SHA256SUMS'),
        ('data/open-images/preparation/phase10c2-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1/submissions/manual-v1','SHA256SUMS'),
        ('data/open-images/preparation/phase10c3-v1','SHA256SUMS'),('data/open-images/preparation/phase10c4-v1','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c5-v1','INITIAL-SHA256SUMS')):
        folder=ROOT/rel;count=0
        for line in (folder/seal).read_text().splitlines():
            value,path=line.split('  ',1);p=(folder/path).resolve()
            if not(p.is_relative_to(folder) or (rel.endswith('phase10c5-v1') and p.is_relative_to(ROOT/'reports/phase10c5-v1'))) or sha(p)!=value:raise ValueError('Historical seal changed: '+str(p))
            count+=1
        preserved[rel+'_sealed_files']=count
    for rel in ('tools/phase10c5-v1','reports/phase10c5-v1','tools/phase10c4-v1'):
        subprocess.run(['git','diff','--exit-code','85c3cc5','--',rel],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    reg=prior.registry();payload=read(RAW/'empty-review.json');d=compile_dataset(payload,reg,verify_files=True)
    if digest(d)!=digest(read(RAW/'supervision-manifest-initial-v2.json')):raise ValueError('Compiled overlay is not reproducible')
    for row in d['rows']:
        _,mask=category_masks([[.01,.01,.02,.02]],row)
        if any(mask[0]):raise ValueError('Empty human review created absence targets')
    save(REPORT/'preservation.json',preserved)
    save(REPORT/'tests.json',dict(status='PASS',CPU_tests=sum(s['passed'] for s in suites),suites=suites,GPU_forwards=0,optimizer_updates=0,real_human_decisions_created=0))
    print('PRESERVATION PASS',preserved['historical_data_model_files'],'files;reproducible overlay;0newnegative targets',flush=True)

if __name__=='__main__':run()
