"""CPU regressions, original-model/history seals and reproducible evidence audit."""
import importlib.util,os,re,subprocess,uuid
from common6 import *

def run():
    suites=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for name in ('phase10c6-v1','phase10c5-v2','phase10c5-v1','phase10c4-v1','phase10c3-v1','phase10c2-v1','chair-fp-audit-v1','phase10c-v1','phase10b-v1','phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        log=RAW/('tests-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();n=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not n or 'skipped=' in text:raise RuntimeError('CPU suite failed: '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('PASS',name,n[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('auto6_prior_verify',ROOT/'tools/phase10c-v1/verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);preservation=mod.preservation()
    for rel,seal in (('data/open-images/experiments/phase10c-recovery-v1','SHA256SUMS'),('reports/chair-fp-investigation-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1/submissions/manual-v1','SHA256SUMS'),('data/open-images/preparation/phase10c3-v1','SHA256SUMS'),('data/open-images/preparation/phase10c4-v1','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c5-v1','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c5-v2','INITIAL-SHA256SUMS')):
        folder=ROOT/rel;count=0
        for line in (folder/seal).read_text().splitlines():
            value,path=line.split('  ',1);p=(ROOT/path).resolve() if rel.endswith('phase10c5-v2') else (folder/path).resolve()
            allowed=p.is_relative_to(folder) or (rel.endswith('phase10c5-v1') and p.is_relative_to(ROOT/'reports/phase10c5-v1')) or (rel.endswith('phase10c5-v2') and p.is_relative_to(ROOT/'reports/phase10c5-v2'))
            if not allowed or sha(p)!=value:raise ValueError('Historical seal changed: '+str(p))
            count+=1
        preservation[rel+'_sealed_files']=count
    for rel in ('tools/phase10c5-v2','tools/phase10c5-v1','reports/phase10c5-v2','reports/phase10c5-v1'):
        subprocess.run(['git','diff','--exit-code','3b074e9','--',rel],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
    source=read(RAW/'training-evidence-manifest.json');original=[r for r in manifest()['rows'] if optimizer_member(r)]
    if source['training_rows']!=original or len(original)!=59 or sum(not a['ignore'] for r in original for a in r['annotations'])!=182:raise ValueError('Original source positives/partition changed')
    from masks6 import validate_bundle
    frozen=candidates();observations={r['candidate_id']:read(RAW/'observations'/(r['candidate_id']+'.json')) for r in frozen};validate_bundle(read(RAW/'source-negative-evidence.json'),manifest()['rows'],frozen,observations)
    save(REPORT/'preservation.json',preservation);save(REPORT/'tests.json',dict(status='PASS',CPU_tests=sum(s['passed'] for s in suites),suites=suites,model_forwards=0,GPU_forwards=0,optimizer_updates=0,human_reviews_created=0))
    print('PRESERVATION PASS',preservation['historical_data_model_files'],'files;',preservation['phase10b_snapshots'],'snapshots;source positives unchanged',flush=True)

if __name__=='__main__':run()
