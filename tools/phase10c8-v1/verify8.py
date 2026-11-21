"""CPU regression suites and byte-level preservation; never constructs a model."""
import subprocess,re,importlib.util
from common8 import *

def run(post_run=False):
    results=[]
    for suite in ('phase10c8-v1','phase10c7-v1','phase10c6-v1','phase10c5-v2','phase10c4-v1','phase10c3-v1','phase10c2-v1','phase10c-v1','phase10b-v1','phase10a-v1','phase5c-v1'):
        log=RUN/('tests-'+suite+'.log')
        if not log.exists():
            with log.open('x') as f:
                result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/suite),'-p','test*.py','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
            if result.returncode:raise RuntimeError('CPU tests failed: '+str(log))
        value=log.read_text();n=re.search(r'Ran (\d+) tests?',value)
        if not n or 'skipped=' in value or not value.rstrip().endswith('OK'):raise RuntimeError('CPU tests failed: '+str(log))
        results.append(dict(suite=suite,passed=int(n[1]),log_sha256=sha(log)));print('PASS',suite,n[1],flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('historical_preservation_c8',ROOT/'tools/phase10c-v1/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    preservation=m.preservation()
    extra=[]
    for phase in ('phase10c5-v1','phase10c5-v2','phase10c6-v1','phase10c7-query-smoke-v1'):
        folders=[ROOT/'data/open-images/preparation'/phase,ROOT/'data/open-images/experiments'/phase]
        for folder in folders:
            for name in ('INITIAL-SHA256SUMS','SHA256SUMS'):
                p=folder/name
                if not p.exists():continue
                count=0
                for line in p.read_text().splitlines():
                    expected,rel=line.split('  ',1)
                    target=Path(rel) if Path(rel).is_absolute() else ((folder/rel) if phase=='phase10c5-v1' else ROOT/rel)
                    if sha(target)!=expected:raise ValueError('Historical evidence seal changed: '+str(target))
                    count+=1
                extra.append(dict(seal=str(p.relative_to(ROOT)),checked=count,sha256=sha(p)))
    subprocess.run(['git','diff','--exit-code','7bcdc90','--','tools/phase10c7-v1','reports/phase10c7-v1'],cwd=ROOT,check=True)
    verify_binding(read(RUN/'binding.json'))
    suffix='-post-run' if post_run else ''
    save(REPORT/('tests'+suffix+'.json'),dict(status='PASS',suites=results,passed=sum(r['passed'] for r in results)))
    save(REPORT/('preservation'+suffix+'.json'),dict(status='PASS',historical=preservation,extra_evidence_seals=extra,source_binding_reverified=True,main_commit=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),private_images_uploaded=False))
    print('HISTORICAL PRESERVATION PASS',preservation['historical_data_model_files'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--post-run',action='store_true');run(p.parse_args().post_run)
