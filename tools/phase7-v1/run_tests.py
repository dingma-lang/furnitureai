"""Record CPU/offline regression suites, each in its own module namespace."""
import re
import subprocess
import sys
from contracts import ROOT,OUT,write_new

SUITES=['phase7-v1','phase5-v1','phase5-metadata-v1','phase5-images-v1','phase5b-v1','phase5c-v1','phase6b-v1','phase6c-v1','phase6d-v1']

def run():
    results=[]
    for name in SUITES:
        log=OUT/f'tests-{name}.log'
        with log.open('x') as f:
            result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test_*.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=log.read_text();match=re.search(r'Ran (\d+) tests?',text)
        results.append(dict(suite=name,returncode=result.returncode,tests=int(match[1]) if match else None,log=log.name))
        print(name,result.returncode,results[-1]['tests'],flush=True)
        if result.returncode:raise RuntimeError('Regression suite failed '+name)
    write_new(OUT/'test-results.json',dict(status='PASS',suites=results,total_tests=sum(r['tests'] for r in results),network_in_dry_runs=False,model_inference_training=False))

if __name__=='__main__':run()
