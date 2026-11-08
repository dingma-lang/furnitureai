"""Relevant no-download CPU regressions, per-suite module isolation."""
import re
import subprocess
import sys
from pilot_core import *

def run():
 suites=['phase7-images-v1','phase7-rights-v1','phase7-acquisition-v1','phase7-v1','phase5-v1','phase5-metadata-v1','phase5-images-v1','phase5b-v1','phase5c-v1','phase6b-v1','phase6c-v1','phase6d-v1'];results=[]
 for name in suites:
  log=OUT/f'tests-{name}.log'
  with log.open('x') as f:p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test_*.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  text=log.read_text();m=re.search(r'Ran (\d+) tests?',text);results.append(dict(suite=name,tests=int(m[1]) if m else None,returncode=p.returncode,log=log.name));print(name,results[-1]['tests'],'PASS' if p.returncode==0 else 'FAIL',flush=True)
  if p.returncode or 'skipped=' in text:raise RuntimeError('Test failure or unexpected skip '+name)
 commit(OUT/'test-results.json',dict(status='PASS',total_tests=sum(r['tests'] for r in results),suites=results,no_network_downloads=True,no_project_inference_training_or_scoring=True))

if __name__=='__main__':run()
