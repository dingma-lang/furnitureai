"""Offline CPU suites into this new preparation report, never historical output directories."""
import re
import subprocess
import sys
from prep_core import *

def run():
 names=['phase8a-v1','phase7-images-v1','phase7-rights-v1','phase7-acquisition-v1','phase7-v1','phase5-v1','phase5-metadata-v1','phase5-images-v1','phase5b-v1','phase5c-v1','phase6b-v1','phase6c-v1','phase6d-v1'];records=[]
 for name in names:
  log=OUT/f'tests-{name}.log'
  with log.open('x') as f:p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test_*.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  text=log.read_text();match=re.search(r'Ran (\d+) tests?',text);records.append(dict(suite=name,tests=int(match[1]) if match else None,returncode=p.returncode,log=log.name))
  print(name,records[-1]['tests'],'PASS' if p.returncode==0 else 'FAIL',flush=True)
  if p.returncode or 'skipped=' in text:raise ValueError('Test failure/skip '+name)
 commit(OUT/'test-results.json',dict(status='PASS',total_tests=sum(r['tests'] for r in records),suites=records,network_requests=0,project_model_updates=0))

if __name__=='__main__':run()
