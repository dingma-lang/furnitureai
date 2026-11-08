"""CPU-compatible regression execution into this new metadata audit only."""
import re
import subprocess
import sys
from core import *

def run():
 suites=['phase7-acquisition-v1','phase7-v1','phase5-v1','phase5-metadata-v1','phase5-images-v1','phase5b-v1','phase5c-v1','phase6b-v1','phase6c-v1','phase6d-v1']
 records=[]
 for name in suites:
  path=OUT/f'tests-{name}.log'
  with path.open('x') as f:r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test_*.py','-v'],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=180)
  m=re.search(r'Ran (\d+) tests?',path.read_text())
  record=dict(suite=name,tests=int(m[1]) if m else None,returncode=r.returncode,log=path.name);records.append(record)
  print(name,record['tests'],'PASS' if r.returncode==0 else 'FAIL',flush=True)
  if r.returncode:raise RuntimeError('Regression failure '+name)
 commit(OUT/'test-results.json',dict(status='PASS',total_tests=sum(r['tests'] for r in records),suites=records,project_training_inference_or_scoring=False,network_downloads_in_tests=False))

if __name__=='__main__':run()
