"""CPU/no-network regression suites, isolated module namespaces and fresh audit logs."""
import re
import subprocess
import sys
from audit_core import *

def run():
 if not (OUT/'summary.json').exists():raise ValueError('Finalize audit before final complete-suite run')
 suites=['phase7-rights-v1','phase7-acquisition-v1','phase7-v1','phase5-v1','phase5-metadata-v1','phase5-images-v1','phase5b-v1','phase5c-v1','phase6b-v1','phase6c-v1','phase6d-v1'];records=[]
 for name in suites:
  log=OUT/f'tests-{name}.log'
  with log.open('x') as f:r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test_*.py','-v'],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=180)
  text=log.read_text();match=re.search(r'Ran (\d+) tests?',text);record=dict(suite=name,tests=int(match[1]) if match else None,returncode=r.returncode,log=log.name);records.append(record)
  print(name,record['tests'],'PASS' if r.returncode==0 else 'FAIL',flush=True)
  if r.returncode or 'skipped=' in text:raise ValueError('Final suite failed or unexpectedly skipped '+name)
 commit(OUT/'test-results.json',dict(status='PASS',total_tests=sum(r['tests'] for r in records),suites=records,project_training_inference_or_scoring=False,network_downloads_in_tests=False))

if __name__=='__main__':run()
