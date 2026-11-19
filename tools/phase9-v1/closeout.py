"""Tests, preservation and reporting only. Does not launch models or optimisation."""
import re
import subprocess
import sys
from core import *

def run():
 tests=[]
 for suite in ('phase9-v1','phase5b-v1','phase5c-v1'):
  log=OUT/('tests-'+suite+'.log')
  with log.open('x') as f:p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/suite),'-p','test_*.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  text=log.read_text();n=re.search(r'Ran (\d+) tests?',text)
  if p.returncode or not n or 'skipped=' in text:raise ValueError('Required suite failed '+suite)
  tests.append(dict(suite=suite,passed=int(n[1]),status='PASS'));print('TESTS',suite,n[1],'PASS',flush=True)
 save(OUT/'tests.json',dict(passed=sum(t['passed'] for t in tests),suites=tests,status='PASS',project_model_training=False))
 for script in ('verify.py','gallery.py','report.py'):
  subprocess.run([sys.executable,'-B',str(HERE/script)],cwd=ROOT,check=True)
 total=space();selection=checked(OUT/'selection.json');save(OUT/'completion.json',dict(status='COMPLETE_INFERENCE_ONLY',selected=selection['selected'],
  qualified_on_exposed=selection['robust_improvement_on_exposed_protocol'],training_updates=0,production_defaults_changed=False,
  final36_comparison_count=1,final_results_not_used_to_retune=True,frozen_photo_reads=0,network_dataset_or_weight_downloads=0,
  tests_passed=sum(t['passed'] for t in tests),allocated_bytes_before_completion_and_seal=total,storage_cap_bytes=536870912,
  branch='experiment/phase9-multiview-detection',code_commit_and_remote_receipt='Recorded separately after Git commit/push; not a checkpoint or dataset backup'))
 save(OUT/'implementation-sha256.json',{p.name:sha(p) for p in sorted(HERE.glob('*')) if p.is_file()})
 seal=OUT/'ARTIFACT-SHA256SUMS'
 with seal.open('x') as f:
  for p in sorted(OUT.rglob('*')):
   if p.is_file() and p!=seal:f.write(sha(p)+'  '+str(p.relative_to(OUT))+'\n')
 print('COMPLETE; allocated bytes',space(),'; no production switch/training/merge',flush=True)

if __name__=='__main__':run()
