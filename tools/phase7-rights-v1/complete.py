"""Finish local-only reports/tests/seals after the paced source queue has checkpointed."""
import subprocess
import sys
import time
from audit_core import *

def run():
 print('Waiting for the rights-audit completion checkpoint; no extra network requests',flush=True)
 while not (OUT/'rights-audit.json').exists():
  if (OUT/'STOP-FINALIZATION').exists():raise RuntimeError('Finalization explicitly stopped')
  time.sleep(5)
 for script in ('finalize.py','run_tests.py','verify.py','report.py'):
  subprocess.run([sys.executable,'-B',str(HERE/script)],cwd=ROOT,check=True)
 summary=read(OUT/'summary.json');tests=read(OUT/'test-results.json');verified=read(OUT/'integrity-verification.json')
 if verified['status']!='PASS' or tests['status']!='PASS':raise ValueError('Final checks did not pass')
 implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
 commit(OUT/'implementation-sha256.json',implementation)
 commit(OUT/'completion.json',dict(status='rights_audit_complete_with_explicit_coverage_and_eligibility',completed_utc=now(),
  manifest_sha256=PIN,candidates=400,attempted=summary['attempted_candidates'],unattempted=summary['unattempted_candidates'],
  licensing_status_counts=summary['licensing_status_counts'],eligible_proposed_photos=summary['eligible_proposed_photos'],
  actual_body_bytes=summary['actual_body_bytes'],conservative_charge=summary['conservative_charge'],tests_passed=tests['total_tests'],
  existing_registration_photos=10,existing_inactive_annotations=33,photograph_bodies_downloaded=0,source_geometry_or_partitions_modified=False,
  model_training_inference_or_rendering=False,report='report.md',next_approval='next-approval-request.json',mac_project_accessed=False))
 resource=OUT/'resource-usage.json';commit(resource,dict(status='measuring'))
 seal=OUT/'ARTIFACT-SHA256SUMS'
 def seals():return ''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p!=seal)
 with seal.open('x') as f:f.write(seals())
 for _ in range(2):
  extra=OUT.parent.lstat().st_blocks*512
  usage=dict(total_new_allocated_bytes=allocated()+extra,cap_bytes=DISK,self_and_seal_included=True,
    accounting='Linux st_blocks*512 over new rights-audit tree, isolated tool tree and new rights-audits parent directory',
    actual_downloaded_response_body_bytes=summary['actual_body_bytes'],conservative_response_body_charge=summary['conservative_charge'],transfer_cap_bytes=CAP,
    photo_bodies=0,source_archive_extraction=False)
  resource.write_text(json.dumps(usage,indent=2)+'\n');seal.write_text(seals())
 if allocated()+OUT.parent.lstat().st_blocks*512!=usage['total_new_allocated_bytes'] or usage['total_new_allocated_bytes']>DISK:raise ValueError('Final storage allocation mismatch/limit')
 for line in seal.read_text().splitlines():
  expected,relative=line.split('  ',1)
  if sha(OUT/relative)!=expected:raise ValueError('New artifact seal mismatch')
 for filename,digest in implementation.items():
  if sha(ROOT/filename)!=digest:raise ValueError('Final implementation seal mismatch')
 print('COMPLETE',json.dumps(dict(attempted=summary['attempted_candidates'],unattempted=summary['unattempted_candidates'],
  verified=summary['verified_licensing_candidates'],eligible=summary['eligible_proposed_photos'],actual_body_bytes=summary['actual_body_bytes'],
  allocated_bytes=usage['total_new_allocated_bytes'],tests=tests['total_tests'])),flush=True)

if __name__=='__main__':run()
