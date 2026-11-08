"""Local-only finalisation; requires a completed source-queue checkpoint."""
import subprocess
from audit_core import *

def run():
 if not (OUT/'rights-audit.json').exists():raise RuntimeError('Source queue has not checkpointed; no extra requests are made here')
 for name in ('finalize.py','verify.py','report.py'):
  subprocess.run([sys.executable,'-B',str(HERE/name)],cwd=ROOT,check=True)
 summary=read(OUT/'summary.json');tests=read(OUT/'test-results.json')
 if tests['status']!='PASS' or read(OUT/'integrity-verification.json')['status']!='PASS':raise ValueError('Required checks failed')
 implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
 commit(OUT/'implementation-sha256.json',implementation)
 commit(OUT/'completion.json',dict(status='RIGHTS_AUDIT_FINISHED_WITH_EXPLICIT_QUARANTINES',phase='8B',completed_utc=now(),
  manifest_sha256=PIN,candidates=44,original_allocation={'train':8,'development':36},attempted=summary['attempted_candidates'],pending=summary['unattempted_candidates'],
  rights_verified_before_identity=summary['verified_licensing_candidates'],eligible_for_acquisition=summary['eligible_proposed_photos'],
  actual_body_bytes=summary['actual_body_bytes'],conservative_body_charge=summary['conservative_charge'],
  tests_passed=tests['total_tests'],photo_bodies=0,model_operations=0,old296_requests=0,activated_LVIS_labels=0,
  preserved_original_models_datasets_and_frozen_partitions=True,new_photo_acquisition_approved=False))
 resource=OUT/'resource-usage.json';commit(resource,dict(status='measuring'))
 seal=OUT/'ARTIFACT-SHA256SUMS'
 def contents():return ''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p!=seal)
 with seal.open('x') as f:f.write(contents())
 for _ in range(3):
  usage=dict(actual_new_allocated_bytes=allocated(),cap_bytes=DISK,
   accounting='st_blocks*512 for all new isolated tool and approved audit trees including directories, evidence, records, logs, closeout and seal',
   actual_downloaded_body_bytes=summary['actual_body_bytes'],conservative_body_charge=summary['conservative_charge'],body_cap_bytes=CAP,
   photograph_bodies=0,all_closeout_files_and_seal_included=True)
  resource.write_text(json.dumps(usage,indent=2,sort_keys=True)+'\n');seal.write_text(contents())
 if allocated()!=usage['actual_new_allocated_bytes'] or allocated()>DISK:raise ValueError('Final allocated bytes mismatch/limit')
 for line in seal.read_text().splitlines():
  digest,relative=line.split('  ',1)
  if sha(OUT/relative)!=digest:raise ValueError('Final artifact seal mismatch')
 for relative,digest in implementation.items():
  if sha(ROOT/relative)!=digest:raise ValueError('Final implementation seal mismatch')
 print(json.dumps(dict(status='COMPLETE',attempted=summary['attempted_candidates'],pending=summary['unattempted_candidates'],
  rights_verified=summary['verified_licensing_candidates'],proposed_photographs=summary['eligible_proposed_photos'],
  actual_response_body_bytes=summary['actual_body_bytes'],allocated_bytes=allocated(),tests=tests['total_tests'])),flush=True)

if __name__=='__main__':run()
