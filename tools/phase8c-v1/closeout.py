"""Seal completed acquisition/screening/readiness only; never launch requests or models."""
import subprocess
import sys
from pilot_core import *

def run():
 if read(OUT/'test-results.json')['status']!='PASS' or read(OUT/'integrity-verification.json')['status']!='PASS':raise ValueError('Final checks not passed')
 subprocess.run([sys.executable,'-B',str(HERE/'report.py')],cwd=ROOT,check=True)
 tests=read(OUT/'test-results.json');net=read(OUT/'network-ledger.json');ready=read(OUT/'combined-training-readiness.json')
 implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
 commit(OUT/'implementation-sha256.json',implementation)
 commit(OUT/'completion.json',dict(status='PHASE8C_ACQUISITION_AND_CONSERVATIVE_SCREENING_COMPLETE',completed_utc=now(),
  manifest_sha256=PIN,downloaded=27,original_allocation={'train':7,'development':20},body_attempts=27,
  actual_response_body_bytes=net['actual_body_bytes'],conservative_body_charge=net['conservative_charge'],
  retained_provisional_images=19,new_train_images=6,new_fresh_development_images=13,new_native_boxes_masks=39,
  quarantined_images=8,unresolved_native_annotations=60,original_native_annotations_preserved=99,
  combined_split_totals={p:{k:v for k,v in s.items() if k!='per_category'} for p,s in ready['split_totals'].items()},
  tests_passed=tests['total_tests'],exact_duplicate_pairs=0,suspected_perceptual_pairs=0,model_operations=0,live_rights_requests=0,
  previously_inactive33_still_inactive=True,historical_preservation='PASS',next_actions_approved=False))
 resource=OUT/'resource-usage.json';commit(resource,dict(status='measuring'))
 artifact=OUT/'ARTIFACT-SHA256SUMS'
 def seal():return ''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p!=artifact)
 with artifact.open('x') as f:f.write(seal())
 for _ in range(3):
  usage=dict(actual_new_allocated_bytes=allocated()+allocated(HERE),cap_bytes=DISK,
   actual_response_body_bytes=net['actual_body_bytes'],conservative_body_charge=net['conservative_charge'],transfer_cap_bytes=CAP,
   all_new_tools_photos_partials_derivatives_reports_and_seal_included=True,accounting='st_blocks*512 over both isolated new trees and directories',
   partial_files_remaining=len(list(OUT.rglob('*.part'))),downloaded_photographs=27,model_weights_or_metadata_archives_downloaded=0,
   rendered_synthetic_views=0)
  resource.write_text(json.dumps(usage,indent=2,sort_keys=True)+'\n');artifact.write_text(seal())
 if allocated()+allocated(HERE)!=usage['actual_new_allocated_bytes'] or usage['actual_new_allocated_bytes']>DISK:raise ValueError('Final storage accounting mismatch/cap')
 for line in artifact.read_text().splitlines():
  digest,relative=line.split('  ',1)
  if sha(OUT/relative)!=digest:raise ValueError('Final artifact seal mismatch')
 for relative,digest in implementation.items():
  if sha(ROOT/relative)!=digest:raise ValueError('Implementation seal mismatch')
 print(json.dumps(dict(status='COMPLETE',downloaded=27,actual_response_body_bytes=net['actual_body_bytes'],allocated_bytes=usage['actual_new_allocated_bytes'],eligible=19,quarantined=8,new_native_targets=39,tests=tests['total_tests'])),flush=True)

if __name__=='__main__':run()
