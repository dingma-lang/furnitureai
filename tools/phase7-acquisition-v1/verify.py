"""Independent SHA, counts, approval, source identity and historical preservation audit."""
import importlib.util
import sys
from core import *

def sums(base):
 n=0
 for line in (base/'ARTIFACT-SHA256SUMS').read_text().splitlines():
  expected,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=expected:raise ValueError('Historical seal changed '+str(path))
  n+=1
 return n

def run():
 p=proposal();net=read(OUT/'transfer-ledger.json');pre=read(OUT/'preflight.json');integrity=read(OUT/'archive-integrity.json')
 if len(net['requests'])!=2 or any(not r['complete'] or r['attempt']!=1 for r in net['requests']):raise ValueError('Unexpected acquisition attempt state')
 if sum(r['actual_body_bytes'] for r in net['requests'])!=414291789 or net['actual_body_bytes']!=414291789+pre['document_body_bytes'] or net['conservative_body_charge']!=net['actual_body_bytes']:raise ValueError('Network ledger does not reconcile')
 if net['conservative_body_charge']>CAP or any(OUT.glob('*.part')) or any(OUT.glob('*.pending')):raise ValueError('Incomplete acquisition/limit')
 for r in p['objects']:
  path=OUT/r['expected_local_filename']
  if path.stat().st_size!=r['bytes'] or sha(path)!=pre['published_sha256'][path.name]:raise ValueError('Independent checksum mismatch')
 for doc in pre['documents']:
  if 'file' in doc and sha(OUT/doc['file'])!=doc['sha256']:raise ValueError('Saved source evidence changed')
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
 spec=importlib.util.spec_from_file_location('phase7_preservation_again',ROOT/'tools/phase7-v1/preserve.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 historical=mod.verify();phase7=ROOT/'reports/phase7-data-strategy-v1';sealed=sums(phase7)
 for filename,expected in read(phase7/'implementation-sha256.json').items():
  if sha(ROOT/filename)!=expected:raise ValueError('Phase7 preparation implementation modified')
 space()
 commit(OUT/'integrity-verification.json',dict(status='PASS',archive_sha256=pre['published_sha256'],archive_crc_verified=integrity['status']=='PASS',
  archive_response_body_bytes=414291789,document_response_body_bytes=pre['document_body_bytes'],actual_total_body_bytes=net['actual_body_bytes'],conservative_transfer_charge=net['conservative_body_charge'],
  acquisition_attempts=2,retries=0,completed_resume_skips=2,allocated_bytes_at_verification=allocated(),historical=historical,
  phase7_preparation_artifact_seal_entries=sealed,photographs_downloaded=0,full_json_extraction=False,model_updates=0,model_inference_or_holdout_scoring=False))
 print('INTEGRITY AND HISTORICAL PRESERVATION PASS',flush=True)

if __name__=='__main__':run()
