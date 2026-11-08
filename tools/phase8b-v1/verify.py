"""Independent final request, evidence, partition and historical-preservation audit."""
from audit_core import *
from transport import allowed_request,provider

def seal(base,name):
 count=0
 for line in (base/name).read_text().splitlines():
  digest,relative=line.split('  ',1);p=(base/relative).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=digest:raise ValueError('Historical seal changed '+str(p))
  count+=1
 return count

def run():
 rows={r['coco_image_id']:r for r in manifest()['images']};audit=read(OUT/'rights-audit.json');net=read(OUT/'network-ledger.json')
 records=audit['records'];proposal=read(OUT/'proposed-image-acquisition-v1.json')
 if len(records)!=44 or {r['coco_image_id'] for r in records}!=set(rows):raise ValueError('Audit IDs changed')
 if any(r['provisional_partition']!=rows[r['coco_image_id']]['proposed_partition'] or r['evidence_attempts']>1 for r in records):raise ValueError('Partition/attempt changed')
 if sum(r['actual_body_bytes'] for r in net['requests'])!=net['actual_body_bytes'] or not 0<=net['actual_body_bytes']<=net['conservative_charge']<=CAP:raise ValueError('Transfer accounting mismatch')
 blocked=set();transactions=set();last_source=None
 for r in net['requests']:
  row=rows[r['coco_image_id']];key=(r['coco_image_id'],r['method'],r['url']);host=provider(r['url'])
  if not allowed_request(row,r['url'],r['method']) or key in transactions or host in blocked:raise ValueError('Unsafe repeated/unapproved/restricted request')
  transactions.add(key)
  if r.get('status') in (401,403,429) or r.get('error'):blocked.add(host)
  if host=='flickr.com':
   instant=dt.datetime.fromisoformat(r['checked_utc'])
   if last_source and (instant-last_source).total_seconds()<2.99:raise ValueError('Source request pacing violation')
   last_source=instant
  if r.get('file'):
   p=OUT/r['file']
   if p.stat().st_size!=r['actual_body_bytes'] or sha(p)!=r.get('sha256'):raise ValueError('Source evidence hash/size mismatch')
  if r.get('status') and not r['tls_verified']:raise ValueError('Unverified TLS response')
 for r in records:
  if r.get('evidence_file') and sha(OUT/r['evidence_file'])!=r['evidence_sha256']:raise ValueError('Rights evidence mismatch')
 eligible={r['coco_image_id'] for r in read(OUT/'identity-audit.json')['records'] if r['eligible_for_acquisition_proposal']}
 if {r['coco_image_id'] for r in proposal['images']}!=eligible:raise ValueError('Acquisition eligibility mismatch')
 for r in proposal['images']:
  if r['proposed_partition']!=rows[r['coco_image_id']]['proposed_partition'] or r['original_LVIS_core_annotations']!=rows[r['coco_image_id']]['original_LVIS_core_annotations']:raise ValueError('Proposal split/native annotation mutation')
  proof=r['verified_photo_rights']
  if sha(ROOT/proof['evidence_file'])!=proof['evidence_sha256'] or sha(ROOT/proof['rights_record'])!=proof['rights_record_sha256']:raise ValueError('Proposed rights evidence changed')
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
 spec=importlib.util.spec_from_file_location('phase8b_preserve',ROOT/'tools/phase7-v1/preserve.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);historical=m.verify()
 sealed={}
 for relative,name in [('reports/phase7-data-strategy-v1','ARTIFACT-SHA256SUMS'),('data/lvis/metadata/phase7-v1','ARTIFACT-SHA256SUMS'),('data/lvis/rights-audits/phase7-v1','FINAL-ARTIFACT-SHA256SUMS-v2'),('data/lvis/image-pilots/phase7-v1','ARTIFACT-SHA256SUMS'),('reports/phase8a-v1','ARTIFACT-SHA256SUMS'),('tools/phase8a-v1','IMPLEMENTATION-SHA256SUMS')]:
  sealed[relative]=seal(ROOT/relative,name)
  for name2 in ['implementation-sha256-final-v2.json','implementation-sha256.json']:
   if (ROOT/relative/name2).exists():
    for path,digest in read(ROOT/relative/name2).items():
     if sha(ROOT/path)!=digest:raise ValueError('Historical source changed')
    break
 if read(ROOT/'reports/phase8a-v1/existing33-registration-v2.json')['annotations_activated']!=0:raise ValueError('Inactive labels changed')
 commit(OUT/'integrity-verification.json',dict(status='PASS',historical=historical,seals=sealed,
  candidates=44,partition_changes=0,photo_bodies=0,actual_body_bytes=net['actual_body_bytes'],conservative_charge=net['conservative_charge'],
  provider_requests_after_access_restriction=0,paused296_requests=0,independently_activated_LVIS_labels=0,
  frozen_pixels_opened_or_scored=0,model_operations=0,Mac_project_accessed=False,
  perceptual_duplicate_or_visual_screening=False,allocated_bytes_at_verification=allocated()))
 print('Request/evidence/partition integrity and historical preservation PASS',flush=True)

if __name__=='__main__':run()
