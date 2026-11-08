"""Final attribution ledger and exact eligible-subset photograph proposal."""
import collections as C
import csv
from common import *

def run():
 m=manifest();records=read(OUT/'rights-audit-v3.json')['records'];identity=read(OUT/'identity-audit.json');decisions={r['image_id']:r for r in identity['records']};byid={r['image_id']:r for r in records}
 accepted=[];ledger=[]
 for row in m['images']:
  r=byid[row['image_id']];d=decisions[row['image_id']];source_file=OUT/'records-v3'/(row['image_id'].replace(':','-')+'.json')
  entry=dict(**r,rights_record=str(source_file.relative_to(ROOT)),rights_record_sha256=sha(source_file),identity_decision=d,
   # An out-of-profile licence can also contradict the old metadata; retain
   # both facts without silently changing the primary status category.
   licence_differs_from_metadata=bool(r.get('license_url') and r.get('license_url')!=license_kind(row['metadata_license_url'])[1]))
  ledger.append(entry)
  if not d['eligible_for_image_acquisition_proposal']:continue
  get_records=[e for e in r['evidence_requests'] if e['method']=='GET' and e.get('file') and e['status']==200]
  if not get_records:raise ValueError('No photo-specific source document')
  evidence=get_records[-1]
  if sha(OUT/evidence['file'])!=evidence['sha256']:raise ValueError('Evidence bytes changed')
  accepted.append(dict(row,license_verified=True,current_license_url=r['license_url'],creator=r['creator'],creator_url=r['creator_url'],source_title=r['title'],verified_source_page=r['source_url'],rights_checked_utc=r['checked_utc'],
   rights_record=str(source_file.relative_to(ROOT)),rights_record_sha256=sha(source_file),source_document=str((OUT/evidence['file']).relative_to(ROOT)),source_document_sha256=evidence['sha256'],attribution=r['attribution'],
   metadata_identity_group=row['identity_group'],identity_group=d['known_source_group'],known_source_group=d['known_source_group'],creator_group_verified=True,rights_verification_required_before_photo_approval=False,independent_physical_identity_verified=False,remote_object=r['image_object'],image_sha256=None,
   rights_status='verified_commercially_compatible',photograph_acquisition_approved=False,scene_verified=False,visual_quality_verified=False,perceptual_duplicates_checked=False))
 expected=sum(r['remote_object']['bytes'] for r in accepted);groups={r['known_source_group'] for r in accepted};counts={}
 for part in ('train','development'):
  rs=[r for r in accepted if r['proposed_split']==part]
  counts[part]=dict(images=len(rs),known_source_groups=len({r['known_source_group'] for r in rs}),categories={label:dict(images=sum(any(a['category_id']==mid for a in r['original_furniture_annotations']) for r in rs),boxes=sum(a['category_id']==mid for r in rs for a in r['original_furniture_annotations']),polygon_annotations=sum(a['category_id']==mid and isinstance(a['segmentation'],list) for r in rs for a in r['original_furniture_annotations']),crowd_regions=sum(a['category_id']==mid and bool(a['iscrowd']) for r in rs for a in r['original_furniture_annotations'])) for mid,label in [(62,'chair'),(63,'sofa'),(65,'bed'),(67,'dining_table')]})
 plan=dict(schema='phase6c-rights-verified-image-acquisition-proposal-v1',source_manifest_sha256=PIN,source_candidate_count=400,original_allocation=dict(train=320,development=80),original_slots_preserved=True,images=accepted,image_count=len(accepted),known_source_group_count=len(groups),summary=counts,
  rights_ledger='license-attribution-ledger.json',identity_audit='identity-audit.json',expected_photo_response_bytes=expected,max_two_complete_attempt_photo_bytes=2*expected,
  proposed_cumulative_transfer_cap=100000000,proposed_allocated_storage_cap=536870912,per_photo_max=PHOTO_MAX,max_attempts_per_photo=2,destination=str(ROOT/'data/coco/image-pilots/phase6d-v1'),
  photograph_acquisition_approved=False,training_approved=False,model_inference_approved=False,rights_freshness_days_at_acquisition=7,
  excluded_source_ids=[r['image_id'] for r in m['images'] if r['image_id'] not in {x['image_id'] for x in accepted}],
  scope='Only the exact rights-verified, identity-isolated72-source subset; no replacement IDs or partition moves. Preserve all original COCO annotations and masks. Photograph GET requires a new explicit approval.',
  remaining_checks=['Recheck saved rights evidence freshness and hashes before acquisition','Official S3 HEAD size/ETag must remain pinned','Actual photograph SHA256, integrity, decoding and dimensions','Exact/pixel/perceptual duplicate and cross-dataset overlap checks','Real indoor scene suitability, blur and partial/mixed-object boxes','Native dining table versus broad table scope; absent Desk/Cabinet/Lamp remain unknown; preserve crowd masks separately'])
 if expected*2+read(OUT/'network-ledger.json')['conservative_charge']>plan['proposed_cumulative_transfer_cap']:raise ValueError('Future resource proposal insufficient')
 commit(OUT/'license-attribution-ledger.json',dict(records=ledger,primary_status_counts=dict(C.Counter(r['licensing_status'] for r in records)),known_metadata_licence_conflict_records=sum(r['licence_differs_from_metadata'] for r in ledger),photo_bodies_downloaded=0,head_is_not_rights_verification=True))
 commit(OUT/'proposed-image-acquisition-v1.json',plan)
 with (OUT/'attribution-and-decisions.csv').open('x',newline='') as f:
  fields=['image_id','source_identity','proposed_split','licensing_status','creator','title','source_url','license_url','metadata_license_url','checked_utc','rights_record_sha256','quarantined','identity_flags']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in ledger:
   d=r['identity_decision'];w.writerow(dict(**{k:r.get(k,'') for k in fields if k not in ('quarantined','identity_flags')},quarantined=d['quarantined'],identity_flags=';'.join(d['identity_flags'])))
 commit(OUT/'image-acquisition-approval-request.json',dict(approval_granted=False,manifest='proposed-image-acquisition-v1.json',manifest_sha256=sha(OUT/'proposed-image-acquisition-v1.json'),photographs=len(accepted),partitions={k:v['images'] for k,v in counts.items()},expected_photo_bytes=expected,transfer_cap=100000000,storage_cap=536870912,per_photo_max=PHOTO_MAX,max_attempts=2,destination=plan['destination'],scope='Acquisition and screening only; no scoring, training, substitutions or partition changes.'))
 print('FINAL PROPOSAL',len(accepted),'images',len(groups),'known source groups',expected,'expected photo bytes',flush=True)

if __name__=='__main__':run()
