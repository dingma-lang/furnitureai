"""Rights and identity decisions remain separate; quarantine all known cross-split groups."""
import collections as C
import hashlib
import importlib.util
import sys
from audit_core import *

def category_counts(rows):
 result={}
 for cid,name in CORE.items():
  present=[r for r in rows if any(a['category_id']==cid for a in r['original_lvis_furniture_annotations'])]
  anns=[a for r in present for a in r['original_lvis_furniture_annotations'] if a['category_id']==cid]
  result[name]=dict(native_id=cid,images=len(present),instances=len(anns),boxes=len(anns),masks=sum(bool(a.get('segmentation')) for a in anns))
 return result

def identity_decisions(rows,records,uf,protected_anchors):
 # Include restricted/unverified rows with an independently source-bound creator too: excluded bridges must not split families.
 for r in records:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
 protected={uf.find(p) for p in protected_anchors};byid={r['coco_image_id']:r for r in rows};groups=C.defaultdict(list)
 for r in records:groups[uf.find(r['source_photo_id'])].append(r)
 urls=C.defaultdict(list);head_signatures=C.defaultdict(list)
 for r in records:
  row=byid[r['coco_image_id']];urls[row['original_source_asset_url']].append(r['coco_image_id'])
  obj=r.get('image_object',{})
  if r['image_object_status']=='available':head_signatures[obj['etag'],obj['bytes']].append(r['coco_image_id'])
 duplicate_urls=[v for v in urls.values() if len(v)>1];header_suspects=[v for v in head_signatures.values() if len(v)>1]
 duplicated={i for v in duplicate_urls+header_suspects for i in v};decisions=[]
 for group,members in groups.items():
  parts={r['provisional_partition'] for r in members};flags=[]
  if group in protected:flags.append('historical_source_or_creator_group')
  if len(parts)>1:flags.append('creator_or_source_group_crosses_train_development')
  for r in members:
   own=list(flags)
   if r['coco_image_id'] in duplicated:own.append('duplicate_source_URL_or_equal_image_HEAD_signature_unresolved')
   eligible=r['licensing_status']==STATUSES[0] and r['image_object_status']=='available' and not own
   decisions.append(dict(coco_image_id=r['coco_image_id'],source_photo_id=r['source_photo_id'],provisional_partition=r['provisional_partition'],
    known_source_group='phase8b-rights-group:'+hashlib.sha256(group.encode()).hexdigest()[:24],source_group_members=len(members),
    creator_aliases=r.get('creator_aliases',[]),identity_flags=own,licensing_status=r['licensing_status'],image_object_status=r['image_object_status'],
    eligible_for_acquisition_proposal=eligible,quarantined=not eligible,
    physical_scene_product_independence_verified=False,original_partition_changed=False))
 return decisions,duplicate_urls,header_suspects

def run():
 m=manifest();audit=read(OUT/'rights-audit.json');records=audit['records'];rows=m['images'];byid={r['coco_image_id']:r for r in rows}
 if len(records)!=44 or {r['coco_image_id'] for r in records}!=set(byid):raise ValueError('Audit coverage IDs differ from approved scope')
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase7_rights_protected_graph',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 uf,protected,history=history_graph();decisions,duplicate_urls,header_suspects=identity_decisions(rows,records,uf,protected)
 decision_by_id={r['coco_image_id']:r for r in decisions};groups=C.defaultdict(list)
 for r in decisions:groups[r['known_source_group']].append(r)
 repeated_creators=[dict(group=g,images=[r['coco_image_id'] for r in members],partitions=sorted({r['provisional_partition'] for r in members}),
   conflicts=sorted({f for r in members for f in r['identity_flags']})) for g,members in groups.items() if len(members)>1]
 commit(OUT/'identity-audit.json',dict(records=decisions,known_historical_graph=history,repeated_source_or_creator_groups=repeated_creators,
  duplicate_original_source_URL_groups=duplicate_urls,equal_HEAD_etag_size_suspicions=header_suspects,
  partition_reassignments=0,physical_identities_merged=0,perceptual_duplicate_or_visual_screening=False,
  limitation='Creator/source groups are conservative isolation components, not assertions that unrelated objects are one physical product. Unknown aliases and cross-creator scene reuse remain unresolved.'))
 selected=[]
 for result in records:
  decision=decision_by_id[result['coco_image_id']]
  if not decision['eligible_for_acquisition_proposal']:continue
  row=dict(byid[result['coco_image_id']]);record_path=OUT/'records'/f"{result['coco_image_id']:012d}.json"
  row.update(verified_photo_rights=dict(creator=result['creator'],creator_url=result['creator_url'],source_url=result['source_url'],license_url=result['license_url'],
    title=result['title'],attribution=result['attribution'],other_rights=result['other_rights'],evidence_url=result['evidence_url'],evidence_checked_utc=result['evidence_checked_utc'],
    evidence_file=str((OUT/result['evidence_file']).relative_to(ROOT)),evidence_sha256=result['evidence_sha256'],rights_record=str(record_path.relative_to(ROOT)),rights_record_sha256=sha(record_path)),
    identity_decision=decision,remote_object=result['image_object'],photo_acquisition_approved=False,training_approved=False,
    metadata_commercial_rights_status_preserved=row['commercial_rights_status'],current_rights_status=STATUSES[0],human_reviewed=False)
  selected.append(row)
 verified_rows=[byid[r['coco_image_id']] for r in records if r['licensing_status']==STATUSES[0]]
 category_value=dict(rights_verified_before_identity_exclusions=category_counts(verified_rows),eligible_for_acquisition_proposal=category_counts(selected),
  scope_policy='Native classes and subcategories retained. Negative/nonexhaustive flags preserved; missing labels never imply absence.')
 commit(OUT/'annotation-value.json',category_value)
 expected=sum(r['remote_object']['bytes'] for r in selected)
 proposal=dict(schema='phase8b-LVIS-individually-rights-verified-photo-acquisition-v1',images=selected,image_count=len(selected),
  manifest_sha256=PIN,source_manifest=str(MANIFEST.relative_to(ROOT)),original_provisional_allocation_preserved=True,
  proposed_allocation=dict(C.Counter(r['proposed_partition'] for r in selected)),category_distribution=category_counts(selected),
  expected_photo_body_bytes=expected,expected_max_two_attempt_body_bytes=2*expected,
  proposed_cumulative_body_cap=max(100000000,2*expected+1000000) if selected else 0,
  proposed_per_photo_max=3145728,proposed_max_attempts_per_photo=2,proposed_new_allocated_storage_cap=536870912 if expected<200000000 else 1073741824,
  estimated_allocated_photo_mask_contact_bytes_range=[int(expected*1.3)+16*2**20,int(expected*2)+64*2**20],
  photograph_acquisition_approved=False,training_approved=False,
  required_post_download_checks=['decodability, dimensions and SHA256','exact and perceptual duplicates across known nonfrozen material','source/creator and train-development isolation','indoor scene suitability, furniture visibility and native annotation/mask registration','separate semantic unit policy for Cabinet/Desk/Table and explicit partial-label losses'],
  no_visual_or_perceptual_screening_claimed=True)
 commit(OUT/'proposed-image-acquisition-v1.json',proposal)
 unattempted=[r for r in records if not r['attempted']]
 commit(OUT/'unattempted-checkpoint-v1.json',dict(records=[dict(coco_image_id=r['coco_image_id'],source_photo_id=r['source_photo_id'],provisional_partition=r['provisional_partition'],reason=r.get('reason')) for r in unattempted],
  manifest_sha256=PIN,count=len(unattempted),provisional_allocation=dict(C.Counter(r['provisional_partition'] for r in unattempted)),
  retry_or_resume_authorized=False,new_separate_approval_required=True,source_provider_circuits=audit['circuits'],
  policy='Only never-attempted candidates may be considered for a separately approved future phase. No current circuit is automatically cleared and attempted candidates are never retried.'))
 if selected:
  next_request=dict(scope='Photograph acquisition and screening only for exact independently rights-verified eligible IDs',manifest='proposed-image-acquisition-v1.json',manifest_sha256=sha(OUT/'proposed-image-acquisition-v1.json'),
   count=len(selected),provisional_allocation=proposal['proposed_allocation'],expected_transfer=expected,max_cumulative_body_bytes=proposal['proposed_cumulative_body_cap'],
   max_attempts_per_photo=2,max_photo_bytes=3145728,max_new_allocated_storage=proposal['proposed_new_allocated_storage_cap'],destination='data/lvis/image-pilots/phase8b-v1/',approved=False,
   no_training_inference_or_rendering=True,unattempted_rights_audit_resume_approved=False)
 else:
  next_request=dict(scope='No photograph acquisition is eligible. A future rights-only approval may consider the exact never-attempted subset after provider access is available.',
   unattempted_manifest='unattempted-checkpoint-v1.json',unattempted_manifest_sha256=sha(OUT/'unattempted-checkpoint-v1.json'),count=len(unattempted),
   provisional_allocation=dict(C.Counter(r['provisional_partition'] for r in unattempted)),max_cumulative_body_bytes=CAP,max_new_allocated_storage=DISK,
   max_attempts_per_candidate=1,photo_bodies=False,approved=False,provider_access_must_be_restored=True)
 commit(OUT/'next-approval-request.json',next_request)
 stats=dict(total_candidates=44,attempted_candidates=sum(r['attempted'] for r in records),unattempted_candidates=len(unattempted),
  licensing_status_counts=dict(C.Counter(r['licensing_status'] for r in records)),verified_licensing_candidates=len(verified_rows),eligible_proposed_photos=len(selected),
  identity_quarantined_verified_candidates=sum(r['licensing_status']==STATUSES[0] and bool(r['identity_flags']) for r in decisions),
  source_unavailable_or_unverified_image_objects=sum(r['licensing_status']==STATUSES[0] and r['image_object_status']!='available' for r in records),
  proposed_allocation=proposal['proposed_allocation'],expected_photo_body_bytes=expected,
  annotation_value=category_value,source_GET_attempts=sum(r['method']=='GET' for r in read(OUT/'network-ledger.json')['requests']),
  source_successful_document_GETs=sum(r['method']=='GET' and r.get('status')==200 and r.get('complete') for r in read(OUT/'network-ledger.json')['requests']),
  actual_body_bytes=audit['actual_body_bytes'],conservative_charge=audit['conservative_charge'],provider_circuits=audit['circuits'],photo_bodies_downloaded=0)
 commit(OUT/'summary.json',stats);space();print('FINALIZED',stats['verified_licensing_candidates'],'rights verified;',len(selected),'eligible;',len(unattempted),'unattempted',flush=True)

if __name__=='__main__':run()
