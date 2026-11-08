"""Reconcile actual137-photo inventory with source targets, coverage and rights, offline."""
import collections as C
import csv
from prep_core import *
from ontology import mapping,native_scope,OI,COCO,LVIS

def summary(rows):
 result={}
 for group in CLASSES:
  selected=[r for r in rows if any(a['comparison_group']==group for a in r['resolved_targets'])]
  targets=[a for r in selected for a in r['resolved_targets'] if a['comparison_group']==group]
  source={}
  for name in sorted({r['source'] for r in selected}):
   subset=[r for r in selected if r['source']==name];aa=[a for r in subset for a in r['resolved_targets'] if a['comparison_group']==group]
   source[name]=dict(images=len(subset),source_creator_components=len({r['known_source_component'] for r in subset}),boxes=len(aa),masks=sum(a['has_mask'] for a in aa))
  def image_count(predicate):return sum(any(a['comparison_group']==group and predicate(a) for a in r['resolved_targets']) for r in selected)
  result[group]=dict(images=len(selected),source_creator_components=len({r['known_source_component'] for r in selected}),instances=len(targets),boxes=len(targets),masks=sum(a['has_mask'] for a in targets),
   by_source=source,confirmed_native_nonexhaustive_images=image_count(lambda a:a['native_scope']=='positive_nonexhaustive'),
   class_completeness_unverified_images=image_count(lambda a:a['native_scope']=='positive_completeness_unverified'),
   source_scope_counts=dict(C.Counter(a['native_scope'] for a in targets)),
   images_requiring_cross_dataset_unit_policy=image_count(lambda a:a['mapping_scope']=='native_only_unit_policy_required'),
   border_or_source_truncation_images=image_count(lambda a:a['border_or_truncation']),
   rights_verified_saved_evidence_images=len(selected),new_rights_checks=0,eligible_status='Provisional source-native positives; no independent accuracy or universal semantic verification')
 return result

def run():
 data=read(PILOT/'combined-training-readiness.json');inventory=data['source_rows'];old=read(ROOT/'data/coco/image-pilots/phase6d-v1/phase6e-training-readiness.json')['source_rows'];old_by_id={r['image_id']:r for r in old}
 legacy={r['image_id']:r for r in read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')['rows']}
 oi_current={r['image_id']:r for r in read(ROOT/'data/open-images/image-pilots/phase6a-v1/screening-v3.json')['images']}
 coco={r['image_id']:r for r in read(ROOT/'data/coco/image-pilots/phase6d-v1/screening-v2.json')['images']}
 lvis={r['image_id']:r for r in read(PILOT/'screening-v2.json')['images']}
 oi_ids={r['image_id'] for r in inventory if 'openimages' in r['source']};labels=C.defaultdict(dict)
 meta=ROOT/'data/open-images/metadata/phase5-v1'
 for part in ('validation','test'):
  with (meta/f'{part}-annotations-human-imagelabels-boxable.csv').open(newline='') as f:
   for row in csv.DictReader(f):
    if row['ImageID'] in oi_ids and row['LabelName'] in OI and float(row['Confidence']) in (0,1):labels[row['ImageID']][row['LabelName']]=int(float(row['Confidence']))
 resolved=[];all_target_ids=set()
 for item in inventory:
  r=dict(item);iid=r['image_id'];targets=[]
  if 'openimages' in r['source']:
   dataset='OpenImages';native_rows=old_by_id[iid]['usable_annotations'];source=legacy.get(iid,oi_current.get(iid));scope={'native_image_level_labels':labels[iid]};occurrences=C.Counter()
   if len(native_rows)!=len(r['targets']):raise ValueError('OI resolved target count drift')
   for a,t in zip(native_rows,r['targets']):
    native=a['source_mid'];uid0=stable_box_id(iid,a);occ=occurrences[uid0];occurrences[uid0]+=1;uid=stable_box_id(iid,a,occ)
    if OI[native]!=t['project_group']:raise ValueError('OI native mapping mismatch')
    plan=mapping(dataset,native);flags=a.get('flags',{})
    targets.append(dict(native_dataset=dataset,native_id=native,comparison_group=t['project_group'],stable_audit_annotation_id=uid,
     original_annotation=a,has_mask=False,native_scope=native_scope(dataset,native,True,scope),mapping_scope=plan['scope'],
     border_or_truncation=bool(flags.get('IsTruncated')) or any(abs(v)<1e-6 or abs(v-1)<1e-6 for v in a['box']),source_flags=flags,
     group_of=bool(flags.get('IsGroupOf')),independently_semantically_verified=False))
  else:
   source=lvis[iid] if r['source']=='phase7_LVIS' else coco[iid];dataset='LVIS' if r['source']=='phase7_LVIS' else 'COCO';scope=source.get('source_image_scope',{})
   native={a['id']:a for a in source['original_annotations']}
   for t in r['targets']:
    a=native[t['source_annotation_id']];cid=a['category_id'];plan=mapping(dataset,cid);x,y,w,h=a['bbox']
    if plan['comparison_group']!=t['project_group']:raise ValueError('Native comparison grouping mismatch')
    targets.append(dict(native_dataset=dataset,native_id=cid,comparison_group=t['project_group'],stable_audit_annotation_id=f'{dataset}:{iid}:{a["id"]}',
     original_annotation=a,has_mask=bool(a.get('segmentation')),native_scope=native_scope(dataset,cid,True,scope),mapping_scope=plan['scope'],
     border_or_truncation=min(x,y,source['width']-x-w,source['height']-y-h)<=1,group_of=bool(a.get('iscrowd',0)),independently_semantically_verified=False))
  for a in targets:
   if a['stable_audit_annotation_id'] in all_target_ids:raise ValueError('Duplicate audit annotation identity')
   all_target_ids.add(a['stable_audit_annotation_id'])
  r.update(resolved_targets=targets,source_native_scope=scope,rights_status='saved_photo_specific_grant_verified_in_preserved_prior_audit',
   original_status=source.get('status',source.get('eligibility_status',source.get('eligibility'))),
   full_seven_class_annotation_completeness_independently_verified=False,quarantine_status='not_in_prior_prospective_quarantine',annotations_activated=False)
  resolved.append(r)
 if len(resolved)!=137 or len({r['source_photo_id'] for r in resolved})!=137 or len({r['sha256'] for r in resolved})!=137:raise ValueError('Image count or exact duplicate drift')
 train=[r for r in resolved if r['split']=='train'];dev=[r for r in resolved if r['split']=='development']
 for rows,n,boxes,masks in [(train,114,361,285),(dev,23,99,76)]:
  if len(rows)!=n or sum(len(r['resolved_targets']) for r in rows)!=boxes or sum(a['has_mask'] for r in rows for a in r['resolved_targets'])!=masks:raise ValueError('Prior coverage totals fail reconciliation')
 result=dict(schema='phase8a-exact-current-coverage-v1',source_readiness_sha256=sha(PILOT/'combined-training-readiness.json'),
  split_totals={name:dict(images=len(rows),known_source_creator_components=len({r['known_source_component'] for r in rows}),boxes=sum(len(r['resolved_targets']) for r in rows),masks=sum(a['has_mask'] for r in rows for a in r['resolved_targets']),per_category=summary(rows),source_distribution=dict(C.Counter(r['source'] for r in rows))) for name,rows in [('train',train),('fresh_development',dev)]},
  rows=resolved,all_images_are_provisional_not_human_verified=True,existing33_overlay_annotations_not_counted=True,
  restored_OI_annotation_ID_policy='Native CSV lacks IDs. Stable derived image+native box/flags hashes with occurrence counters replace nonunique native-box placeholders only in this new audit, never in history.',
  full_project_annotation_completeness_verified_images=0,training_or_inference=0,new_network_requests=0)
 commit(OUT/'current-coverage.json',result)
 # Dispositions are cohort-specific, not an assertion that all rights-only metadata rows were acquired images.
 cohorts=[]
 for name,path,key in [('OI_original','data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json','eligibility_status'),('OI_phase6a','data/open-images/image-pilots/phase6a-v1/screening-v3.json','status'),('COCO_phase6d','data/coco/image-pilots/phase6d-v1/screening-v2.json','status'),('LVIS_phase7','data/lvis/image-pilots/phase7-v1/screening-v2.json','status')]:
  source_rows=read(ROOT/path)['images'];records=[]
  for r in source_rows:
   status=r.get(key,'unknown');native_anns=r.get('original_annotations',[]);groups=set()
   for a in native_anns:
    if 'category_id' in a:groups.add((COCO if name=='COCO_phase6d' else LVIS).get(a['category_id']))
    else:groups.add(a.get('label') or OI.get(a.get('LabelName',a.get('source_mid'))))
   records.append(dict(image_id=r['image_id'],status=status,original_partition=r.get('proposed_split'),comparison_categories=sorted(g for g in groups if g in CLASSES),flags=r.get('flags',[])))
  cohorts.append(dict(cohort=name,source=path,records=records,status_counts=dict(C.Counter(r['status'] for r in records)),
    noneligible_category_counts={g:sum(g in r['comparison_categories'] for r in records if r['status'] not in ('provisionally_accepted','provisional_candidate','provisional_automated_candidate','provisional_source_candidate')) for g in CLASSES}))
 commit(OUT/'quarantine-and-rights-dispositions.json',dict(cohorts=cohorts,
  original_OI_licensing_slots_not_acquired=32,rights_verified_identity_quarantined_not_acquired=9,paused_unattempted_rights_candidates=296,
  photo_quality_quarantines_are_not_combined_with_unacquired_rights_records=True,all_current137_have_saved_photo_specific_rights_evidence=True,
  independent_current_ownership_or_other_releases_verified=False,source_counts_not_physical_identity_proof=True))
 print('COVERAGE RECONCILED114/23 photographs361/99 boxes285/76 masks',flush=True)

if __name__=='__main__':run()
