"""New derived combined inventory; no historical membership, source geometry or labels edited."""
import collections as C
import importlib.util
import sys
from pilot_core import *

def run():
 sys.path.insert(0,str(ROOT/'tools/phase8a-v1'))
 spec=importlib.util.spec_from_file_location('phase8c_coverage',ROOT/'tools/phase8a-v1/coverage.py');cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
 from ontology import mapping,native_scope
 sys.path.insert(0,str(ROOT/'tools/phase8b-v1'))
 spec=importlib.util.spec_from_file_location('phase8c_combined_graph',ROOT/'tools/phase8b-v1/audit_core.py');graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
 uf,_,_=graph.history_graph()
 for record in read(RIGHTS/'rights-audit.json')['records']:
  for alias in record.get('creator_aliases',[]):uf.join(record['source_photo_id'],alias)
 rows=read(ROOT/'reports/phase8a-v1/current-coverage.json')['rows'];new=read(OUT/'eligible-candidates-v1.json')['images'];source={r['coco_image_id']:r for r in manifest()['images']}
 for r in new:
  original=source[r['coco_image_id']];targets=[]
  for a in r['original_annotations']:
   if a['id'] not in r['retained_annotation_ids']:continue
   cid=a['category_id'];plan=mapping('LVIS',cid);sc=native_scope('LVIS',cid,True,r['source_image_scope']);derived=r['derived_training_scope'][str(cid)]
   x,y,w,h=a['bbox']
   targets.append(dict(native_dataset='LVIS',native_id=cid,comparison_group=CLASSES[cid],stable_audit_annotation_id=f"LVIS:{r['image_id']}:{a['id']}",
    original_annotation=a,has_mask=bool(a.get('segmentation')),native_scope='positive_nonexhaustive' if derived['derived_positive_nonexhaustive'] else sc,
    native_scope_before_screening=sc,mapping_scope=plan['scope'],border_or_truncation=min(x,y,r['width']-x-w,r['height']-y-h)<=1,
    group_of=False,independently_semantically_verified=False,derived_training_scope=derived))
  rows.append(dict(source='phase8c_LVIS',image_id=r['image_id'],source_photo_id=r['source_identity'],split=r['proposed_split'],sha256=r['sha256'],decoded_pixel_sha256=r['decoded_pixel_sha256'],
   source_file=str((OUT/r['file']).relative_to(ROOT)),resolved_targets=targets,source_native_scope=r['source_image_scope'],
   unresolved_native_annotation_ids=r['unresolved_annotation_ids'],screening='assistant_only_provisional_native_positive',
   photograph_licensing='Saved photo-specific CC BY2.0 grant replayed offline; no ownership/release warranty',
   rights_status='saved_photo_specific_grant_verified_in_preserved_prior_audit',shared_project_unit_verification=False,
   full_seven_class_annotation_completeness_independently_verified=False,annotations_activated=False,
   source_manifest_sha256=PIN,rights_evidence_sha256=original['verified_photo_rights']['evidence_sha256']))
 groups=C.defaultdict(set);reserved=set()
 for r in read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']:
  if r['proposed_split']!='train':reserved.add(uf.find(r['source_identity']))
 for r in rows:r['known_source_component']=uf.find(r['source_photo_id']);groups[r['known_source_component']].add(r['split'])
 keep=[];excluded=[];seen_photos={};seen_hashes={}
 for r in rows:
  reason=[];group=r['known_source_component']
  if len(groups[group])>1:reason.append('known_source_creator_crosses_training_development')
  if group in reserved:reason.append('known_source_creator_touches_historical_reserved_development_or_frozen')
  if r['source_photo_id'] in seen_photos or r['sha256'] in seen_hashes:reason.append('duplicate_photo_requires_overlay_reconciliation')
  seen_photos[r['source_photo_id']]=r['image_id'];seen_hashes[r['sha256']]=r['image_id']
  if reason:excluded.append(dict(image_id=r['image_id'],reasons=reason))
  else:keep.append(r)
 splits={}
 for part in ('train','development'):
  subset=[r for r in keep if r['split']==part];targets=[a for r in subset for a in r['resolved_targets']]
  splits[part]=dict(images=len(subset),known_source_creator_components=len({r['known_source_component'] for r in subset}),boxes=len(targets),masks=sum(a['has_mask'] for a in targets),
   per_category=cov.summary(subset),source_distribution=dict(C.Counter(r['source'] for r in subset)))
 commit(OUT/'combined-training-readiness.json',dict(schema='phase8c-derived-real-source-readiness-v1',training_approved=False,source_rows=keep,split_totals=splits,
  source_group_or_duplicate_exclusions=excluded,old_source_coverage_sha256=sha(ROOT/'reports/phase8a-v1/current-coverage.json'),
  new_screening_sha256=sha(OUT/'screening-v2.json'),original_source_geometry_and_partitions_modified=False,
  existing33_LVIS_annotations_inactive=True,COCO_LVIS_photo_overlays_not_double_counted=True,
  historical_frozen_membership_and_scoring_unchanged=True,physical_scene_or_product_independence_verified=False,
  completeness='Source-native scopes retained; derived subsets downgrade positive completeness when same-native targets are excluded. Unknown and unresolved never background.',
  no_full_seven_class_completeness_claim=True,shared_project_mapping_independently_verified=False,
  baseline_photographs_not_newly_visually_reaudited=True,ready_for_broad_seven_class_training=False))
 print('COMBINED', {p:(s['images'],s['known_source_creator_components'],s['boxes'],s['masks']) for p,s in splits.items()},'excluded',len(excluded),flush=True)

if __name__=='__main__':run()
