"""Prospective real-source inventory, exact-photo dedup and known-creator isolation."""
import collections as C
import importlib.util
import sys
from pilot_core import *

def summarize(rows):
 result={}
 for name in ('sofa','chair','table','bed','desk','cabinet','lamp'):
  matches=[r for r in rows if any(a['project_group']==name for a in r['targets'])];ann=[a for r in matches for a in r['targets'] if a['project_group']==name]
  result[name]=dict(images=len(matches),instances=len(ann),boxes=len(ann),instance_masks=sum(a['has_mask'] for a in ann),
   native_source_strata=dict(C.Counter(a['native_stratum'] for a in ann)))
 return result

def run():
 baseline=read(ROOT/'data/coco/image-pilots/phase6d-v1/phase6e-training-readiness.json');old=baseline['source_rows'];new=read(OUT/'eligible-candidates-v1.json')['images'];source=manifest();newsource={r['coco_image_id']:r for r in source['images']}
 oi_meta={}
 for path in [ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json',ROOT/'data/open-images/preparation/phase6a-v1/acquisition-manifest-v2.json']:
  oi_meta.update({r['image_id']:r for r in read(path)['images']})
 coco_meta={r['coco_image_id']:r for r in read(ROOT/'data/coco/image-pilots/phase6d-v1/approved-manifest.json')['images']}
 signatures={}
 for path in [ROOT/'data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json',ROOT/'data/open-images/image-pilots/phase6a-v1/screening-v3.json',ROOT/'data/coco/image-pilots/phase6d-v1/screening-v2.json']:
  for r in read(path)['images']:signatures[r['image_id']]=r
 inventory=[]
 for r in old:
  targets=[]
  if 'coco' in r['source']:
   iid=int(r['image_id'].split(':')[1]);meta=coco_meta[iid];photo=meta['source_identity']
   for a in r['usable_annotations']:
    original=a['original_annotation'];label='table' if a['label']=='dining_table' else a['label']
    targets.append(dict(project_group=label,native_stratum=f"COCO:{original['category_id']}:{a['label']}",source_annotation_id=original['id'],has_mask=bool(original.get('segmentation'))))
  else:
   meta=oi_meta[r['image_id']];photo=meta['source_identity']
   for a in r['usable_annotations']:targets.append(dict(project_group=a['label'],native_stratum='OpenImages:'+a['source_mid'],source_annotation_id=a.get('source_annotation_id',a.get('annotation_id','native-box')),has_mask=False))
  sig=signatures[r['image_id']]
  inventory.append(dict(source=r['source'],image_id=r['image_id'],source_photo_id=photo,split=r['split'],sha256=sig['sha256'],decoded_pixel_sha256=sig.get('decoded_pixel_sha256'),targets=targets,photograph_licensing='Previously verified photo-specific source evidence',original_source_group=r['original_source_group']))
 for r in new:
  meta=newsource[r['coco_image_id']];targets=[]
  for a in r['original_annotations']:
   if a['category_id'] in CLASSES and a['id'] in r['valid_core_annotation_ids']:targets.append(dict(project_group=CLASSES[a['category_id']],native_stratum=f"LVIS:{a['category_id']}",source_annotation_id=a['id'],has_mask=True))
  inventory.append(dict(source='phase7_LVIS',image_id=r['image_id'],source_photo_id=r['source_identity'],split=r['proposed_split'],sha256=r['sha256'],decoded_pixel_sha256=r['decoded_pixel_sha256'],targets=targets,photograph_licensing='Saved photo-specific CC BY2.0 grant reverified offline',original_source_group=r['identity_group']))
 # Equal creator groups do not collapse distinct photographs. Exact photo/hash matches do.
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase7_combined_source_graph',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 uf,protected,history,prior=mod.identity_graph()
 for r in read(RIGHTS/'rights-audit.json')['records']:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
 reserved=C.defaultdict(list)
 for r in read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']:
  if r['proposed_split']!='train':reserved[uf.find(r['source_identity'])].append(dict(image_id=r['image_id'],split=r['proposed_split']))
 group_parts=C.defaultdict(set)
 for r in inventory:r['known_source_component']=uf.find(r['source_photo_id']);group_parts[r['known_source_component']].add(r['split'])
 prospective=[];excluded=[];seen_photo={};seen_hash={};duplicate_refs=[]
 for r in inventory:
  root=r['known_source_component'];flags=[]
  if len(group_parts[root])>1:flags.append('known_creator_source_group_crosses_train_development')
  if root in reserved:flags.append('source_group_touches_reserved_historical_development_or_frozen')
  previous=seen_photo.get(r['source_photo_id']) or seen_hash.get(r['sha256'])
  if previous:
   duplicate_refs.append(dict(a=r['image_id'],b=previous['image_id'],reason='Exact source-photo ID or encoded image SHA'))
   flags.append('duplicate_photo_requires_source_overlay_reconciliation')
  seen_photo.setdefault(r['source_photo_id'],r);seen_hash.setdefault(r['sha256'],r)
  if flags:excluded.append(dict(row=r,reasons=flags))
  else:prospective.append(r)
 train=[r for r in prospective if r['split']=='train'];dev=[r for r in prospective if r['split']=='development']
 result=dict(schema='phase7-combined-real-source-readiness-v1',training_authorized=False,source_annotation_geometry_modified=False,partitions_modified=False,
  old_nominal_training_photos=81,old_positive_scope_training_photos=74,old_fresh_development_photos=15,
  old_nominal_vs_positive_scope_caveat='Seven legacy Open Images training photographs have no retained positive targets after established Desk/Cabinetry and descendant-scope masks. Original datasets remain unchanged.',
  new_LVIS_provisional_train=sum(r['proposed_split']=='train' for r in new),new_LVIS_provisional_development=sum(r['proposed_split']=='development' for r in new),
  prospective_unique_training_photographs=len(train),prospective_unique_fresh_development_photographs=len(dev),
  known_train_source_components=len({r['known_source_component'] for r in train}),known_fresh_development_source_components=len({r['known_source_component'] for r in dev}),
  physical_scene_or_product_independence_verified=False,exact_duplicate_photo_references=duplicate_refs,prospective_source_group_exclusions=excluded,
  per_category_training=summarize(train),per_category_fresh_development=summarize(dev),
  train_source_distribution=dict(C.Counter(r['source'] for r in train)),fresh_development_source_distribution=dict(C.Counter(r['source'] for r in dev)),
  total_training_instances=sum(len(r['targets']) for r in train),total_training_boxes=sum(len(r['targets']) for r in train),total_training_instance_masks=sum(a['has_mask'] for r in train for a in r['targets']),
  total_fresh_development_instances=sum(len(r['targets']) for r in dev),total_fresh_development_boxes=sum(len(r['targets']) for r in dev),total_fresh_development_instance_masks=sum(a['has_mask'] for r in dev for a in r['targets']),
  reserved_historical_development_photographs=18,reserved_historical_development_not_in_fresh_selection_counts=True,original_frozen_eligible_photos_unused=18,all_original_frozen_slots_preserved=60,
  existing_COCO_LVIS_overlay_annotations_inactive=33,existing_overlay_photos_not_added_as_new_photos=10,
  source_rows=prospective,rights_verified_means='Source-specific saved CC BY evidence, not ownership warranty, releases or validated annotation accuracy',
  source_taxonomy_policy='Per-category aggregates are descriptive only. COCO67 dining_table, OI broad Table and LVIS1050 remain separate native strata. LVIS Cabinet181 and OI Cabinetry are not automatically equivalent. Legacy uncertain Desk/Cabinet targets remain masked.',
  ready_for_broad_seven_class_training=False,
  recommendation='Prioritize additional diverse labelled indoor scenes and source-unit/partial-label policy before broad seven-class training. Cabinet has no fresh development supervision; Lamp and Bed development support is sparse. LVIS expansion is Desk-dominated and many views are close workstation surfaces.',
  required_before_training=['Separate explicit training approval and versioned source-native dataset lock','Resolve Cabinet/Desk/Table unit semantics and evaluation strata','Implement/test native federated positive/nonexhaustive/explicit-negative losses; unknown is not background','Add fresh source-group-isolated Cabinet/Lamp/Bed scenes and representative wide rooms','Keep all historical frozen panels out of optimisation and tuning','Do not activate33 existing LVIS overlay annotations without semantic and registration checks'])
 commit(OUT/'combined-training-readiness.json',result);space();print('COMBINED',len(train),'unique train;',len(dev),'fresh dev; known groups',result['known_train_source_components'],result['known_fresh_development_source_components'],'training NOT authorized',flush=True)

if __name__=='__main__':run()
