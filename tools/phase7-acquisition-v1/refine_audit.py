"""Independent category-role accounting and metadata-only frozen/source-group checks."""
import collections as C
import json
from core import *
from audit_lvis import connect
from cross_reference import identity_graph,photo_id

WHOLE_SUBTYPES={19,170,285,301,346,358,367,438,464,558,610,731,889,1051}
PART_OR_OUTDOOR={627,628}

def run():
 db=connect(OUT/'furniture-index.sqlite');db.row_factory=__import__('sqlite3').Row
 mismatch=list(db.execute('SELECT i.split,i.id,i.width,i.height,c.width,c.height FROM images i JOIN coco c ON i.id=c.id WHERE i.width!=c.width OR i.height!=c.height'))
 if mismatch:raise ValueError('COCO/LVIS metadata geometry registration mismatch')
 old=read(OUT/'existing-photo-cross-reference.json');summary={}
 for role,cats in [('seven_core_classes',set(CORE)),('whole_objects_including_distinct_subtypes',set(CORE)|WHOLE_SUBTYPES),('parts_or_outdoor_not_core_furniture',PART_OR_OUTDOOR)]:
  eligible=[]
  for r in old['records']:
   n=sum(c['lvis_instances'] for c in r['category_comparison'] if c['native_lvis_category_id'] in cats)
   if n:eligible.append((r,n))
  summary[role]=dict(images=len(eligible),instances_boxes_and_masks=sum(n for _,n in eligible),historical_partitions=dict(C.Counter(r['historical_partition'] for r,_ in eligible)))
 more=[]
 for r in old['records']:
  annotations=[a for a in r['category_comparison'] if a['native_lvis_category_id'] in (181,361,626)]
  if annotations:
   more.append(dict(coco_image_id=r['coco_image_id'],historical_partition=r['historical_partition'],historical_status=r['historical_status'],
    lvis_split=r['lvis_split'],new_core_class_annotations=annotations,licensing_status=r['licensing_status'],
    preserved_quarantine=r['historical_status']=='quarantined',
    additional_checks=['native_Cabinet_Desk_Lamp_unit_comparability','COCO_LVIS_mask_registration','source_creator_scene_group_confirmation','preserve_original_split_and_quality_status']))
 uf,blocked,old_summary,rights=identity_graph()
 inventory=read(ROOT/'reports/phase7-data-strategy-v1/protected-source-inventory-v1.json')['records']
 protected_exact=C.defaultdict(list);group_members=C.defaultdict(list)
 for r in inventory:
  for alias in r['source_aliases']:
   if alias.startswith('flickr-photo:'):
    protected_exact[alias].append(r);group_members[uf.find(alias)].append(r)
 counts=C.Counter();frozen_hits=[];historical_hits=[]
 for row in db.execute('SELECT i.split,i.id,c.flickr FROM images i JOIN coco c ON i.id=c.id'):
  photo=photo_id(row['flickr'])
  if not photo:continue
  exact=protected_exact.get(photo,[]);related=group_members.get(uf.find(photo),[])
  for key,records in [('exact',exact),('known_source_group',related)]:
   parts={r['partition'] for r in records}
   for part in parts:counts[row['split'],key,part]+=1
  if related:
   item=dict(lvis_image_id=row['id'],lvis_split=row['split'],source_photo_id=photo,
     matched_historical_source_ids=sorted({r['canonical_id'] for r in related}),historical_partitions=sorted({r['partition'] for r in related}),
     exact_source_match=bool(exact),known_source_group=uf.find(photo))
   historical_hits.append(item)
   if 'frozen_evaluation' in item['historical_partitions']:frozen_hits.append(item)
 candidate=read(OUT/'proposed-photo-rights-audit-v1.json');old_ids={r['coco_image_id'] for r in rights};part_groups=C.defaultdict(set)
 for r in candidate['images']:
  photo=r['source_photo_id'];group=uf.find(photo)
  if r['coco_image_id'] in old_ids or protected_exact.get(photo) or group in group_members or group in blocked:raise ValueError('Proposed metadata candidate hits historical/blocked identity')
  part_groups[group].add(r['proposed_partition'])
 if any(len(p)>1 for p in part_groups.values()):raise ValueError('Known-group proposal leakage')
 commit(OUT/'frozen-and-source-group-isolation.json',dict(metadata_only=True,coco_lvis_dimensions_match=True,
  source_partition_overlap_counts=[dict(lvis_split=k[0],match_type=k[1],historical_partition=k[2],images=v) for k,v in sorted(counts.items())],
  historical_source_group_hits=historical_hits,frozen_source_group_hits=frozen_hits,
  proposed_ids_and_known_groups_isolated=True,known_candidate_photo_components=len(part_groups),
  candidates_with_known_creator_alias=sum(bool(r['known_creator_aliases']) for r in candidate['images']),
  actual_independent_scene_count=None,remaining_identity_risks='399 candidate creator aliases are unknown; source components are not confirmed physical identities. Perceptual duplicates and source-account alias resolution require later evidence and photographs.',
  historical_partitions_changed=0,frozen_pixels_opened=0,official_holdout_scoring=0))
 commit(OUT/'annotation-overlay-summary.json',dict(role_accounting=summary,
  all_tracked_categories_including_lampshades=old['summary'],new_core_class_supervision=more,
  new_core_class_total_instances=sum(sum(c['lvis_instances'] for c in r['new_core_class_annotations']) for r in more),
  no_new_independent_images=True,
  interpretation='LVIS annotation overlays provide source supervision, not proven additional physical objects. Lampshades and outdoor lampposts are never counted as whole indoor furniture. Other subtypes retain distinct native IDs.',
  no_quality_or_licensing_quarantine_removed=True))
 db.close();space();print('REFINED',summary,'Frozen-related source photos',len(frozen_hits),flush=True)

if __name__=='__main__':run()
