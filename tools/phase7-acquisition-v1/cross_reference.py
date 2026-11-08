"""Local COCO/LVIS overlays, exact-photo rights joins and conservative proposal selection."""
import collections as C
import hashlib
import importlib.util
import json
import re
import sys
from core import *
from reader import stream
from audit_lvis import connect

def photo_id(url):
 for pattern in (r'/(\d+)_[^/]+\.(?:jpg|png)',r'/photos/[^/]+/(\d+)',r'[?&]id=(\d+)'):
  m=re.search(pattern,url or '',re.I)
  if m:return 'flickr-photo:'+m[1]
 return None

def iou(a,b):
 ax,ay,aw,ah=a;bx,by,bw,bh=b
 overlap=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
 return overlap/(aw*ah+bw*bh-overlap) if aw*ah+bw*bh>overlap else 0

def exact_pairs(native,original,cid):
 mapping={232:62,982:63,77:65}
 if cid not in mapping:return [],list(native)
 edges=sorted([(iou(a['bbox'],b['bbox']),i,j) for i,a in enumerate(native) for j,b in enumerate(original) if b['category_id']==mapping[cid]],reverse=True)
 left=set();right=set();pairs=[]
 for score,i,j in edges:
  if score>=.5 and i not in left and j not in right:
   left.add(i);right.add(j);pairs.append(dict(lvis_annotation_id=native[i]['id'],coco_annotation_id=original[j]['id'],iou=score))
 return pairs,[a for i,a in enumerate(native) if i not in left]

def identity_graph():
 sys.path.insert(0,str(ROOT/'tools/phase6c-v1'))
 spec=importlib.util.spec_from_file_location('phase7_old_identity_graph',ROOT/'tools/phase6c-v1/integrity.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 ledger=read(ROOT/'data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json')['records']
 uf,protected,summary,_=mod.build_graph(ledger)
 # Exclude all previously audited IDs/groups, including the 275 HTTP429 and unresolved source records.
 for r in ledger:protected.add(uf.find(r['source_identity']))
 return uf,protected,summary,ledger

def candidate_partition(rows):
 """Proposed only: whole known groups, deterministic ranking, no historic partition changes."""
 groups=C.defaultdict(list)
 for r in rows:groups[r['known_source_group']].append(r)
 assigned={};development=0;target=round(len(rows)*.2)
 for group,members in sorted(groups.items(),key=lambda kv:hashlib.sha256(('phase7-lvis-20261106:'+kv[0]).encode()).hexdigest()):
  part='development' if development+len(members)<=target else 'train'
  if part=='development':development+=len(members)
  for r in members:assigned[r['coco_image_id']]=part
 return assigned

def preserved_status(record,fallback):
 return record.get('status',record.get('eligibility_status',fallback))

def run():
 db=connect(OUT/'furniture-index.sqlite');db.row_factory=__import__('sqlite3').Row
 targets={r['coco_image_id']:r for r in read(ROOT/'reports/phase7-data-strategy-v1/lvis-overlap-targets-v2.json')['targets']}
 source_ann=C.defaultdict(list);source_cats={}
 for split in ('train','val'):
  path=ROOT/f'data/coco/metadata/phase6b-v1/extracted/annotations/instances_{split}2017.json'
  for kind,r in stream(path):
   if kind=='images':db.execute('INSERT INTO coco VALUES(?,?,?,?,?,?,?)',(r['id'],split,r['width'],r['height'],r['coco_url'],r.get('flickr_url'),r['license']))
   elif kind=='annotations' and r['image_id'] in targets:source_ann[r['image_id']].append(r)
   elif kind=='categories':source_cats[r['id']]=r['name']
  db.commit();space()
 # The identity graph is metadata-only and retains creator aliases from previously verified same-photo evidence.
 uf,protected,identity_summary,rights=identity_graph();by_photo={r['source_identity']:r for r in rights}
 existing_rights=read(ROOT/'data/coco/image-pilots/phase6d-v1/license-and-provenance-ledger.json')['records']
 acquired={r['coco_image_id']:r for r in existing_rights}
 old_coco_status={r['coco_image_id']:r for r in read(ROOT/'data/coco/image-pilots/phase6d-v1/screening-v2.json')['images']}
 old_oi_status={r['image_id']:r for r in read(ROOT/'data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json')['images']}
 for r in read(ROOT/'data/open-images/image-pilots/phase6a-v1/screening-v3.json')['images']:old_oi_status[r['image_id']]=r
 oi_rights={r['image_id']:r for r in read(ROOT/'data/open-images/image-pilots/phase5-v1/licensing-audit.json')['records']}
 oi_rights.update({r['image_id']:r for r in read(ROOT/'data/open-images/preparation/phase6a-v1/license-attribution-ledger.json')['records']})
 overlay=read(OUT/'existing-image-lvis-overlays.json')['records'];per_image=C.defaultdict(list)
 for r in overlay:per_image[r['lvis_annotation']['image_id']].append(r['lvis_annotation'])
 overlap=[];gains=C.Counter();all_new_classes=C.Counter();source_partitions=C.Counter()
 for iid,target in sorted(targets.items()):
  matches=list(db.execute('SELECT * FROM images WHERE id=?',(iid,)))
  if len(matches)>1:raise ValueError('Duplicate source split image')
  im=matches[0] if matches else None;annotations=per_image[iid];annotations_by_class=C.defaultdict(list)
  for a in annotations:annotations_by_class[a['category_id']].append(a)
  if iid in acquired:
   lic=acquired[iid];rights_status='existing_verified_CC_BY_2.0_same_COCO_photo';historic=old_coco_status[iid]
   rights_evidence=dict(creator=lic['creator'],creator_url=lic['creator_url'],photo_license=lic['photo_license'],source_url=lic['original_source_page'],checked_utc=lic['rights_checked_utc'],rights_record_sha256=lic['rights_record_sha256'])
  else:
   oid=target['historical_source'].split(':',1)[1];lic=oi_rights.get(oid,{});historic=old_oi_status.get(oid,{})
   rights_status='historical_OpenImages_evidence_requires_same_photo_content_registration_check'
   rights_evidence=dict(licence_record=lic,same_source_photo_id=True,content_registration_verified=False)
  category_comparison=[]
  for cid,anns in sorted(annotations_by_class.items()):
   pairs,unmatched=exact_pairs(anns,source_ann[iid],cid);gains[cid]+=len(anns)
   if cid in (361,181,626):all_new_classes[cid]+=len(anns)
   category_comparison.append(dict(native_lvis_category_id=cid,lvis_instances=len(anns),boxes=len(anns),masks=sum(bool(a.get('segmentation')) for a in anns),
    category_absent_from_coco_taxonomy=cid in (361,181,626),same_object_iou50_pairs=pairs,
    unmatched_at_iou50_ids=[a['id'] for a in unmatched] if cid in (232,982,77) else None,
    comparison_policy='Native Chair/Sofa/Bed only: deterministic same-category IoU>=0.5 greedy pairing. Table and other subtypes are not declared equivalent. Unmatched is not proof of an additional physical object.'))
  status=preserved_status(historic,target['historical_status'])
  source_partitions[target['historical_partition']]+=int(bool(annotations))
  overlap.append(dict(coco_image_id=iid,historical_source=target['historical_source'],historical_partition=target['historical_partition'],historical_status=status,
   lvis_split=im['split'] if im else None,lvis_image_available=im is not None,
   lvis_native_scope=dict(neg_category_ids=json.loads(im['neg']),not_exhaustive_category_ids=json.loads(im['nex'])) if im else None,
   available_furniture_lvis_instances=len(annotations),native_furniture_categories=sorted(annotations_by_class),category_comparison=category_comparison,
   coco_original_annotations=source_ann[iid],licensing_status=rights_status,rights_evidence=rights_evidence,
   annotation_reuse_authorized_for_training=False,new_independent_photo=False,partition_changed=False,
   unresolved='Preserve old eligibility; native label/box-mask and content-registration checks still required; no prior quarantine is lifted.'))
 commit(OUT/'existing-photo-cross-reference.json',dict(records=overlap,summary=dict(targets=77,lvis_image_matches=sum(r['lvis_image_available'] for r in overlap),
  images_with_relevant_furniture=sum(bool(r['available_furniture_lvis_instances']) for r in overlap),new_lvis_source_instances=sum(gains.values()),native_category_gains=dict(gains),
  coco_missing_core_class_instances=dict(all_new_classes),historical_partitions_with_furniture=dict(source_partitions),
  counts_are_annotation_overlays_not_new_photos=True,official_frozen_pixels_opened=0,model_scoring=0)))
 # Metadata-only future rights-audit pool: source LVIS train + COCO train; no old candidate ID/group is retried.
 banned_ids={r['coco_image_id'] for r in rights}|set(targets)
 exclusions=C.Counter();candidate_rows=[]
 grouped=C.defaultdict(list)
 for r in db.execute("SELECT f.*,i.width,i.height,i.url,i.neg,i.nex FROM facts f JOIN images i ON f.split=i.split AND f.image=i.id WHERE f.split='train' AND f.category IN (77,181,232,361,626,982,1050)"):
  grouped[r['image']].append(dict(r))
 for iid,facts in grouped.items():
  coco=db.execute('SELECT * FROM coco WHERE id=?',(iid,)).fetchone()
  if iid in banned_ids:exclusions['existing_photo_or_previous_400_audit_id']+=1;continue
  if coco is None or coco['source_split']!='train':exclusions['not_known_COCO_train_metadata']+=1;continue
  if coco['license']!=4:exclusions['not_strict_CC_BY_2.0_metadata_claim']+=1;continue
  photo=photo_id(coco['flickr'])
  if not photo:exclusions['no_exact_source_photo_id']+=1;continue
  group=uf.find(photo)
  if group in protected:exclusions['known_historical_or_previously_audited_source_creator_group']+=1;continue
  if min(coco['width'],coco['height'])<256:exclusions['small_source_dimensions']+=1;continue
  good=[]
  for f in facts:
   flags=set(json.loads(f['quality']))
   if not flags&{'degenerate_box','box_outside_image','missing_mask','small_furniture_region'}:good.append(f)
  if not good:exclusions['no_large_structurally_usable_core_label']+=1;continue
  categories=sorted(f['category'] for f in good)
  known_aliases=[r for r in (group,) if r.startswith('flickr-account:')]
  candidate_rows.append(dict(coco_image_id=iid,lvis_image_id=iid,lvis_split='train',coco_source_split='train',
   source_photo_id=photo,known_source_group=group,known_creator_aliases=known_aliases,
   creator_identity_verified=False,physical_scene_identity_verified=False,
   original_source_asset_url=coco['flickr'],source_page_resolver='https://www.flickr.com/photo.gne?id='+photo.split(':')[1],
   official_image_object_url=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{iid:012d}.jpg',
   width=coco['width'],height=coco['height'],metadata_license_id=4,metadata_license_url='https://creativecommons.org/licenses/by/2.0/',
   commercial_rights_status='potentially_compatible_but_individually_unverified',photo_acquisition_eligible=False,
   native_categories=categories,source_furniture_facts=facts,preferred_large_native_categories=categories,
   label_source='LVIS v1 native source annotations',label_confidence='source_annotation_no_model_probability_or_human_confirmation',
   source_image_scope=dict(neg_category_ids=json.loads(facts[0]['neg']),not_exhaustive_category_ids=json.loads(facts[0]['nex'])),
   uncertainty=['individual_photo_rights_unverified','creator_aliases_incomplete','physical_identity_unverified','indoor_scene_not_visually_confirmed','partial_federated_class_coverage','Cabinet_Desk_Table_native_unit_policy_not_resolved']))
 priorities={361:100,181:100,626:100,77:25,982:25,232:25,1050:25};selected=[];used=set()
 for cid,target in priorities.items():
  pool=[r for r in candidate_rows if cid in r['native_categories'] and r['coco_image_id'] not in used]
  pool.sort(key=lambda r:(cid in r['source_image_scope']['not_exhaustive_category_ids'],hashlib.sha256(f"phase7-native-20261106:{cid}:{r['coco_image_id']}".encode()).hexdigest()))
  for r in pool[:target]:
   r['selection_anchor_native_id']=cid;selected.append(r);used.add(r['coco_image_id'])
 # Never weaken label/rights requirements to fill a rare category quota.
 partitions=candidate_partition(selected)
 for r in selected:r['proposed_partition']=partitions[r['coco_image_id']]
 byid={r['coco_image_id']:r for r in selected}
 for r in byid.values():r['original_lvis_furniture_annotations']=[]
 for kind,a in stream(OUT/'lvis_v1_train.json.zip'):
  if kind=='annotations' and a['image_id'] in byid and a['category_id'] in RELEVANT:byid[a['image_id']]['original_lvis_furniture_annotations'].append(a)
 for r in selected:
  if not r['original_lvis_furniture_annotations']:raise ValueError('Selected source annotation missing')
  r['annotation_archive_sha256']=read(OUT/'acquisition-complete.json')['archives']['lvis_v1_train.json.zip']['sha256']
  r['annotation_license']='https://creativecommons.org/licenses/by/4.0/'
  r['human_reviewed']=False
  r['estimated_image_bytes_range']=[150000,400000]
 groups=C.defaultdict(set)
 for r in selected:groups[r['known_source_group']].add(r['proposed_partition'])
 if any(len(p)>1 for p in groups.values()):raise ValueError('Known-group proposal leakage')
 counts={}
 for cid,name in CORE.items():
  relevant=[r for r in selected if any(a['category_id']==cid for a in r['original_lvis_furniture_annotations'])]
  counts[name]=dict(native_id=cid,images=len(relevant),boxes=sum(sum(a['category_id']==cid for a in r['original_lvis_furniture_annotations']) for r in relevant),
   masks=sum(sum(a['category_id']==cid and bool(a.get('segmentation')) for a in r['original_lvis_furniture_annotations']) for r in relevant))
 manifest=dict(schema='phase7-LVIS-future-rights-audit-candidates-v1',images=selected,image_count=len(selected),
  category_distribution=counts,selection_anchor_counts=dict(C.Counter(str(r['selection_anchor_native_id']) for r in selected)),proposed_allocation=dict(C.Counter(r['proposed_partition'] for r in selected)),
  known_source_groups=len(groups),all_rights_individually_unverified=True,photograph_acquisition_approved=False,training_approved=False,
  exact_or_perceptual_duplicate_screening_performed=False,estimated_image_bytes_range=[150000*len(selected),400000*len(selected)],
  hypothetical_per_image_cap=3145728,hypothetical_max_image_transfer=3145728*len(selected),
  duplicate_or_identity_policy='Exact source IDs and known-source components excluded; unknown creators and scene/visual duplicates require later rights and image checks. Source groups are not physical identity proof.',
  deterministic_seed='phase7-native-20261106',partitions_provisional_until_rights_and_identity_checks=True,
  licences='LVIS annotation CC BY4.0 does not grant photo rights. COCO image licence4 is only a claim pending independent same-photo evidence.')
 commit(OUT/'proposed-photo-rights-audit-v1.json',manifest)
 commit(OUT/'future-candidate-exclusion-audit.json',dict(metadata_screened_images=len(grouped),metadata_candidates_before_rank=len(candidate_rows),selected=len(selected),exclusions=dict(exclusions),
  identity_graph=identity_summary,all_old_400_source_ids_excluded=True,previous_HTTP429_ids_not_retried=True,live_creator_requests=0,
  rights_verified_novel_photos=0,physical_identity_independence_verified=False))
 commit(OUT/'next-approval-request.json',dict(scope='Rights evidence audit only for exact IDs in proposed-photo-rights-audit-v1.json',approved=False,
  manifest_sha256=sha(OUT/'proposed-photo-rights-audit-v1.json'),image_count=len(selected),proposed_allocation=manifest['proposed_allocation'],
  max_cumulative_response_body_bytes=128000000,max_new_allocated_storage=536870912,max_attempts_per_source=1,
  destination='data/lvis/rights-audits/phase7-v1/',photo_bodies_authorized=False,head_for_exact_official_image_objects=True,
  no_rate_limit_bypass=True,HTTP429_policy='Stop the provider circuit and preserve unresolved status; no endpoint/proxy rotation or silent alternative IDs',
  no_training_inference_or_scoring=True,photograph_acquisition_requires_another_exact_eligible_manifest=True))
 db.close();space();print('CROSS REFERENCE',sum(bool(r['available_furniture_lvis_instances']) for r in overlap),'existing images with furniture;',len(selected),'new metadata-only rights candidates',flush=True)

if __name__=='__main__':run()
