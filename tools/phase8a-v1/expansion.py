"""Rank new source IDs using existing metadata only. Never resume paused Flickr work."""
import collections as C
import hashlib
import importlib.util
import sqlite3
import sys
from prep_core import *

def select(pool,targets):
 selected=[];used=set();groups=set()
 # Round-robin selection supports an interleaved future queue rather than a long Desk-only prefix.
 queues={cid:sorted([r for r in pool if cid in r['eligible_anchor_ids']],key=lambda r:(-r['core_cooccurrence_count'],hashlib.sha256(f'phase8a-v1:{cid}:{r["coco_image_id"]}'.encode()).hexdigest())) for cid in targets}
 done=C.Counter()
 while any(done[cid]<target for cid,target in targets.items()):
  progress=False
  for cid,target in targets.items():
   if done[cid]>=target:continue
   while queues[cid] and (queues[cid][0]['coco_image_id'] in used or queues[cid][0]['known_source_group'] in groups):queues[cid].pop(0)
   if not queues[cid]:continue
   r=dict(queues[cid].pop(0),selection_anchor_native_id=cid);selected.append(r);used.add(r['coco_image_id']);groups.add(r['known_source_group']);done[cid]+=1;progress=True
  if not progress:break
 return selected,done

def run():
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase8a_local_graph',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 uf,protected,history,old_records=mod.identity_graph();protected_nodes=set(protected)
 prior=read(META/'proposed-photo-rights-audit-v1.json')['images'];prior_ids={r['coco_image_id'] for r in prior}
 for r in read(RIGHTS/'rights-audit.json')['records']:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
 for r in prior:protected_nodes.add(r['source_photo_id'])
 protected={uf.find(n) for n in protected_nodes}
 banned=prior_ids|{r['coco_image_id'] for r in old_records}|{r['coco_image_id'] for r in read(META/'lvis-overlap-targets-v2.json')['targets']} if (META/'lvis-overlap-targets-v2.json').exists() else prior_ids|{r['coco_image_id'] for r in old_records}|{r['coco_image_id'] for r in read(ROOT/'reports/phase7-data-strategy-v1/lvis-overlap-targets-v2.json')['targets']}
 db=sqlite3.connect(f'file:{META}/furniture-index.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 grouped=C.defaultdict(list)
 for r in db.execute("SELECT f.*,i.width,i.height,i.neg,i.nex,c.flickr,c.license,c.source_split FROM facts f JOIN images i ON f.split=i.split AND f.image=i.id JOIN coco c ON i.id=c.id WHERE f.split='train' AND f.category IN (77,181,232,361,626,982,1050)"):grouped[r['image']].append(dict(r))
 pool=[];exclusions=C.Counter()
 for iid,facts in grouped.items():
  r=facts[0]
  if iid in banned:exclusions['prior_consumed_proposed_or_paused_ID']+=1;continue
  if r['license']!=4 or r['source_split']!='train':exclusions['not_COCO_train_CC_BY2_metadata_claim']+=1;continue
  photo=mod.photo_id(r['flickr'])
  if not photo or uf.find(photo) in protected:exclusions['known_historical_or_paused_source_creator_component']+=1;continue
  if min(r['width'],r['height'])<256:exclusions['small_source_photo']+=1;continue
  core_ids=sorted(f['category'] for f in facts)
  if len(core_ids)<2:exclusions['insufficient_multicategory_room_context_proxy']+=1;continue
  good=[]
  for f in facts:
   if f['category'] not in (181,626,77):continue
   flags=set(__import__('json').loads(f['quality']))
   if f['category'] in __import__('json').loads(f['nex']):continue
   if flags&{'degenerate_box','box_outside_image','missing_mask','small_furniture_region'}:continue
   if f['masks']!=f['instances']:continue
   good.append(f['category'])
  if not good:exclusions['no_large_exhaustive_anchor_label']+=1;continue
  pool.append(dict(coco_image_id=iid,lvis_image_id=iid,source_photo_id=photo,known_source_group=uf.find(photo),
   source_page_resolver='https://www.flickr.com/photo.gne?id='+photo.split(':')[1],official_image_object_url=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{iid:012d}.jpg',
   original_flickr_asset_url=r['flickr'],width=r['width'],height=r['height'],metadata_license_claim='https://creativecommons.org/licenses/by/2.0/',
   photo_rights_status='individually_unverified',creator=None,source_split_LVIS='train',source_split_COCO='train',
   eligible_anchor_ids=good,core_native_ids=core_ids,core_cooccurrence_count=len(core_ids),
   native_scope=dict(neg_category_ids=__import__('json').loads(r['neg']),not_exhaustive_category_ids=__import__('json').loads(r['nex'])),
   indoor_scene_quality_verified=False,physical_scene_independence_verified=False,
   context_proxy='>=2 core native categories; metadata proxy only, not visual proof of an indoor or wide-room photograph'))
 selected,anchors=select(pool,{181:45,626:45,77:30});part_counts=C.Counter()
 # Distinct known components; distribute every fifth selected photo to training, keeping96/24 if120 slots exist.
 for index,r in enumerate(selected):r['proposed_partition']='train' if index%5==4 else 'development';part_counts[r['proposed_partition']]+=1
 selected_ids={r['coco_image_id']:r for r in selected}
 for r in selected:r['original_LVIS_core_annotations']=[]
 for kind,a in mod.stream(META/'lvis_v1_train.json.zip'):
  if kind=='annotations' and a['image_id'] in selected_ids and a['category_id'] in (77,181,232,361,626,982,1050):selected_ids[a['image_id']]['original_LVIS_core_annotations'].append(a)
 if set(selected_ids)&prior_ids:raise ValueError('Paused/prior parent ID reused')
 db.close()
 categories={cid:dict(images=sum(any(a['category_id']==cid for a in r['original_LVIS_core_annotations']) for r in selected),boxes=sum(a['category_id']==cid for r in selected for a in r['original_LVIS_core_annotations'])) for cid in (77,181,232,361,626,982,1050)}
 out=dict(schema='phase8a-targeted-new-rights-candidates-v1',images=selected,count=len(selected),proposed_allocation=dict(part_counts),selection_anchor_counts=dict(anchors),category_counts=categories,
  metadata_eligible_pool_size=len(pool),pool_anchors_available={str(cid):sum(cid in r['eligible_anchor_ids'] for r in pool) for cid in (181,626,77)},exclusions=dict(exclusions),
  prior296_paused_ids_all_excluded=True,all_previous400_LVIS_and400_COCO_audit_IDs_excluded=True,new_photograph_rights_verified=0,
  current_network_requests=0,rights_audit_approved=False,photo_acquisition_approved=False,training_approved=False,
  provisional_partition_policy='New source IDs only. Proposed partitions become fixed only after separately approved rights/identity checks. Never change an older ID/group to gain development coverage.',
  rate_limit_policy='Do not execute while source access is suspended. Separate future approval and restored provider access required; no retries of old candidates, alternative endpoints, proxies or identity rotation.',
  source_archive_sha256=sha(META/'lvis_v1_train.json.zip'))
 commit(OUT/'targeted-rights-candidates-v1.json',out)
 commit(OUT/'approval-A-targeted-rights-audit.json',dict(approved=False,scope='Rights evidence only for exact newly proposed IDs',candidate_manifest='targeted-rights-candidates-v1.json',manifest_sha256=sha(OUT/'targeted-rights-candidates-v1.json'),
  count=len(selected),proposed_allocation=dict(part_counts),max_cumulative_response_body_bytes=32000000,max_new_allocated_storage_bytes=134217728,
  max_evidence_attempts_per_candidate=1,min_source_request_interval_seconds=3,one_worker=True,stop_provider_on_HTTP429=True,respect_RetryAfter=True,
  destination='data/lvis/rights-audits/phase8a-v1/',photo_bodies=False,exact_official_image_objects_HEAD_only=True,
  expected_document_transfer_bytes_projection=len(selected)*135000,projected_later_photo_transfer_bytes_range=[len(selected)*150000,len(selected)*400000],
  projected_photo_sizes_not_HEAD_verified=True,provider_access_restoration_required=True,paused296_resume_authorized=False,install_or_training=False))
 print('TARGETED NEW METADATA',len(selected),'allocation',dict(part_counts),'anchors',dict(anchors),'eligible metadata pool',len(pool),flush=True)

if __name__=='__main__':run()
