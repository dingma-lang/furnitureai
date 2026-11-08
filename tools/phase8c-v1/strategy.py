"""Offline next-stage proposals only; no rights requests, acquisition or training."""
import collections as C
import importlib.util
import sqlite3
import sys
from pilot_core import *

def run():
 sys.path.insert(0,str(ROOT/'tools/phase8b-v1'))
 spec=importlib.util.spec_from_file_location('phase8c_strategy_graph',ROOT/'tools/phase8b-v1/audit_core.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 uf,protected,_=old.history_graph();anchors=set(protected)
 for r in read(RIGHTS/'rights-audit.json')['records']:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
  anchors.add(r['source_photo_id'])
 protected={uf.find(a) for a in anchors}
 banned={r['coco_image_id'] for r in read(ROOT/'reports/phase8a-v1/targeted-rights-candidates-v1.json')['images']}
 banned|={r['coco_image_id'] for r in read(META/'proposed-photo-rights-audit-v1.json')['images']}
 banned|={r['coco_image_id'] for r in read(ROOT/'data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json')['records']}
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase8c_source_reader',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross)
 db=sqlite3.connect(f'file:{META}/furniture-index.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 facts=list(db.execute("SELECT f.*,c.flickr,c.license,c.source_split,i.width,i.height,i.neg,i.nex FROM facts f JOIN coco c ON f.image=c.id JOIN images i ON f.split=i.split AND f.image=i.id WHERE f.split='train' AND f.category IN (181,438) AND c.source_split='train' AND c.license=4"));pool={};exclusions=C.Counter()
 for f in facts:
  iid=f['image'];photo=cross.photo_id(f['flickr'])
  if iid in banned:exclusions['all_prior_IDs_including44_and_paused296']+=1;continue
  if not photo or uf.find(photo) in protected:exclusions['known_historical_or_prior_source_group']+=1;continue
  if min(f['width'],f['height'])<256:exclusions['small_dimensions']+=1;continue
  if f['masks']!=f['instances'] or f['category'] in json.loads(f['nex']) or set(json.loads(f['quality']))&{'degenerate_box','box_outside_image','missing_mask','small_furniture_region'}:exclusions['source_geometry_or_completeness']+=1;continue
  r=pool.setdefault(iid,dict(coco_image_id=iid,source_photo_id=photo,known_source_group=uf.find(photo),width=f['width'],height=f['height'],
   source_page_resolver='https://www.flickr.com/photo.gne?id='+photo.split(':')[1],official_image_object_url=f'https://s3.amazonaws.com/images.cocodataset.org/train2017/{iid:012d}.jpg',
   metadata_license_claim='https://creativecommons.org/licenses/by/2.0/',creator=None,rights_status='individually_unverified',
   native_scope=dict(neg_category_ids=json.loads(f['neg']),not_exhaustive_category_ids=json.loads(f['nex'])),native_category_ids=[],annotations=[]))
  r['native_category_ids'].append(f['category'])
 for kind,a in cross.stream(META/'lvis_v1_train.json.zip'):
  if kind=='annotations' and a['image_id'] in pool and a['category_id'] in pool[a['image_id']]['native_category_ids']:pool[a['image_id']]['annotations'].append(a)
 db.close()
 chosen=[];groups=set()
 ranked=sorted(pool.values(),key=lambda r:(438 not in r['native_category_ids'], -max(a['bbox'][2]*a['bbox'][3]/(r['width']*r['height']) for a in r['annotations']),hashlib.sha256(f"phase8c-cab-20261107:{r['coco_image_id']}".encode()).hexdigest()))
 for r in ranked:
  if r['known_source_group'] in groups:continue
  # This proposal intentionally explores single-class cabinet contexts and the
  # distinct file_cabinet subtype. It does not weaken any current training rule.
  r.update(proposed_partition='train' if len(chosen)%5==4 else 'development',native_subtype_mapping_approved=False,
   whole_cabinet_unit_visually_verified=False,annotation_and_photo_licensing_separate=True,
   indoor_scene_verified=False,label_provenance='Native LVIS181 cabinet or438 file_cabinet; not a shared-class relabel')
  chosen.append(r);groups.add(r['known_source_group'])
  if len(chosen)==10:break
 candidate=OUT/'next-cabinet-rights-candidates-v1.json'
 commit(candidate,dict(images=chosen,count=len(chosen),source_archive_sha256=sha(META/'lvis_v1_train.json.zip'),
  proposed_allocation=dict(C.Counter(r['proposed_partition'] for r in chosen)),rights_verified=0,eligibility_exclusions=dict(exclusions),
  previously_excluded_or_paused_IDs_reused=False,policy_difference='Explore source-native cabinet/file_cabinet and single-class contexts. No whole-unit or indoor proof from metadata; no subtype merge or training eligibility change.',
  expected_photo_download_bytes=None,source_object_HEADs_performed=0,all_actions_unapproved=True))
 commit(OUT/'approval-A-next-cabinet-rights-audit.json',dict(approved=False,scope='Rights evidence only; exact new cabinet exploration IDs, not photograph acquisition',
  candidate_manifest=candidate.name,manifest_sha256=sha(candidate),count=len(chosen),proposed_allocation=dict(C.Counter(r['proposed_partition'] for r in chosen)),
  max_cumulative_response_body_bytes=10000000,max_new_allocated_storage_bytes=67108864,max_evidence_attempts_per_candidate=1,
  workers=1,min_source_request_interval_seconds=3,stop_on_provider_restriction=True,respect_RetryAfter=True,
  old296_and_all_prior44_excluded=True,photo_bodies=False,training=False,source_subtype_mapping_not_approved=True,
  expected_evidence_bytes_projection=len(chosen)*135000,destination='data/lvis/rights-audits/phase8d-cabinet-v1/'))
 commit(OUT/'approval-B-conditional-head-study.json',dict(approved=False,execution_ready=False,training_authorization_requested_now=False,
  source_readiness='combined-training-readiness.json',source_readiness_sha256=sha(OUT/'combined-training-readiness.json'),
  prerequisites=['Whole-unit semantics and native strata locked; no door/front-to-whole Cabinet conversion',
   'Production matcher/loss propagates partial/federated masks, with no unknown/objectness negatives',
   'Exact training config and eligible target IDs separately approved',
   'Cabinet development coverage established before seven-class aggregate selection'],
  proposed_future_head_only_configs=2,proposed_future_max_epochs_per_config=5,proposed_future_max_updates_per_config=150,
  proposed_future_max_total_optimizer_updates=300,proposed_future_wall_time_limit_seconds=3600,
  proposed_future_batch_size=1,proposed_future_gradient_accumulation=4,mixed_precision='Conditional CUDA smoke verification',
  soft_VRAM_stop_target_GiB=6.5,hard_VRAM_reservation_guaranteed=False,
  comparison_checkpoints=['immutable_original_OWLv2','immutable_Phase5C_selected','future_expanded_head_only'],
  frozen_evaluation_used=False,downloads=0,no_full_backbone_LoRA_or_deployment=True,
  immediate_recommendation='Prepare/test partial-label matcher and exact dataset lock offline first. Defer executable seven-class training approval.'))
 commit(OUT/'approval-C-synthetic-stages.json',dict(approved=False,priority='Engineering supplement after real Cabinet coverage; never real evaluation replacement',
  assets_request='reports/phase8a-v1/approval-C1-exact-CC0-asset-bundle.json',assets_request_sha256=sha(ROOT/'reports/phase8a-v1/approval-C1-exact-CC0-asset-bundle.json'),
  assets_expected_bytes=11026401,assets_transfer_cap_bytes=25000000,assets_storage_cap_bytes=134217728,
  runtime_metadata_request='reports/phase8a-v1/approval-C2-runtime-metadata-preflight.json',runtime_metadata_transfer_cap_bytes=5000000,runtime_metadata_storage_cap_bytes=20971520,
  binary_installation_approval_ready=False,installation_bytes_unknown=True,
  conditional_render_request='reports/phase8a-v1/approval-C3-conditional-render-pilot.json',views=100,layouts=25,render_output_cap_bytes=1073741824,render_wall_time_seconds=7200,
  rendering_execution_ready=False,rendering_requires_separate_approval_after_assets_runtime_and_install=True,
  no_synthetic_in_real_development_or_frozen_evaluation=True))
 print('NEXT RIGHTS PROPOSAL',len(chosen),'new IDs',dict(C.Counter(r['proposed_partition'] for r in chosen)),
  'native categories',dict(C.Counter(cid for r in chosen for cid in r['native_category_ids'])),'NO REQUESTS MADE',flush=True)

if __name__=='__main__':run()
