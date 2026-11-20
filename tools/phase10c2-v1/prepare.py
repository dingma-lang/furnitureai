"""Metadata-only inventory plus local review preparation; no source mutations."""
from collections import Counter
import sys
from n2 import *
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from common import READINESS
from policy import CORE
import metric_cpu

def run():
    if (RAW.exists() and any(RAW.iterdir())) or (REPORT.exists() and any(REPORT.iterdir())):raise FileExistsError('Fresh preparation version required')
    RAW.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    original=manifest();index={r['image_id']:r for r in original['rows']};ready=read(READINESS)
    full=read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']
    protected={r['image_id'] for r in full if r['proposed_split']!='train'}
    records=[];nominated=[]
    for s in ready['source_rows']:
        r=index.get(s['image_id']);scope=s['source_native_scope'];explicit=scope.get('native_image_level_labels',{}).get('/m/01mzpv')==0 or 232 in scope.get('neg_category_ids',[])
        native_claim=bool(r and metric_cpu.scopes(r)['fp_coverage']['chair'])
        eligible=bool(r and optimizer_member(r) and s['image_id'] not in protected)
        target=[t for t in s['resolved_targets'] if t['comparison_group']=='chair']
        x=dict(image_id=s['image_id'],source_photo_id=s['source_photo_id'],source=s['source'],sha256=s['sha256'],
            identity_group=r['identity_group'] if r else s['known_source_component'],historical_partition=s['split'],internal_partition=r['internal_partition'] if r else ('excluded_from_active_pool' if s['split']=='train' else 'external_development'),
            category_annotations=[dict(category=t['comparison_group'],native_dataset=t['native_dataset'],native_id=t['native_id'],box=t['original_annotation'].get('box',t['original_annotation'].get('bbox')),scope=t['native_scope']) for t in s['resolved_targets']],
            chair_annotation_count=len(target),source_native_scope=scope,chair_explicit_native_negative=explicit,
            native_chair_completeness_claim=native_claim,human_verified_chair_complete=False,unannotated_chairs_may_be_present=True,
            rights_status=s.get('rights_status','not recorded'),quarantine_status=s.get('quarantine_status','not recorded; no new eligibility inferred'),training_partition_eligible=eligible,
            eligible_for_new_negative_supervision=False,requires_human_scope_review=bool(eligible and native_claim),
            candidate_reason='Native exhaustiveness claim; independent Chair completeness review required' if eligible and native_claim else 'No verified Chair-negative evidence; partition/scope remains inactive')
        records.append(x)
        if x['requires_human_scope_review']:nominated.append(r)
    if len(nominated)!=9 or any(x['chair_explicit_native_negative'] for x in records if x['historical_partition']=='train'):raise ValueError('Negative-evidence inventory drift')
    save(RAW/'negative-candidates-v1.json',dict(schema='phase10c2-category-negative-inventory-v1',source_readiness_sha256=sha(READINESS),
        dataset_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),records=records,
        protected_historical_development_or_frozen_IDs=sorted(protected),frozen_photos_opened=0,positive_images_never_assumed_negative=True))
    save(RAW/'trusted-chair-evidence-v1.json',dict(schema='phase10c2-trusted-evidence-v1',records=[],trusted_records=0,training_enabled=False))
    save(RAW/'optimization-row-bindings.json',{r['image_id']:digest(r) for r in original['rows'] if optimizer_member(r)})
    sample=read(ROOT/'reports/chair-fp-investigation-v1/review-sample.json')['records'];queue=[]
    for s in sample:
        r=index[s['image_id']]
        queue.append(dict(item_id=s['sample_id'],kind='prediction_diagnostic',image_id=r['image_id'],sha256=r['sha256'],identity_group=r['identity_group'],historical_partition=r['historical_partition'],internal_partition=r['internal_partition'],
            prediction_index=s['index'],box=s['box'],score=s['score'],native_status=s['status'],source_review_sample_sha256=sha(ROOT/'reports/chair-fp-investigation-v1/review-sample.json'),
            permitted_training_evidence=False))
    for i,r in enumerate(nominated):
        queue.append(dict(item_id=f'scope-{i+1:03d}',kind='training_scope_review',image_id=r['image_id'],sha256=r['sha256'],identity_group=r['identity_group'],historical_partition=r['historical_partition'],internal_partition=r['internal_partition'],
            category='chair',native_category=next(a['native_id'] for a in r['annotations'] if a['label']=='chair'),permitted_training_evidence=True))
    save(RAW/'review-queue.json',dict(schema='phase10c2-review-queue-v1',records=queue,decisions=[],human_confirmations=0))
    counts=dict(prospective_training_photographs=120,active_optimization_photographs=59,internal_validation=19,excluded_prospective_train=42,
        external_development=36,trustworthy_new_chair_negative_records=0,explicit_chair_negative_photos_in120=0,human_verified_complete_training_photos=0,
        priority_training_scope_reviews=9,other_optimization_photos_without_evidence=50,diagnostic_prediction_reviews=56,scored_FP_reviews=48,unscored_controls=8,
        total_review_items=65,corrected_training_ready=False)
    save(REPORT/'candidate-manifest-index.json',dict(schema='phase10c2-private-inventory-index-v1',counts=counts,
        private_manifest=str((RAW/'negative-candidates-v1.json').relative_to(ROOT)),private_manifest_sha256=sha(RAW/'negative-candidates-v1.json'),
        queue_path=str((RAW/'review-queue.json').relative_to(ROOT)),queue_sha256=sha(RAW/'review-queue.json'),raw_photographs_or_private_dataset_uploaded=False))
    common=dict(approved=False,training_authorized=False,base='models/owlv2-original-baseline-20261103',base_sha256='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7',
        data_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),dataset_sha256=digest(original),
        evidence_sha256=digest([]),legacy_weights=dict(classification=1,l1=5,giou=2,gamma=2,positive_alpha=.25,negative_alpha=.75),
        learning_rate=1e-5,seed=20261118,weight_decay=.01,epochs=5,max_updates=75,batch_size=1,accumulation=4,precision='FP32',max_queue_seconds=3600,
        max_updates_both_arms=150,allocated_VRAM_stop_bytes=int(6.5*2**30),max_new_storage_bytes=4*2**30,
        selection='No test/exposed-panel tuning; paired fixed-epoch mechanism comparison only; independent validation currently unavailable',
        training_photos=59,training_groups=56,internal_validation_photos=19,protected_panels_not_used_for_tuning=True,per_image_predictions_before_CPU_metrics=True,
        smoke_required=True,frozen_backbone_and_objectness=True,trainable='same12 class+box tensors in both arms',phase10D_box_only_authorized=False)
    save(REPORT/'arm-A-phase10b.json',dict(common,mode='phase10b',enabled=False,validated=False,approval_reference=None))
    save(REPORT/'arm-B-corrected.json',dict(common,mode='trusted_chair_negative_v1',enabled=False,validated=False,approval_reference=None,
        row_bindings_path=str((RAW/'optimization-row-bindings.json').relative_to(ROOT)),row_bindings_sha256=sha(RAW/'optimization-row-bindings.json'),blocked='Zero trusted Chair-negative evidence; no effective loss-only A/B contrast until scope reviews are accepted'))
    save(REPORT/'experiment-protocol.json',dict(approved=False,status='BLOCKED_PENDING_EVIDENCE_AND_VALID_EVALUATION',arms=2,expected_updates_per_arm=75,max_updates_total=150,
        compare='Same pixels, positives, full source rows, seed, initialization, masks outside trusted added negatives, coefficient normalization and class/head scope; only trusted Chair-negative mask differs',
        exclude='All56 reviewed predictions are internal validation and never train; no new images/labels in primary A/B',
        no_tuning_on_exposed_19=True,no_frozen_scoring=True,independent_80precision_90recall_claim_allowed=False,
        metrics=['P/R@.15','precision@matched reference recall','AP50','AP75','ChairFP/FN','Sofa/BedTP/AP75','per-image predictions','score distributions','runtime/VRAM'],
        success='Primary: >=30% fewer Chair FP at0.15 with Chair FN not greater than original, precision>=original+3pp at original recall, no Sofa/BedTP loss and no per-category AP75 drop>2pp. B must improve over A as well. Exploratory internal evidence only.',
        reject='No measurable added trusted-negative terms, uncertain evidence, leakage, changing inputs between arms, nonfinite values, resource limits or failed gates; do not expand budget',
        unresolved='Need trusted scope reviews on existing optimization images and a genuinely unexposed evaluation protocol; no current evidence supports a valid independent accuracy comparison'))
    print(counts)

if __name__=='__main__':run()
