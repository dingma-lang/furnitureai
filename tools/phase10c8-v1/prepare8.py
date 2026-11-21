"""Freeze this authorized A/B and metric protocol before any new training."""
import subprocess,importlib.metadata
from common8 import *

def run():
    if RUN.exists() or REPORT.exists():raise FileExistsError('Fresh versioned output required')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='experiment/phase10c8-controlled-negatives':raise ValueError('Dedicated branch required')
    prepared=manifest();train,dev=subsets(prepared);bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
    from masks6 import validate_bundle,authorized
    validate_bundle(bundle,prepared['rows'],frozen,observations)
    c7=read(ROOT/'reports/phase10c7-v1/results.json')
    if c7['actual_new_Chair_negative_queries']!=292 or c7['positive_or_protected_overlap_targets'] or c7['coordinate_membership_discrepancies'] or len(bundle['records'])!=41:raise ValueError('Smoke/evidence gate failed')
    corrected=dict(mode='source_polygon_negative_v1',enabled=True,approved=True,approval_reference='User Phase10C.8 matched A/B training authorization',evidence_sha256=digest(bundle),manifest_sha256=digest(prepared),legacy_loss=dict(classification=1,L1=5,GIoU=2,alpha=.25,gamma=2))
    authorized(corrected,bundle,prepared)
    reg={r['image_id']:r for r in registry()['photos'] if r['role']=='optimization'}
    for row in train+dev:
        image_path(row,row['internal_partition'],prepared)
        if row in train and reg[row['image_id']]['existing_positive_annotation_digest']!=digest(row['annotations']):raise ValueError('Positive/ignored scope drift')
    if not all(s in str(TAXONOMY).lower() for s in ('bar','seating','footstool','ottoman','step')):raise ValueError('Confirmed Chair/stool taxonomy changed')
    model_files=read(ROOT/'reports/phase10a-v1/input-contract.json')['model_files'];files={str((BASE/n).relative_to(ROOT)):h for n,h in model_files.items()}
    for folder in (HERE,ROOT/'tools/phase10c6-v1'):
        for p in folder.glob('*.py'):files[str(p.relative_to(ROOT))]=sha(p)
    for p in (SOURCE/'source-negative-evidence.json',SOURCE/'training-evidence-manifest.json',ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json',ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json',ROOT/'tools/phase10a-v1/expanded.py',ROOT/'tools/phase10c-v1/metric_cpu.py'):
        files[str(p.relative_to(ROOT))]=sha(p)
    eval_protocol=dict(scope='Already exposed19-photo training-derived internal diagnostics; not independent acceptance',
        row_ids=[r['image_id'] for r in dev],rows_sha256=digest(dev),category_order=list(CLASSES),prompts=PROMPTS,confidence=.15,ap_score_floor=.001,nms_iou=.5,nms_cap=300,IoU=[.5,.75],
        AP='macro supported-class101point interpolation',partial_labels='Unchanged native scopes; COCO absent labels unknown; LVIS non-exhaustive flags preserved',
        inference_precision='FP32 identical to historical validation; training/probes use FP16 autocast',checkpoint_comparisons=[1,3,5],primary_endpoint='epoch5, fixed before results; no adaptive best-epoch selection',
        matched_recall='PR-envelope precision at original reference recall, descriptive only; no threshold adopted',official_reserved_frozen_images=False,
        metric_implementation='unchanged tools/phase10c-v1/metric_cpu.py',metric_implementation_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'))
    settings=dict(initialization='original OWLv2',base_sha256=model_files['model.safetensors'],learning_rate=1e-5,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,
        seed=6106,batch_size=1,accumulation=4,last_group_normalization='actual group size3',gradient_clip_norm=1.,precision='FP16 autocast; FP32 loss/weights',loss_coefficients=dict(classification=1,L1=5,GIoU=2,alpha=.25,gamma=2),
        scaler_init=128,scaler_growth_interval=2000,max_epochs=5,max_updates=75,trainable='same12 class_head/box_head tensors',frozen='vision/text/objectness/all remaining parameters',
        probe_epochs=list(PROBE_EPOCHS),probe_photos='exact24 C7 source photos',probe_objective='actual current GradScaler scale; autograd.grad on additional-negative and used classification loss; no probe optimizer updates',
        snapshots='after every committed optimizer update; unique immutable filenames',nonfinite_policy='stop arm immediately, no replay or loss-weight tuning')
    configs={arm:dict(settings,arm=arm,negative_policy='legacy_Phase10B' if arm==ARMS[0] else 'source_polygon_visible_Chair_negative',source_configuration=corrected if arm==ARMS[1] else None) for arm in ARMS}
    common_configs=[{k:v for k,v in configs[a].items() if k not in ('arm','negative_policy','source_configuration')} for a in ARMS]
    if common_configs[0]!=common_configs[1]:raise ValueError('An unintended arm difference was introduced')
    intent=dict(training_authorized=True,approval_reference='User Phase10C.8 explicit bounded A/B',arms=list(ARMS),max_updates_per_arm=75,max_total_updates=150,max_epochs=5,max_queue_seconds=3600,max_vram_bytes=MAX_VRAM,files=files,
        settings=settings,configs=configs,evaluation_protocol=eval_protocol,training_rows_sha256=digest(train),development_rows_sha256=digest(dev),evidence_sha256=sha(SOURCE/'source-negative-evidence.json'),
        environment={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow','scipy')},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),Phase10D_authorized=False)
    verify_binding(intent)
    disk=os.statvfs(ROOT)
    if disk.f_bavail*disk.f_frsize<8*2**30:raise ValueError('Insufficient space for immutable per-update recovery checkpoints')
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True);save(RUN/'binding.json',intent);save(REPORT/'evaluation-protocol.json',{k:v for k,v in eval_protocol.items() if k!='row_ids'});save(REPORT/'arm-configurations.json',configs)
    save(REPORT/'preflight.json',dict(status='PASS',train_photos=59,train_groups=56,positive_boxes=182,internal_dev_photos=19,source_regions=41,
        original_checkpoint_sha256=model_files['model.safetensors'],same_initialization_and_settings=True,sole_intentional_difference='Chair-negative mask policy',
        seven_categories_unchanged=True,stools_policy_sha256=digest(TAXONOMY),validation_never_optimization=True,model_files_verified=len(model_files),source_evidence_validated=True))
    print('PREFLIGHT PASS59optimization/19internaldev/182positiveboxes/41regions;two matched arms;max150updates',flush=True)

if __name__=='__main__':run()
