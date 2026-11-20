"""Build source-grounded review tasks and an inactive hard-negative overlay."""
from collections import Counter,defaultdict
from g5_common import *

def run():
    if (RAW/'review-registry.json').exists():raise FileExistsError('New review version required')
    inventory=read(RAW/'native-object-inventory.json');prepared=manifest();byrow={r['image_id']:r for r in prepared['rows']}
    photos=[];tasks=[]
    from PIL import Image
    for p in inventory['photos']:
        if not p['eligible_for_primary_AB']:continue
        row=byrow[p['image_id']]
        with Image.open(ROOT/p['path']) as im:
            o=im.getexif().get(274,1)
            if im.size!=(p['width'],p['height']) or o not in range(1,9):raise ValueError('Photo coordinate registration')
        key='train-'+str(len(photos)).zfill(3);source_annotations=p['annotations'];native=[]
        for a in source_annotations:
            native.append(dict(annotation_key=a['annotation_key'],label=a['native_label'],category=a['mapped_auxiliary_category'],box=a['box'],dataset=a['dataset'],
                ignored=a['source_native_ignored'],source_sha256=a['annotation_source_sha256'],object_group=a['object_hypothesis_group']))
        protected=[]
        for a in row['annotations']:
            if a['label']=='chair' or a['ignore']:protected.append(dict(box=a['box'],reason='Existing Chair or ignored native annotation'))
        for a in source_annotations:
            if a['mapped_auxiliary_category']=='protected_furniture':
                n=normal(a['native_label'])
                if n in ('chair','armchair','stool','bar stool','music stool','folding chair','rocking chair','highchair','deck chair','wheelchair') or a['source_native_ignored']:
                    protected.append(dict(box=a['box'],reason='Additional native seating/ignored scope; protective only, not activated as a training positive'))
        photo=dict(photo_key=key,image_id=p['image_id'],sha256=p['sha256'],path=p['path'],width=p['width'],height=p['height'],orientation=o,
            identity_group=p['identity_group'],role='optimization',source=p['source'],source_photo_id=p['source_photo_id'],rights=p['licence'],
            existing_positive_annotation_digest=digest(row['annotations']),native=native,protected=protected)
        photos.append(photo)
        groups=defaultdict(list)
        for a in source_annotations:
            if not a['source_native_ignored']:groups[a['object_hypothesis_group']].append(a)
        for gid,variants in groups.items():
            variants.sort(key=lambda a:(a['dataset']!='COCO',a['annotation_key']));a=variants[0]
            if a['mapped_auxiliary_category']=='protected_furniture' and normal(a['native_label']) not in ('chair','armchair','stool','bar stool','highchair','sofa','couch','bed','table','dining table','desk','cabinet','cupboard','lamp'):continue
            task=dict(task_id='native-'+digest([p['image_id'],gid])[:18],photo_key=key,kind='source_annotation',label=a['native_label'],group=a['mapped_auxiliary_category'],box=a['box'],
                source_dataset=a['dataset'],source_annotation_keys=[v['annotation_key'] for v in variants],source_grounded=True,eligible_for_training_review=True,
                Chair_absence_verified=False,model_prediction=False,automatic_approval=False)
            task['hard_negative_geometrically_blocked']=any(overlap(task['box'],b['box']) for b in protected)
            tasks.append(task)
    # Preserve old review as context only; no automatic conversion of decisions.
    q=read(ROOT/'data/open-images/preparation/phase10c2-v1/review-queue.json')['records'];previous=read(ROOT/'data/open-images/preparation/phase10c2-v1/submissions/manual-v1/human-submission.json')['records'];prev={r['item_id']:r for r in previous}
    diagnostic_photos={}
    for item in q:
        if item['kind']!='prediction_diagnostic':continue
        row=byrow[item['image_id']]
        if row['internal_partition']!='inner_development':raise ValueError('Diagnostic source drift')
        if row['image_id'] not in diagnostic_photos:
            key='diagnostic-'+str(len(diagnostic_photos)).zfill(3)
            with Image.open(ROOT/row['path']) as im:o=im.getexif().get(274,1)
            p=dict(photo_key=key,image_id=row['image_id'],sha256=row['sha256'],path=row['path'],width=row['width'],height=row['height'],orientation=o,identity_group=row['identity_group'],role='internal_validation_diagnostic_only',source=row['source'],source_photo_id=row['source_photo_id'],native=[dict(label=a['label'],category='protected_furniture',box=a['box'],dataset=a['native_dataset'],ignored=a['ignore']) for a in row['annotations']],protected=[])
            diagnostic_photos[row['image_id']]=p;photos.append(p)
        tasks.append(dict(task_id=item['item_id'],photo_key=diagnostic_photos[row['image_id']]['photo_key'],kind='saved_model_prediction',label='Chair',group='saved_prediction',
            box=item['box'],score=item['score'],source_dataset='Phase10B LR3e-5 E5 recovery',source_grounded=False,eligible_for_training_review=False,
            previous_decision=prev[item['item_id']]['decision'],previous_evidence=prev[item['item_id']]['evidence'],Chair_absence_verified=False,model_prediction=True,automatic_approval=False,
            hard_negative_geometrically_blocked=True))
    priority={'plants_flowers':0,'people':1,'furniture_parts':2,'people_parts':3,'wall_art':4,'tableware_small_objects':5,'windows_doors':6,'clothing_bags':7,'screens':8,'kitchen_appliances':9,'animals':10,'protected_furniture':11,'saved_prediction':12}
    tasks.sort(key=lambda t:(not t['eligible_for_training_review'],priority.get(t['group'],20),t['hard_negative_geometrically_blocked'],t['photo_key'],t['task_id']))
    registry=dict(schema='phase10c5-fast-registry-v1',production_categories=list(CLASSES),taxonomy=TAXONOMY,taxonomy_sha256=digest(TAXONOMY),
        photos=photos,tasks=tasks,source_inventory_sha256=sha(RAW/'native-object-inventory.json'),source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        source_licensing_replayed=True,model_forwards=0,decisions_prefilled=0)
    save(RAW/'review-registry.json',registry)
    initial=dict(schema=SCHEMA,registry_sha256=digest(registry),taxonomy_sha256=digest(TAXONOMY),reviews=[],proposals=[])
    save(RAW/'empty-review.json',initial)
    furniture=[dict(image_id=r['image_id'],sha256=r['sha256'],identity_group=r['identity_group'],annotations=[a for a in r['annotations'] if not a['ignore']],annotations_digest=digest(r['annotations'])) for r in prepared['rows'] if optimizer_member(r)]
    save(RAW/'hard-negative-dataset-v1.json',dict(schema='phase10c5-prepared-inactive-dataset-v1',existing_furniture_photos=59,existing_native_positive_boxes=182,
        existing_furniture_positives=furniture,source_supported_distractor_inventory='native-object-inventory.json',auxiliary_labels_automatically_activated=False,
        accepted_hard_negatives=[],reviews_required_for_new_absence_claims=True,all_unknown_regions_ignored=True,
        expansion_pool_kept_separate=True,production_categories=list(CLASSES),training_enabled=False,validation_photos_not_in_training=True))
    save(REPORT/'review-summary.json',dict(registry_sha256=digest(registry),training_photos=59,diagnostic_photos=len(diagnostic_photos),
        source_grounded_review_objects=sum(t['source_grounded'] for t in tasks),diagnostic_prediction_objects=56,
        source_review_groups=dict(Counter(t['group'] for t in tasks if t['source_grounded'])),
        source_candidates_with_geometric_separation=sum(t['source_grounded'] and not t['hard_negative_geometrically_blocked'] and t['group']!='protected_furniture' for t in tasks),
        geometric_separation_is_not_Chair_absence_proof=True,human_approved_hard_negatives=0,training_enabled=False))
    print('FAST REVIEW TASKS',len(tasks),'training photos59;diagnostic photos',len(diagnostic_photos),';0new confirmations',flush=True)

if __name__=='__main__':run()
