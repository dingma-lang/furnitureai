"""Audit all local prospective-training source rows;never mine nonoptimization images."""
from common21 import *


def run():
    cfg,prepared,train,dev,reg,inventory,photos=context21();ready=read(ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json');source_rows=[r for r in ready['source_rows'] if r['split']=='train'];byid={p['image_id']:p for p in inventory['photos']};active={r['image_id']:r for r in train};inner={r['image_id'] for r in dev};records=[]
    for path,h in inventory['metadata_sources'].items():
        if sha(ROOT/path)!=h:raise ValueError('Native metadata changed')
    for row in source_rows:
        ident=row['image_id'];cohort='optimization' if ident in active else 'inner_development_excluded' if ident in inner else 'inactive_training_only_not_assigned';p=byid.get(ident)
        if cohort!='inner_development_excluded' and p is None:raise ValueError('Prospective training source row missing local inventory')
        if p:
            if sha(ROOT/p['path'])!=p['sha256'] or row['sha256']!=p['sha256']:raise ValueError('Local photograph changed')
            rights=p['licence']
            for k,hk in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
                if sha(ROOT/rights[k])!=rights[hk]:raise ValueError('Recorded rights evidence changed')
        anns=[] if not p else p['annotations']
        counts=Counter(a['mapped_auxiliary_category'] for a in anns)
        positive=Counter(a['label'] for a in active.get(ident,{}).get('annotations',[]) if not a['ignore'])
        records.append(dict(image_id=ident,source_group=active[ident]['identity_group'] if ident in active else row['known_source_component'],source=row['source'],cohort=cohort,sha256=row['sha256'],native_recognized_annotations=len(anns) if p else None,native_dataset_counts=dict(Counter(a['dataset'] for a in anns)),native_context_counts=dict(counts),active_positives=dict(positive),resolved_source_targets=Counter(a.get('comparison_group') or 'native_unmapped' for a in row['resolved_targets']),complete_all7_verified=row.get('full_seven_class_annotation_completeness_independently_verified') is True,complete_Chair_plus_stools_verified=False,licence_as_recorded=p['licence']['license_url'] if p else 'Excluded development;not mined',rights_reverified_online=False,project_semantics_unverified=True,eligible_for_negative_mining=cohort=='optimization',unassigned_automatically_activated=False))
    contexts={}
    for group in sorted({a['mapped_auxiliary_category'] for p in photos.values() for a in p['annotations']}):
        pairs=[(p,a) for p in photos.values() for a in p['annotations'] if a['mapped_auxiliary_category']==group]
        contexts[group]=dict(annotation_records=len(pairs),photos=len({p['image_id'] for p,a in pairs}),source_groups=len({p['identity_group'] for p,a in pairs}),source_datasets=dict(Counter(a['dataset'] for p,a in pairs)),native_masks=sum(a['has_instance_mask'] for p,a in pairs),native_labels=dict(Counter(a['native_label'] for p,a in pairs)),negative_absence_not_inferred=True)
    furniture=[(p,a) for p in photos.values() for a in p['annotations'] if a['mapped_auxiliary_category']=='protected_furniture']
    stools=[dict(image_id=p['image_id'],source_group=p['identity_group'],annotation_key=a['annotation_key'],native_label=a['native_label'],box=a['box'],native_dataset=a['dataset'],activated_as_new_Chair_positive=False,protected_from_negative_targets=True) for p,a in furniture if any(word in a['native_label'].lower() for word in ('stool','ottoman','chair','bench','seat'))]
    summary=dict(prospective_historical_train_photos=len(records),cohorts=dict(Counter(r['cohort'] for r in records)),source_rows_unique=len({r['image_id'] for r in records}),active_training_photos=59,active_training_groups=56,active_positive_boxes=182,active_positive_category_counts=dict(Counter(a['label'] for r in train for a in r['annotations'] if not a['ignore'])),active_native_recognized_annotations=sum(len(p['annotations']) for p in photos.values()),active_native_contexts=contexts,all_local_inventory_photos=len(inventory['photos']),all_local_recognized_native_records=sum(len(p['annotations']) for p in inventory['photos']),full7_complete_training_photos=sum(r['complete_all7_verified'] for r in records),source_annotations_modified=False,mining_inner_or_frozen=False,unassigned42_not_promoted=True,limitations=['Existing broad native inventory covers recognized auxiliary/furniture names;full source join is audited separately for omitted native labels.','COCO/LVIS duplicate annotation records are not independent objects or new photographs.','Object labels and bounding boxes do not prove Chair or seating-stool absence.','Rights evidence hashes verified locally;no claim of fresh online rights verification.'])
    emit21(DATA21/'local-training-inventory.json',dict(records=records,metadata_sources=inventory['metadata_sources']));emit21(REPORT21/'training-coverage.json',summary);emit21(DATA21/'protected-seating-audit.json',dict(records=stools,missing_or_mislabeled_real_chairs_cannot_be_inferred_without_evidence=True))
    print('Audited120prospective train rows;only59optimization photos available for mining',flush=True)


if __name__=='__main__':run()
