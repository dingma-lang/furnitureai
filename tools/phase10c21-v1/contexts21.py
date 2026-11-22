"""Attach complete provenance to automatic contextual candidates,not negative labels."""
from common21 import *
from review21 import semantic_protected


def context_kind(annotation):
    name=annotation['native_label'].lower().replace('_',' ')
    if name.startswith('human ') or name in ('hand','foot','head'):return 'people_parts'
    if name=='piano':return 'musical_instruments'
    if any(x in name for x in ('drawer','headboard','crossbar','hinge','knob','tablecloth','cushion')):return 'furniture_parts_or_coverings'
    if name in ('building','house','fireplace','vent','wall socket','countertop','sink'):return 'architectural_or_fixed_details'
    return annotation['context']


def run():
    source={p['image_id']:p for p in read(DATA21/'full-local-native-inventory.json')['photos']};records=[]
    for r in read(DATA21/'native-confuser-context.json')['records']:
        p=source[r['image_id']];a=next(x for x in p['annotations'] if x['annotation_key']==r['annotation_key']);blocked=semantic_protected(a) or a['native_ignored'];tier='C' if blocked else 'B'
        records.append(dict(r,record_kind='native_bbox_overlap_context_only',evidence_tier=tier,context_hypothesis=context_kind(a),native_dataset=a['dataset'],native_category_id=a['native_category_id'],source_sha256=p['sha256'],annotation_source=a['source'],annotation_source_sha256=a['source_sha256'],native_annotation_sha256=a['original_annotation_sha256'],has_instance_mask=a['has_mask'],mask_reference=dict(inventory='full-local-native-inventory.json',annotation_key=a['annotation_key']),completeness=p['native_source_scopes'],photograph_rights=p['rights'],protected_overlap_not_resolved=True,protected_source_object=blocked,Chair_negative_proven=False,training_eligible=False,training_activated=False,human_reviewed=False))
    emit21(DATA21/'automatic-context-candidates-v1.json',dict(records=records,annotation_records_not_independent_objects=True,new_trusted_labels=0))
    emit21(REPORT21/'context-coverage.json',dict(records=len(records),tiers=dict(Counter(r['evidence_tier'] for r in records)),by_context={k:dict(records=sum(r['context_hypothesis']==k for r in records),photos=len({r['image_id'] for r in records if r['context_hypothesis']==k}),source_groups=len({r['source_group'] for r in records if r['context_hypothesis']==k}),trusted_new_negatives=0) for k in sorted({r['context_hypothesis'] for r in records})},records_are_proposed_context_not_confirmed_mistakes=True))


if __name__=='__main__':run()
