"""Compile evidence into inactive, category-specific targets; no model inference."""
from review2 import *

def compile_dataset(payload,reg,result=None,verify_files=False,rows=None,inventory=None):
    result=result if result is not None else validate(payload,reg)
    if result['blocked']:raise ValueError('Blocked absence claims cannot enter an overlay')
    if rows is None:
        if sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json')!=reg['source_manifest_sha256']:raise ValueError('Immutable source manifest changed')
        rows=manifest()['rows']
    index={r['image_id']:r for r in rows};training=[r for r in rows if optimizer_member(r)];others=[r for r in rows if not optimizer_member(r)]
    for key in ('image_id','sha256','identity_group'):
        if {r[key] for r in training}&{r[key] for r in others}:raise ValueError('Training/evaluation leakage: '+key)
    photos={p['photo_key']:p for p in reg['photos']};tasks={t['task_id']:t for t in reg['tasks']}
    if inventory is None:inventory=read(prior.RAW/'native-object-inventory.json')
    native={a['annotation_key']:a for p in inventory['photos'] for a in p['annotations']}
    output=[]
    for row in training:
        p=next((p for p in reg['photos'] if p['image_id']==row['image_id']),None)
        if p is None or p['role']!='optimization' or any(p[k]!=row[k] for k in ('sha256','identity_group')):raise ValueError('Optimization registry binding changed')
        if p.get('existing_positive_annotation_digest',digest(row['annotations']))!=digest(row['annotations']):raise ValueError('Original positive/ignored annotation lineage changed')
        if verify_files:
            prior.source_photo(p)
            for path_key,sha_key in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
                rights_path=(ROOT/p['rights'][path_key]).resolve()
                if not rights_path.is_relative_to(ROOT) or sha(rights_path)!=p['rights'][sha_key]:raise ValueError('Recorded rights evidence changed')
        negatives=[];aux=[]
        for r in result['accepted']:
            if r['image_id']!=row['image_id']:continue
            negatives.append(dict(category='chair',box=r['box'],evidence_task_id=r['task_id'],source_photo_sha256=r['sha256'],taxonomy_sha256=digest(TAXONOMY),reviewer=r['reviewer'],review_date=r['review_date'],evidence=r['evidence'],hard_negative_for_other_classes=False))
        for r in result['auxiliary_source_confirmations']:
            if r['image_id']!=row['image_id']:continue
            task=tasks[r['task_id']]
            # Preserve each native lineage; candidate hypotheses never collapse
            # category IDs or create extra independently weighted examples.
            lineages=[native[k] for k in task['source_annotation_keys']]
            if not lineages or any(not a['metadata_candidate_eligible'] for a in lineages):raise ValueError('Unusable native auxiliary annotation')
            aux.append(dict(label=task['label'],reporting_group=task['group'],box=task['box'],review_task_id=r['task_id'],lineages=lineages,Chair_negative_implied=False,training_activated=False))
        evidence=None
        if negatives:
            reviewed=[r for r in result['accepted'] if r['image_id']==row['image_id']]
            evidence={k:row[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')}
            evidence.update(category='chair',kind='negative_region',native_category={'COCO':62,'LVIS':232,'OpenImages':'/m/01mzpv'}[row['annotations'][0]['native_dataset']],
                native_annotation_digest=digest(row['annotations']),taxonomy_sha256=digest(TAXONOMY),review_covers_seating_stools=True,
                human_reviewed=True,confirmed=True,trust_basis='human_category_scope_review',
                reviewer='; '.join(sorted({r['reviewer'] for r in reviewed})),review_date=max(r['review_date'] for r in reviewed),evidence=' | '.join(r['evidence'] for r in reviewed),
                regions=[r['box'] for r in reviewed],individual_confirmations=reviewed,
                potential_unannotated_chair_regions=[q['box'] for q in payload.get('proposals',[]) if q['photo_key']==p['photo_key']]+[r['box'] for r in payload.get('reviews',[]) if r['photo_key']==p['photo_key'] and r['decision'] in ('correct_chair','uncertain','missing_annotation')])
            # Retain existing C.3/C.2 integrity gates instead of bypassing them.
            from preflight import validate_region_record
            validate_region_record(evidence,row)
        output.append(dict(image_id=row['image_id'],source=row.get('source'),source_photo_id=row.get('source_photo_id'),sha256=row['sha256'],identity_group=row['identity_group'],historical_partition=row['historical_partition'],internal_partition=row['internal_partition'],
            source_annotations=row['annotations'],source_annotation_digest=digest(row['annotations']),source_positive_boxes=[a for a in row['annotations'] if not a['ignore']],
            source_ignored_boxes=[a for a in row['annotations'] if a['ignore']],additional_protective_regions=p['protected'],
            explicit_native_negative=row.get('explicit_native_negative',{}),known_source_coverage=row.get('coverage',{}),
            reviewed_Chair_negative_regions=negatives,legacy_compatible_Chair_evidence=evidence,confirmed_named_source_objects=aux,
            potential_positive_or_unknown_regions=[q for q in payload.get('proposals',[]) if q['photo_key']==p['photo_key']],
            reviewed_uncertain_regions=[r for r in payload.get('reviews',[]) if r['photo_key']==p['photo_key'] and r['decision'] in ('correct_chair','uncertain','missing_annotation')],
            rights=p.get('rights',{}),unknown_labels_ignored=True))
    return dict(schema='phase10c5-category-masked-overlay-v2',source_manifest_sha256=reg['source_manifest_sha256'],source_inventory_sha256=reg.get('source_inventory_sha256'),registry_sha256=digest(reg),human_review_sha256=digest(payload),taxonomy_sha256=digest(TAXONOMY),
        production_categories=list(CLASSES),rows=output,optimization_photos=len(output),source_groups=len({r['identity_group'] for r in output}),
        positive_boxes=sum(len(r['source_positive_boxes']) for r in output),accepted_Chair_negatives=result['accepted_hard_negatives'],accepted_negative_evidence=[r['legacy_compatible_Chair_evidence'] for r in output if r['legacy_compatible_Chair_evidence'] is not None],
        auxiliary_policy='Named source positives only; no implicit Chair absence or generic Other output; not activated',
        sampling_policy='Fixed deterministic source-group/photo/object-hypothesis caps; no expansion sampling enabled',
        training_authorized=False,auxiliary_training_activated=False,source_annotations_modified=False)

def category_masks(query_xyxy,row,matched_positive_pairs=(),legacy_targets=None,legacy_mask=None):
    """Pure CPU coordinate contract for future loss wiring, not a model matcher.

    Unknowns remain zero. New Chair-negative targets require full containment,
    never just query-center inclusion. Existing source positives are represented
    only through an explicit future matcher result; GT and query boxes unchanged.
    """
    if (legacy_targets is None)!=(legacy_mask is None):raise ValueError('Pass both legacy targets and masks')
    target=copy.deepcopy(legacy_targets) if legacy_targets is not None else [[0]*7 for _ in query_xyxy]
    mask=copy.deepcopy(legacy_mask) if legacy_mask is not None else [[False]*7 for _ in query_xyxy]
    if len(target)!=len(query_xyxy) or len(mask)!=len(query_xyxy) or any(len(r)!=7 for r in target+mask) or any(v not in (0,1) for r in target+mask for v in r):raise ValueError('Invalid seven-category legacy targets/masks')
    for b in query_xyxy:
        if not prior.valid_box(b):raise ValueError('Unsafe model query coordinates')
    supported_categories={a['label'] for a in row['source_positive_boxes']}
    for qi,category in matched_positive_pairs:
        if type(qi) is not int or not 0<=qi<len(query_xyxy) or category not in CLASSES or category not in supported_categories:raise ValueError('Invalid or source-unsupported matcher pair')
        j=CLASSES.index(category);target[qi][j]=1;mask[qi][j]=True
    protected=[p['box'] for p in row['additional_protective_regions']]+[a['box'] for a in row['source_ignored_boxes']]
    protected += [q['box'] for q in row['potential_positive_or_unknown_regions']]+[r['box'] for r in row['reviewed_uncertain_regions']]
    for qi,b in enumerate(query_xyxy):
        if target[qi][1] or any(overlap(b,p) for p in protected):continue
        if any(r['box'][0]<=b[0]<b[2]<=r['box'][2] and r['box'][1]<=b[1]<b[3]<=r['box'][3] for r in row['reviewed_Chair_negative_regions']):mask[qi][1]=True
    return target,mask
