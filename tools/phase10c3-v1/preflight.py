"""CPU-only human-review/taxonomy supervision preflight. No model worker."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW3=ROOT/'data/open-images/preparation/phase10c3-v1'
REPORT3=ROOT/'reports/phase10c3-v1'
sys.path.insert(0,str(ROOT/'tools/phase10c2-v1'))
from n2 import read,save,sha,digest,manifest,optimizer_member,RAW as RAW2,REPORT as REPORT2
from review import validate_submission

TAXONOMY=dict(schema='phase10c3-chair-taxonomy-v1',user_confirmed=True,
    confirmation='User explicitly chose inclusion of ordinary seating stools/bar stools, with footstools/ottomans/step stools separate',
    project_chair_includes=['ordinary_chairs','ordinary_seating_stools','bar_stools'],
    project_chair_excludes=['footstools','ottomans','step_stools'],
    native_labels_not_rewritten=True,old_native_positive_mapping_preserved=True,
    stool_positive_annotations_automatically_activated=False,
    seating_stools_and_ambiguous_seating_protected_from_negative_targets=True,
    old_scope_form_did_not_explicitly_confirm_stool_coverage=True)

def semantic_flags(d):
    """Conservative requests for clarification, never automatic label correction."""
    text=d['evidence'].lower();flags=[];decision=d['decision']
    if d['kind']=='training_scope_review':
        if 'detection' in text:flags.append('Clarify native ground-truth completeness versus detector output')
        if len(text.split())<4 or 'classic' in text:flags.append('Need precise missing/ambiguous-object explanation before accepting any future scope')
        return flags
    if decision=='background_non_furniture' and any(s in text for s in ('child seating chair','part of the bed')):
        flags.append('Evidence mentions furniture content while decision is background; preserve and clarify')
    if decision=='other_furniture' and any(s in text for s in ('plant','people','pencil holder','a leg','chair back','a climb')):
        flags.append('Evidence may describe non-furniture, a person, accessory or Chair part; clarify semantic choice')
    if decision=='genuine_chair' and any(s in text for s in ('table set','whole table','two chairs','partially','chair back','most of them')):
        flags.append('Chair content does not verify a single whole-object detection or box correctness')
    if 'detected as a table' in text:flags.append('Reference category differs from saved Chair prediction')
    return flags

def validate_region_record(record,row):
    """Additional mandatory taxonomy guard before the unchanged C.2 validator."""
    from trusted_loss import evidence_valid
    if record.get('taxonomy_sha256')!=digest(TAXONOMY) or record.get('review_covers_seating_stools') is not True:
        raise PermissionError('Human region review must explicitly cover the confirmed Chair+seating-stool taxonomy')
    if record.get('kind')!='negative_region':raise PermissionError('All submitted whole-image scope decisions declined completeness')
    evidence_valid(record,row)
    return True

def paths():
    return dict(submission=RAW2/'submissions/manual-v1/human-submission.json',queue=RAW2/'review-queue.json',
        lvis=ROOT/'data/lvis/metadata/phase7-v1/native-category-definitions.json',
        coco=ROOT/'data/coco/metadata/phase6b-v1/category-and-license-definitions.json',
        oi_classes=ROOT/'data/open-images/metadata/phase5-v1/oidv7-class-descriptions-boxable.csv',
        oi_hierarchy=ROOT/'data/open-images/metadata/phase5-v1/bbox_labels_600_hierarchy.json')

def run():
    if RAW3.exists() or REPORT3.exists():raise FileExistsError('New version required; no historical overwrite')
    RAW3.mkdir(parents=True);REPORT3.mkdir(parents=True)
    sources=paths();submitted=read(sources['submission']);queue=read(sources['queue']);prepared=manifest();rows={r['image_id']:r for r in prepared['rows']}
    old_index=read(REPORT2/'candidate-manifest-index.json')
    if sha(sources['queue'])!=old_index['queue_sha256']:raise ValueError('Frozen review queue changed')
    result=validate_submission(submitted,queue,prepared['rows']);byitem={r['item_id']:r for r in queue['records']}
    if result['reviewed_items']!=65 or result['pending_items']!=0:raise ValueError('This preflight pins the completed65-item export; partial submissions need a separately counted version')
    # Exact byte comparison to recorded original export, not its reserialized copy.
    matches=list(Path('/mnt/c/Users').glob('*/Downloads/phase10c2-review-2026-11-20T07-29-56-191Z.json'))
    if len(matches)!=1:raise ValueError('Exactly one original Windows export expected for this version')
    source_export=matches[0]
    if sha(source_export)!='ef5d9ef5df383afcbdf9605e27c1896d696498a0598a284585083d15f21ff8cf' or read(source_export)!=submitted:raise ValueError('Human export/source differs')
    recovered=read(ROOT/'data/open-images/experiments/phase10c-recovery-v1/post-nms-predictions.json');observations=[]
    for d in submitted['records']:
        q=byitem[d['item_id']];r=rows[d['image_id']];p=ROOT/r['path']
        if sha(p)!=r['sha256']:raise ValueError('Submitted photograph identity changed')
        if d['kind']=='prediction_diagnostic':
            pred=recovered[d['image_id']][q['prediction_index']]
            if pred['label']!='chair' or pred['box']!=q['box'] or pred['score']!=q['score']:raise ValueError('Reviewed box differs from saved prediction')
        observations.append(dict(item_id=d['item_id'],kind=d['kind'],image_id=d['image_id'],source_photo_sha256_verified=True,
            partition=r['internal_partition'],decision=d['decision'],evidence=d['evidence'],structural_status='PASS',
            clarification_flags=semantic_flags(d),training_eligible=optimizer_member(r),
            negative_supervision_accepted=False,original_decision_changed=False))
    lvis=read(sources['lvis'])['categories'];wanted={232,1018,468,738,1013,558,19,358,464,889,710}
    native=[c for c in lvis if c['id'] in wanted]
    if {c['id']:c['name'] for c in native}.get(1018)!='stool':raise ValueError('LVIS taxonomy changed')
    import csv
    with sources['oi_classes'].open() as f:oi=dict(csv.reader(f))
    if oi.get('/m/01mzpv')!='Chair' or oi.get('/m/0fqt361')!='Stool':raise ValueError('Open Images taxonomy changed')
    found={}
    def walk(node,path):
        here=path+[node.get('LabelName')]
        if node.get('LabelName') in ('/m/01mzpv','/m/0fqt361'):found.setdefault(node['LabelName'],[]).append(here)
        for key in ('Subcategory','Part'):
            for child in node.get(key,[]):walk(child,here)
    walk(read(sources['oi_hierarchy']),[])
    taxonomy=dict(policy=TAXONOMY,policy_sha256=digest(TAXONOMY),native_lvis_definitions=native,OpenImages_paths=found,
        COCO_chair=[c for c in read(sources['coco'])['categories'] if c['id']==62],
        caveat='COCO category metadata does not establish an explicit stool boundary; never infer stool absence from missing native Chair boxes',
        user_policy_does_not_relabel_native_annotations=True)
    save(RAW3/'taxonomy.json',taxonomy)
    scope_decisions={d['image_id']:d for d in submitted['records'] if d['kind']=='training_scope_review'}
    supervision=[]
    for r in prepared['rows']:
        if not optimizer_member(r):continue
        d=scope_decisions.get(r['image_id'])
        supervision.append(dict(image_id=r['image_id'],source=r['source'],source_photo_id=r['source_photo_id'],sha256=r['sha256'],identity_group=r['identity_group'],
            historical_partition=r['historical_partition'],internal_partition=r['internal_partition'],
            native_annotations_sha256=digest(r['annotations']),existing_positive_annotations_preserved=True,
            existing_explicit_source_negatives_preserved=r['explicit_native_negative'],
            chair_negative_regions=[],new_trusted_chair_negative_targets=False,
            chair_negative_unknown_regions=[[0.,0.,1.,1.]],
            unknown_mask_scope='New Chair-negative targets only; never erases existing matched native positives',
            human_scope_decision=d['decision'] if d else 'unreviewed',
            reason='Human declined whole-image completeness; no regional absence coordinates provided' if d else 'No accepted human region/completeness evidence'))
    if len(supervision)!=59 or result['trusted_records']:raise ValueError('Unexpected eligible evidence/data drift')
    save(RAW3/'supervision-manifest-v1.json',dict(schema='phase10c3-additive-supervision-v1',source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        taxonomy_sha256=digest(TAXONOMY),human_submission_sha256=sha(sources['submission']),training_enabled=False,source_annotations_modified=False,
        rows=supervision,accepted_negative_evidence=[],validation_or_reserved_images_included=False))
    save(RAW3/'review-validation.json',dict(reviewed_items=result['reviewed_items'],pending_items=result['pending_items'],records=observations,
        unique_photo_hashes_checked=len({d['image_id'] for d in submitted['records']}),prediction_boxes_checked=56,scope_decisions_preserved=9,
        trusted_negative_regions=0,human_labels_not_silently_corrected=True,diagnostic_labels_not_box_correctness_or_accuracy=True))
    template=[]
    for ident,d in scope_decisions.items():
        r=rows[ident]
        template.append(dict({k:r[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')},
            category='chair',native_category=next(a['native_id'] for a in r['annotations'] if a['label']=='chair'),kind='negative_region',
            regions=[],potential_unannotated_chair_regions=[],native_annotation_digest=digest(r['annotations']),
            taxonomy_sha256=digest(TAXONOMY),review_covers_seating_stools=False,human_reviewed=False,confirmed=False,reviewer='',review_date='',evidence='',
            trust_basis='human_category_scope_review',template_only=True))
    save(RAW3/'region-evidence-template.json',dict(records=template,accepted_records=0,coordinates_invented=False,
        instructions='Review exact optimization image; mark normalized XYXY rectangles that contain no Chair/seating-stool or ambiguous part. Mark potential positives touching any proposed region; confirm taxonomy/reviewer/date/evidence. Empty templates are never trusted.'))
    summary=dict(status='BLOCKED_NO_TRUSTED_NEGATIVE_REGIONS',submitted=65,structural_pass=65,source_photos_byte_verified=17,prediction_boxes_verified=56,
        semantic_clarification_items=sum(bool(r['clarification_flags']) for r in observations),whole_scope_declined=9,accepted_training_negative_images=0,accepted_training_negative_regions=0,
        training_rows_preserved=59,training_groups_preserved=56,unreviewed_optimization_images=50,training_enabled=False,project_Chair_includes_stools_user_confirmed=True,
        policy_sha256=digest(TAXONOMY),supervision_manifest_path=str((RAW3/'supervision-manifest-v1.json').relative_to(ROOT)),supervision_manifest_sha256=sha(RAW3/'supervision-manifest-v1.json'),
        all_new_chair_negative_masks_unknown=True,meaningful_AB_contrast=False,GPU_forwards=0,optimizer_updates=0)
    save(REPORT3/'summary.json',summary)
    print(summary)

if __name__=='__main__':run()
