"""Fast action-based review: object labels do not imply furniture absence."""
from datetime import date,datetime,timezone
from collections import defaultdict
import uuid
from g5_common import *

ACTION_STATEMENT='I explicitly confirm this exact box contains no Chair, seating stool or bar stool, including ambiguous, occluded or partial seating.'
DECISIONS=('correct_chair','not_chair','missing_annotation','uncertain')

def registry():
    r=read(RAW/'review-registry.json')
    if digest(r)!=read(REPORT/'review-summary.json')['registry_sha256']:raise ValueError('Registry changed')
    if r['source_inventory_sha256']!=sha(RAW/'native-object-inventory.json') or r['source_manifest_sha256']!=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'):raise ValueError('Inputs changed')
    return r

def source_photo(photo):
    rows={r['image_id']:r for r in manifest()['rows']};row=rows.get(photo['image_id'])
    if row is None or row['sha256']!=photo['sha256'] or row['identity_group']!=photo['identity_group']:raise PermissionError('Unapproved photo identity')
    if (photo['role']=='optimization')!=optimizer_member(row):raise PermissionError('Partition changed')
    p=(ROOT/photo['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or p!= (ROOT/row['path']).resolve() or sha(p)!=photo['sha256']:raise ValueError('Source bytes/path changed')
    return p

def valid_box(b):
    import math
    return isinstance(b,list) and len(b)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in b) and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1

def validate(payload,reg):
    if payload.get('schema')!=SCHEMA or payload.get('registry_sha256')!=digest(reg) or payload.get('taxonomy_sha256')!=digest(TAXONOMY):raise ValueError('Review/schema/taxonomy binding changed')
    reviews=payload.get('reviews',[]);proposals=payload.get('proposals',[])
    if not isinstance(reviews,list) or not isinstance(proposals,list) or len(reviews)>1500 or len(proposals)>200:raise ValueError('Bounded review arrays required')
    photos={p['photo_key']:p for p in reg['photos']};tasks={t['task_id']:t for t in reg['tasks']};seen=set();accepted=[];aux=[];blocked=[]
    for p in proposals:
        if p.get('photo_key') not in photos or not valid_box(p.get('box')) or not isinstance(p.get('proposal_id'),str):raise ValueError('Invalid proposed annotation')
        if p.get('approved_for_training') is not False:raise ValueError('New/corrected positive annotations remain proposals, not activated GT')
        photo=photos[p['photo_key']]
        if any(p.get(k)!=photo[k] for k in ('image_id','sha256','identity_group','width','height','orientation')):raise ValueError('Proposed annotation photo binding changed')
    for r in reviews:
        ident=r.get('task_id')
        if ident not in tasks or ident in seen:raise ValueError('Unknown/duplicate review task')
        seen.add(ident);t=tasks[ident];p=photos[t['photo_key']]
        if r.get('photo_key')!=t['photo_key'] or any(r.get(k)!=p[k] for k in ('image_id','sha256','identity_group','width','height','orientation')):raise ValueError('Review photograph identity changed')
        if not valid_box(r.get('box')) or r.get('decision') not in DECISIONS:raise ValueError('Invalid review decision/box')
        if r.get('human_reviewed') is not True or r.get('confirmed') is not True or not r.get('reviewer','').strip() or len(r.get('evidence','').strip())<8:raise ValueError('Explicit human action/session identity required')
        date.fromisoformat(r.get('review_date',''))
        if r.get('evidence_kind') not in ('human_note','explicit_action_log'):raise ValueError('Evidence must be a human note or truthful log of an explicit action')
        if r.get('geometry_kind')=='source':
            if r['box']!=t['box']:raise ValueError('Source box silently changed')
        elif r.get('geometry_kind')!='human_corrected':raise ValueError('Geometry provenance required')
        if r.get('hard_negative'):
            reasons=[]
            if not t['eligible_for_training_review'] or p['role']!='optimization':raise PermissionError('Validation diagnostics never become training negatives')
            if r['decision']!='not_chair' or r.get('explicit_absence_statement')!=ACTION_STATEMENT or r.get('confirmed_box')!=r['box'] or r.get('review_covers_seating_stools') is not True:raise ValueError('A Not-Chair object decision alone is not Chair-absence evidence')
            if r['box']==[0,0,1,1]:reasons.append('Whole-image absence requires completeness review')
            for b in p['protected']:
                if overlap(r['box'],b['box']):reasons.append(b['reason'])
            for q in proposals:
                if q['photo_key']==p['photo_key'] and overlap(r['box'],q['box']):reasons.append('Overlapping missing/uncertain proposed object')
            for q in reviews:
                if q.get('photo_key')==p['photo_key'] and q.get('decision') in ('correct_chair','missing_annotation','uncertain') and valid_box(q.get('box')) and overlap(r['box'],q['box']):reasons.append('Overlapping reviewed Chair/missing/uncertain object')
            if reasons:blocked.append(dict(task_id=ident,reasons=sorted(set(reasons))))
            else:accepted.append(r)
        # Named source-object confirmation is independent of a Chair-negative claim.
        if r.get('source_label_confirmed'):
            if r['decision']!='not_chair' or not t['source_grounded'] or t['group']=='protected_furniture' or r['geometry_kind']!='source':raise ValueError('Auxiliary source-label confirmation lacks native grounding')
            if t['eligible_for_training_review']:aux.append(r)
    overlapping=set()
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            if a['photo_key']==b['photo_key'] and overlap(a['box'],b['box']):overlapping.update((a['task_id'],b['task_id']))
    for ident in sorted(overlapping):blocked.append(dict(task_id=ident,reasons=['Overlapping hard-negative boxes; do not double-count']))
    accepted=[r for r in accepted if r['task_id'] not in overlapping];groups={r['identity_group'] for r in accepted}
    return dict(reviewed=len(reviews),accepted_hard_negatives=len(accepted),negative_source_groups=len(groups),accepted=accepted,
        confirmed_named_source_objects=len(aux),auxiliary_source_confirmations=aux,blocked=blocked,proposed_positive_boxes=len(proposals),
        training_enabled=False,unknown_regions_ignored=True,Chair_negative_never_implies_other_furniture_background=True)

def compare_masks(evidence):
    # Reuse the unchanged C.4 CPU fixture comparison, never real model inference.
    import importlib.util
    spec=importlib.util.spec_from_file_location('g5_prior_export',ROOT/'tools/phase10c4-v1/exporter.py');m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(ROOT/'tools/phase10c4-v1'));spec.loader.exec_module(m)
    return m.mask_comparison(evidence)

def export_review(payload,reg):
    result=validate(payload,reg)
    if result['blocked']:raise ValueError('Resolve blocked hard-negative claims: '+str(result['blocked']))
    photos={p['photo_key']:p for p in reg['photos']};tasks={t['task_id']:t for t in reg['tasks']}
    for key in {r['photo_key'] for r in payload['reviews']}|{r['photo_key'] for r in payload['proposals']}:source_photo(photos[key])
    folder=RAW/'exports'/('review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10]);folder.mkdir(parents=True,exist_ok=False)
    save(folder/'human-review.json',payload);save(folder/'validation.json',result)
    byphoto=defaultdict(list)
    for r in result['accepted']:byphoto[r['photo_key']].append(r)
    ev=[];old={r['image_id']:r for r in manifest()['rows']}
    for key,rs in byphoto.items():
        p=photos[key];row=old[p['image_id']];record={k:row[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')}
        record.update(category='chair',kind='negative_region',native_category={'COCO':62,'LVIS':232,'OpenImages':'/m/01mzpv'}[row['annotations'][0]['native_dataset']],
            regions=[r['box'] for r in rs],potential_unannotated_chair_regions=[q['box'] for q in payload['proposals'] if q['photo_key']==key],native_annotation_digest=digest(row['annotations']),
            reviewer='; '.join(sorted({r['reviewer'] for r in rs})),review_date=max(r['review_date'] for r in rs),human_reviewed=True,confirmed=True,
            evidence=' | '.join(r['evidence'] for r in rs),trust_basis='human_category_scope_review',taxonomy_sha256=digest(TAXONOMY),review_covers_seating_stools=True,
            individual_confirmations=rs)
        from preflight import validate_region_record
        validate_region_record(record,row);ev.append(record)
    contrast=compare_masks(ev);save(folder/'mask-comparison.json',contrast)
    save(folder/'hard-negative-supervision.json',dict(source_manifest_sha256=reg['source_manifest_sha256'],production_categories=list(CLASSES),
        accepted_Chair_negative_evidence=ev,confirmed_auxiliary_source_objects=[dict(review=r,source_annotation=tasks[r['task_id']]) for r in result['auxiliary_source_confirmations']],
        auxiliary_training_activated=False,positive_box_proposals=payload['proposals'],original_positives_preserved=True,training_authorized=False,diagnostic_examples_not_in_training=True))
    with (folder/'SHA256SUMS').open('x') as f:
        for p in sorted(folder.iterdir()):
            if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
    return dict(export_id=folder.name,reviewed=result['reviewed'],hard_negatives=result['accepted_hard_negatives'],groups=result['negative_source_groups'],named_objects=result['confirmed_named_source_objects'],mask_changed_terms=contrast['changed_mask_terms'],training_enabled=False)
