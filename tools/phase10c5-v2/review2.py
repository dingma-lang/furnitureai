"""Additive v2 review: immutable v1 registry, explicit actions, replayable edits."""
from pathlib import Path
import sys,uuid,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c5-v1'))
import g5_review as prior
from g5_common import read,save,sha,digest,manifest,optimizer_member,CLASSES,TAXONOMY,overlap
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c5-v2'
REPORT=ROOT/'reports/phase10c5-v2'
SCHEMA='phase10c5-fast-review-v2'

def validate(payload,reg):
    if payload.get('schema')!=SCHEMA:raise ValueError('v2 schema required; explicitly import a v1 export first')
    legacy=dict(payload,schema=prior.SCHEMA);result=prior.validate(legacy,reg)
    proposal_ids=[p['proposal_id'] for p in payload.get('proposals',[])]
    if len(proposal_ids)!=len(set(proposal_ids)):raise ValueError('Duplicate proposal IDs')
    events=payload.get('history',[])
    if not isinstance(events,list) or len(events)>10000:raise ValueError('Bounded edit-history array required')
    photos={p['photo_key']:p for p in reg['photos']};tasks={t['task_id']:t for t in reg['tasks']}
    snapshots={'review':{},'proposal':{},'geometry':{}};ids=set()
    for i,e in enumerate(events):
        if not isinstance(e,dict) or type(e.get('sequence')) is not int or e['sequence']!=i+1 or not isinstance(e.get('event_id'),str) or e['event_id'] in ids:raise ValueError('Duplicate or unordered edit event')
        ids.add(e['event_id']);kind=e.get('kind');key=e.get('key')
        if kind not in snapshots or not isinstance(key,str):raise ValueError('Unknown edit-history kind/key')
        try:stamp=datetime.fromisoformat(e['at'].replace('Z','+00:00'))
        except (ValueError,KeyError,AttributeError):raise ValueError('Timestamp required for each edit')
        if stamp.tzinfo is None:raise ValueError('Edit timestamp must include timezone')
        if not isinstance(e.get('action'),str) or not e['action'].strip():raise ValueError('Explicit edit action required')
        photo=photos.get(e.get('photo_key'))
        if photo is None or e.get('image_id')!=photo['image_id'] or e.get('sha256')!=photo['sha256']:raise ValueError('Edit photo identity changed')
        if kind in ('review','geometry') and (key not in tasks or tasks[key]['photo_key']!=photo['photo_key']):raise ValueError('Edit task/partition binding changed')
        before=e.get('before');after=e.get('after')
        if before!=snapshots[kind].get(key):raise ValueError('Edit history does not replay from its prior state')
        if kind=='review' and after is not None:
            if not isinstance(after,dict) or after.get('task_id')!=key or after.get('photo_key')!=photo['photo_key']:raise ValueError('Review event identity changed')
            mini=dict(photos=[photo],tasks=[tasks[key]])
            prior.validate(dict(legacy,registry_sha256=digest(mini),reviews=[after],proposals=[]),mini)
        if kind=='proposal' and after is not None:
            if not isinstance(after,dict) or after.get('proposal_id')!=key or after.get('photo_key')!=photo['photo_key']:raise ValueError('Proposal event identity changed')
            mini=dict(photos=[photo],tasks=[])
            prior.validate(dict(legacy,registry_sha256=digest(mini),reviews=[],proposals=[after]),mini)
        if kind=='geometry' and after is not None and not prior.valid_box(after):raise ValueError('Unsafe draft geometry')
        if after is None:snapshots[kind].pop(key,None)
        else:snapshots[kind][key]=after
    if snapshots['review']!={r['task_id']:r for r in payload.get('reviews',[])} or snapshots['proposal']!={q['proposal_id']:q for q in payload.get('proposals',[])}:raise ValueError('Final decisions/proposals disagree with explicit edit history')
    if snapshots['geometry']!=payload.get('draft_geometry',{}):raise ValueError('Draft geometry disagrees with edit history')
    # Prevent a correction's old absence approval surviving a later box edit.
    for r in payload.get('reviews',[]):
        if r.get('hard_negative') and r['box']!=snapshots['geometry'].get(r['task_id'],tasks[r['task_id']]['box']):raise ValueError('Hard-negative confirmation does not match current geometry')
    result['history_events']=len(events)
    return result

def export(payload,reg):
    result=validate(payload,reg)
    if result['blocked']:raise ValueError('Resolve blocked claims before export: '+str(result['blocked']))
    from dataset2 import compile_dataset
    dataset=compile_dataset(payload,reg,result,verify_files=True)
    folder=RAW/'exports'/('review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10]);folder.mkdir(parents=True,exist_ok=False)
    save(folder/'human-review.json',payload);save(folder/'validation.json',result);save(folder/'supervision-manifest.json',dataset)
    with (folder/'SHA256SUMS').open('x') as f:
        for p in sorted(folder.iterdir()):
            if p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
    save(folder/'COMPLETE.json',dict(schema=SCHEMA,sha256sums=sha(folder/'SHA256SUMS'),training_authorized=False))
    return dict(export_id=folder.name,reviewed=result['reviewed'],hard_negatives=result['accepted_hard_negatives'],groups=result['negative_source_groups'],named_objects=result['confirmed_named_source_objects'],mask_changed_terms=None,training_enabled=False)
