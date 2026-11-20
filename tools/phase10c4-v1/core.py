"""Pixel-edge geometry and human-evidence validation; no model imports."""
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c4-v1'
REPORT=ROOT/'reports/phase10c4-v1'
sys.path.insert(0,str(ROOT/'tools/phase10c3-v1'))
from preflight import TAXONOMY,RAW3,read,save,sha,digest,manifest,optimizer_member

SCHEMA='phase10c4-regions-v1'
STATUSES=('draft','confirmed_negative','uncertain','rejected')

def display_size(w,h,o):
    if min(w,h)<=0 or o not in range(1,9):raise ValueError('Invalid dimensions/orientation')
    return (h,w) if o>=5 else (w,h)

def transform_point(x,y,w,h,o,inverse=False):
    if inverse:
        return {1:(x,y),2:(w-x,y),3:(w-x,h-y),4:(x,h-y),5:(y,x),6:(y,h-x),7:(w-y,h-x),8:(w-y,x)}[o]
    return {1:(x,y),2:(w-x,y),3:(w-x,h-y),4:(x,h-y),5:(y,x),6:(h-y,x),7:(h-y,w-x),8:(y,w-x)}[o]

def transform_box(b,w,h,o,inverse=False):
    pts=[transform_point(x,y,w,h,o,inverse) for x,y in ((b[0],b[1]),(b[0],b[3]),(b[2],b[1]),(b[2],b[3]))]
    return [min(x for x,y in pts),min(y for x,y in pts),max(x for x,y in pts),max(y for x,y in pts)]

def overlaps(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])

def pixel_box(b,w,h):
    if not isinstance(b,list) or len(b)!=4 or any(type(x) is not int for x in b) or not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('Integer original-image pixel-edge XYXY coordinates required')
    return b

def normalized(b,w,h):return [b[0]/w,b[1]/h,b[2]/w,b[3]/h]

def region_identity(r,p):
    for key in ('image_id','sha256','identity_group','width','height','orientation','native_annotation_digest'):
        if r.get(key)!=p[key]:raise ValueError('Photo identity/dimensions/orientation/lineage mismatch: '+key)

def validate(payload,registry):
    """Return accepted rectangles separately; drafts/uncertain are never negatives."""
    if payload.get('schema')!=SCHEMA or payload.get('registry_sha256')!=digest(registry) or payload.get('taxonomy_sha256')!=digest(TAXONOMY):raise ValueError('Registry/schema/taxonomy mismatch')
    regions=payload.get('regions',[])
    if not isinstance(regions,list) or len(regions)>600:raise ValueError('At most600 local rectangles')
    photos={p['key']:p for p in registry['photos']};seen=set();accepted=[];issues=[];valid=[]
    from datetime import date
    for r in regions:
        if not isinstance(r,dict) or not isinstance(r.get('region_id'),str) or not r['region_id'] or len(r['region_id'])>100 or r['region_id'] in seen:raise ValueError('Unique rectangle IDs required')
        seen.add(r['region_id']);p=photos.get(r.get('photo_key'))
        if p is None:raise PermissionError('Not an eligible optimization photograph')
        region_identity(r,p);pixel_box(r.get('box_pixels'),p['width'],p['height'])
        if r.get('taxonomy_sha256')!=digest(TAXONOMY) or r.get('historical_partition')!='train' or r.get('internal_partition')!='optimization':raise PermissionError('Per-rectangle taxonomy and optimization partition required')
        if r.get('kind') not in ('negative','potential_positive') or r.get('status') not in STATUSES:raise ValueError('Unknown rectangle type/status')
        if type(r.get('revision')) is not int or r['revision']<0:raise ValueError('Rectangle revision required')
        if r['kind']=='potential_positive' and r['status']=='confirmed_negative':raise ValueError('Potential seating cannot be a negative')
        if r['status']=='confirmed_negative':
            if r['box_pixels']==[0,0,p['width'],p['height']]:raise ValueError('Whole-image absence needs separate completeness review, not this regional workflow')
            if not(r.get('human_reviewed') is True and r.get('confirmed') is True and r.get('review_covers_seating_stools') is True and r.get('no_chair_stool_or_ambiguous_part') is True and r.get('confirmed_revision')==r['revision'] and r.get('confirmed_box_pixels')==r['box_pixels']):raise ValueError('Explicit current-geometry human and taxonomy confirmation required')
            if not isinstance(r.get('reviewer'),str) or not r['reviewer'].strip() or not isinstance(r.get('evidence'),str) or len(r['evidence'].strip())<8:raise ValueError('Reviewer and specific region evidence required')
            date.fromisoformat(r.get('review_date',''))
        valid.append((r,p))
    for r,p in valid:
        if r['status']!='confirmed_negative':continue
        b=r['box_pixels'];blocked=[]
        for obj in p['restricted']:
            if overlaps(b,obj['box_pixels']):blocked.append(obj['reason'])
        for other,_ in valid:
            if other['photo_key']==r['photo_key'] and other['region_id']!=r['region_id'] and other['status']!='rejected':
                if overlaps(b,other['box_pixels']):blocked.append('overlapping potential seating/uncertain/draft rectangle' if other['kind']=='potential_positive' or other['status']=='uncertain' else 'overlapping negative rectangle')
        if blocked:
            issues.append(dict(region_id=r['region_id'],blocked_by=sorted(set(blocked))));continue
        accepted.append(dict(r,box_normalized=normalized(b,p['width'],p['height']),confirmation_geometry_sha256=digest(dict(box_pixels=b,sha256=p['sha256'],taxonomy_sha256=digest(TAXONOMY)))))
    # A structurally valid file may include blocked attempts; none enter the mask.
    clarifications=payload.get('clarifications',[]);available={c['item_id']:c for c in registry['clarifications']};seen_clar=set()
    for c in clarifications:
        if c.get('item_id') not in available or c['item_id'] in seen_clar:raise ValueError('Unknown/duplicate clarification')
        seen_clar.add(c['item_id'])
        if c.get('action') not in ('clarified','deferred') or c.get('confirmed') is not True or not c.get('reviewer','').strip() or len(c.get('evidence','').strip())<8:raise ValueError('Clarification needs explicit response and reviewer')
        date.fromisoformat(c.get('review_date',''))
    groups={r['identity_group'] for r in accepted}
    return dict(confirmed_negative_rectangles=len(accepted),distinct_source_groups=len(groups),accepted=accepted,blocked=issues,
        draft_or_unaccepted_rectangles=len(regions)-len(accepted),minimum_region_group_gate=len(accepted)>=6 and len(groups)>=3,
        clarification_responses=len(clarifications),unresolved_clarifications=24-len([c for c in clarifications if c['action']=='clarified']),
        training_authorized=False,rest_of_each_image_ignored=True)

def registry():
    r=read(RAW/'registry.json')
    if r['source_manifest_sha256']!=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json') or r['taxonomy_sha256']!=digest(TAXONOMY):raise ValueError('Frozen registry source changed')
    if digest(r)!=read(REPORT/'summary.json')['registry_sha256']:raise ValueError('Registry content changed')
    return r

def photo_path(photo):
    rows={r['image_id']:r for r in manifest()['rows'] if optimizer_member(r)};row=rows.get(photo['image_id'])
    if row is None or row['sha256']!=photo['sha256'] or row['identity_group']!=photo['identity_group']:raise PermissionError('Training-only source identity required')
    p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=photo['sha256']:raise ValueError('Source path/hash changed')
    return p
