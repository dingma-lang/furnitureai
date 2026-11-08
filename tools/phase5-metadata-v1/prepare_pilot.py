"""Prepare an acquisition proposal from local metadata only, never a verified dataset.

No photographs, model scoring, fabricated image hashes or human flags. Exact native
classes only. Unknown indoor/capture/rights facts stay unknown after deterministic ranking.
"""
import argparse
import base64
import collections as C
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase5-v1'))
import detection_prep as d
from metadata_dictionary import read_dictionary, class_mapping

META=ROOT/'data/open-images/metadata/phase5-v1'
SEED='phase5-openimages-metadata-pilot-v1'
INDOOR=re.compile(r'\b(living\s*room|dining\s*room|bed\s*room|bedroom|kitchen|interior|indoors?|office|lounge|hotel\s*room|sitting\s*room)\b',re.I)
OUTDOOR=re.compile(r'\b(outdoors?|garden|patio|beach|picnic|camping|pavement|sidewalk|street|park|yacht|boat)\b',re.I)
INDOOR_OBJECTS={'Curtain','Ceiling fan','Fireplace','Refrigerator','Oven','Kitchen appliance','Kitchen & dining room table','Pillow','Bathtub','Sink'}
OUTDOOR_OBJECTS={'Tree','Palm tree','Boat','Sea','Snowmobile','Bus','Car','Bicycle','Motorcycle','Surfboard'}


def normalized_url(value):
    u=urlparse(value)
    if u.scheme not in ('http','https') or not u.hostname:raise ValueError('Invalid source URL')
    return 'https://'+u.hostname.lower()+u.path.rstrip('/')


def photo_identity(row):
    u=normalized_url(row['OriginalLandingURL'])
    match=re.search(r'^https://(?:www\.)?flickr.com/photos/[^/]+/(\d+)$',u)
    if not match:raise ValueError('Unsupported canonical source-photo identity')
    return 'flickr-photo:'+match[1]


def ordinary(g):
    return not g['ignore'] and all(g['flags'][k] in (0,1) for k in ('IsOccluded','IsTruncated','IsInside'))


def pilot_box(g):
    b=g['box'];return ordinary(g) and b[2]-b[0]>=.05 and b[3]-b[1]>=.05 and d.area(b)>=.005


class Union:
    def __init__(self,ids):self.parent={i:i for i in ids}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def grouped(candidates):
    uf=Union(r['image_id'] for r in candidates);owners={}
    for r in candidates:
        for key in ('source_identity','metadata_original_md5','photographer_identity'):
            value=r[key];tag=(key,value)
            if tag in owners:uf.union(r['image_id'],owners[tag])
            owners[tag]=r['image_id']
    members=C.defaultdict(list)
    for r in candidates:members[uf.find(r['image_id'])].append(r['image_id'])
    for r in candidates:
        ids=sorted(members[uf.find(r['image_id'])]);r['identity_group']='oi-known-family-'+hashlib.sha256('|'.join(ids).encode()).hexdigest()[:24]
        r['known_group_size']=len(ids)
    return candidates


def choose(candidates,seed=SEED):
    used={key:set() for key in ('image_id','source_identity','metadata_original_md5','identity_group')};chosen=[]
    panel_counts={part:C.Counter() for part in ('train','development','frozen_evaluation')}
    schedule=['frozen_evaluation']*60+['train','train','development']*30
    for part in schedule:
        source='test' if part=='frozen_evaluation' else 'validation'
        pool=[r for r in candidates if r['source_partition']==source and r['indoor_evidence_rank']>=1]
        counts=panel_counts[part]
        support={c:sum(r['eligible_box_counts'].get(c,0)>0 for r in pool) for c in d.CLASSES}
        order=sorted(d.CLASSES,key=lambda c:(support[c],c))
        def rank(r):
            return (-r['indoor_evidence_rank'],-len(r['eligible_box_counts']),-r['complete_visible_boxes'],hashlib.sha256((seed+'|'+r['image_id']).encode()).hexdigest())
        pool.sort(key=rank)
        remaining=[r for r in pool if not any(r[k] in used[k] for k in used)]
        if not remaining:continue
        viable=[c for c in order if any(r['eligible_box_counts'].get(c,0) for r in remaining)]
        if not viable:continue
        anchor=min(viable,key=lambda c:(counts[c],order.index(c)))
        r=next(r for r in remaining if r['eligible_box_counts'].get(anchor,0))
        r=dict(r,proposed_split=part,selection_anchor=anchor,
               image_url=f"https://open-images-dataset.s3.amazonaws.com/{source}/{r['image_id']}.jpg",
               acquisition_status='proposed_not_downloaded',image_sha256=None,
               image_quality_status='unverified_without_photograph',indoor_verified=None,
               orientation_verified=False,license_verified=False,
               independent_scene_status='known source/photographer/MD5 groups isolated; unknown physical identity unverified',
               official_abo_hash_overlap_check='pending acquired image SHA/pixel hashes; no heldout photographs opened')
        chosen.append(r)
        for c,num in r['eligible_box_counts'].items():counts[c]+=num>0
        for k in used:used[k].add(r[k])
    assert_proposal_isolation(chosen)
    summary={}
    for part in ('train','development','frozen_evaluation'):
        rows=[r for r in chosen if r['proposed_split']==part]
        summary[part]=dict(images=len(rows),known_groups=len({r['identity_group'] for r in rows}),
             category_images={c:sum(r['eligible_box_counts'].get(c,0)>0 for r in rows) for c in d.CLASSES},
             category_eligible_boxes={c:sum(r['eligible_box_counts'].get(c,0) for r in rows) for c in d.CLASSES},
             category_native_boxes={c:sum(r['native_box_counts'].get(c,0) for r in rows) for c in d.CLASSES},
             category_ordinary_boxes={c:sum(r['ordinary_box_counts'].get(c,0) for r in rows) for c in d.CLASSES},
             indoor_title_supported=sum(bool(r['indoor_title_matches']) for r in rows),
             indoor_context_supported=sum(bool(r['indoor_context_objects']) for r in rows))
    return chosen,summary


def assert_proposal_isolation(rows):
    seen={k:set() for k in ('image_id','source_identity','metadata_original_md5','identity_group')}
    reserved=d.reserved_identities()
    for r in rows:
        if r['proposed_split'] not in ('train','development','frozen_evaluation'):raise ValueError('Invalid partition')
        if (r['proposed_split']=='frozen_evaluation')!=(r['source_partition']=='test'):raise ValueError('Source partition leakage')
        if r.get('image_sha256') is not None or r.get('license_verified') is not False or r.get('indoor_verified') is not None:raise ValueError('Metadata proposal cannot claim image verification')
        for k,values in seen.items():
            if not r.get(k) or r[k] in values or r[k] in reserved:raise ValueError('Duplicate/reserved proposal identity: '+k)
            values.add(r[k])


def build(meta,out):
    if out.exists():raise FileExistsError('Fresh proposal output required')
    ledger=d.read(meta/'transfer-ledger.json')
    if not ledger['complete']:raise ValueError('Acquisition incomplete')
    for name,rec in ledger['objects'].items():
        if d.sha(meta/name)!=rec['sha256'] or (meta/name).stat().st_size!=rec['bytes']:raise ValueError('Source metadata changed')
    audited=d.read(meta/'audit-v1/furniture-candidates.json')
    names=read_dictionary(meta/'oidv7-class-descriptions-boxable.csv')
    mapping=class_mapping(meta/'oidv7-class-descriptions-boxable.csv')
    all_records=[];all_source_identities=[];counts={};excluded=C.Counter();context={};subtype_counts={}
    related={mid:name for mid,name in names.items() if name in {'Coffee table','Kitchen & dining room table','Sofa bed','Studio couch','Loveseat','Infant bed','Filing cabinet','Bathroom cabinet','Dog bed','Cat furniture'}}
    for part in ('validation','test'):
        annotations=audited[part]['annotations'];byid=C.defaultdict(list)
        for r in annotations:byid[r['image_id']].append(r)
        labels={(r['image_id'],r['label']):r['value'] for r in audited[part]['image_label_coverage']}
        infos={}
        with (meta/(part+'-images-with-rotation.csv')).open(newline='') as f:
            for r in csv.DictReader(f):
                # Group using the FULL source information tables, including excluded
                # photos that could bridge two authors through a shared original asset.
                try:
                    md5=base64.b64decode(r['OriginalMD5'],validate=True)
                    if len(md5)!=16:raise ValueError('Invalid source MD5')
                    all_source_identities.append(dict(image_id=r['ImageID'],source_identity=photo_identity(r),photographer_identity=normalized_url(r['AuthorProfileURL']),metadata_original_md5=r['OriginalMD5']))
                except (ValueError,TypeError):pass
                if r['ImageID'] in byid:infos[r['ImageID']]=r
        contexts=C.defaultdict(set);sub_counts=C.defaultdict(lambda:dict(images=set(),boxes=0))
        with (meta/(part+'-annotations-bbox.csv')).open(newline='') as f:
            for r in csv.DictReader(f):
                name=names.get(r['LabelName'])
                if r['ImageID'] in byid and int(r['IsDepiction'])==0 and name in INDOOR_OBJECTS|OUTDOOR_OBJECTS:contexts[r['ImageID']].add(name)
                if r['LabelName'] in related:
                    sc=sub_counts[name];sc['images'].add(r['ImageID']);sc['boxes']+=1
        subtype_counts[part]={k:dict(images=len(v['images']),boxes=v['boxes']) for k,v in sorted(sub_counts.items())}
        eligible_byclass=C.defaultdict(list);metadata_claims={}
        for ident,anns in sorted(byid.items()):
            info=infos[ident];reason=[]
            try:
                identity=photo_identity(info);author=normalized_url(info['AuthorProfileURL'])
                raw_md5=base64.b64decode(info['OriginalMD5'],validate=True)
                if len(raw_md5)!=16:raise ValueError('Invalid original MD5')
                orig_size=int(info['OriginalSize'])
                if orig_size<=0:raise ValueError('Invalid original size')
            except (ValueError,TypeError):reason.append('invalid_source_identity_or_hash_or_size')
            if info['License']!='https://creativecommons.org/licenses/by/2.0/':reason.append('unsupported_or_missing_metadata_license')
            try:
                rotation=float(info['Rotation'])
                if not math.isfinite(rotation) or rotation!=0:reason.append('nonzero_rotation_deferred')
            except ValueError:reason.append('unknown_rotation_deferred')
            title_matches=INDOOR.findall(info['Title']);outside=OUTDOOR.findall(info['Title'])
            if outside:reason.append('outdoor_or_ambiguous_title')
            if any({g['label'],h['label']} in ({'sofa','bed'},{'table','desk'}) and d.iou(g['box'],h['box'])>=.98 for j,g in enumerate(anns) for h in anns[j+1:]):reason.append('near_coincident_sofa_bed_or_table_desk_annotations')
            usable=[g for g in anns if pilot_box(g) and labels.get((ident,g['label']))==1]
            if not usable:reason.append('no_sufficiently_large_ordinary_box')
            for why in reason:excluded[part+':'+why]+=1
            if reason:continue
            indoor=sorted(contexts[ident]&INDOOR_OBJECTS);outdoor=sorted(contexts[ident]&OUTDOOR_OBJECTS)
            if outdoor and not title_matches and not indoor:
                excluded[part+':outdoor_context_without_indoor_evidence']+=1;continue
            native=C.Counter(g['label'] for g in anns);ordinary_counts=C.Counter(g['label'] for g in anns if ordinary(g));eligible=C.Counter(g['label'] for g in usable)
            row=dict(image_id=ident,source_partition=part,source_identity=identity,photographer_identity=author,
                 metadata_original_md5=info['OriginalMD5'],metadata_original_size=orig_size,
                 source_original_url=info['OriginalURL'],source_landing_url=info['OriginalLandingURL'],author=info['Author'],author_profile_url=info['AuthorProfileURL'],title=info['Title'],license_url=info['License'],licensing_status='historical CC BY 2.0 metadata claim; current source verification pending',
                 metadata_rotation_degrees=0,native_box_counts=dict(native),ordinary_box_counts=dict(ordinary_counts),eligible_box_counts=dict(eligible),
                 annotations=anns,verified_classes=sorted({c for (i,c),v in labels.items() if i==ident}),
                 label_source='official human Open Images native bbox + positive image label',
                 label_confidence='published human annotation; taxonomy and scene suitability caveats remain',
                 complete_visible_boxes=sum(g['flags']['IsOccluded']==0 and g['flags']['IsTruncated']==0 for g in usable),
                 indoor_title_matches=title_matches,indoor_context_objects=indoor,outdoor_context_objects=outdoor,
                 indoor_evidence_rank=3 if title_matches else (2 if indoor else (1 if len(eligible)>=2 else 0)))
            all_records.append(row)
            for g in usable:eligible_byclass[g['label']].append(g)
        counts[part]={c:dict(eligible_images=len({g['image_id'] for g in eligible_byclass[c]}),eligible_boxes=len(eligible_byclass[c])) for c in d.CLASSES}
    full_groups=grouped(all_source_identities)
    identity_lookup={r['image_id']:r for r in full_groups}
    for r in all_records:
        if r['image_id'] not in identity_lookup:raise ValueError('Eligible source identity missing from full grouping')
        r['identity_group']=identity_lookup[r['image_id']]['identity_group'];r['known_group_size']=identity_lookup[r['image_id']]['known_group_size']
    selected,summary=choose(all_records)
    selected_ids={r['image_id'] for r in selected}
    known_groups=C.defaultdict(list)
    for row in all_records:known_groups[row['identity_group']].append(row['image_id'])
    result=dict(schema='phase5-metadata-acquisition-proposal-v1',seed=SEED,
         selection_implementation_sha256=d.sha(__file__),dictionary_parser_sha256=d.sha(Path(__file__).parent/'metadata_dictionary.py'),source_metadata_sha256={n:r['sha256'] for n,r in ledger['objects'].items()},native_mapping=mapping,
         image_count=len(selected),images=selected,summary=summary,target=dict(train=60,development=30,frozen_evaluation=60),
         selection_policy='Frozen panel first, then two training slots / one development slot interleaved; rarity-balanced coverage within each panel; indoor metadata evidence required; one image per full known group',proposal_only=True,final_dataset_splits=False,source_hashes_are_original_not_cvdf_image_sha256=True,
         no_images_downloaded=True,no_indoor_or_rights_confirmation_fabricated=True,
         eligibility_policy='Exact native labels; source identity/MD5/author present; historical CC BY 2.0; known zero rotation; ordinary box with normalized sides >=.05 and area >=.005; no outdoor title; no outdoor-only context; exclude near-coincident sofa/bed and table/desk annotations.',
         limitations=['Indoor scores are metadata heuristics, not image verification','OriginalMD5 refers to original source, not CVDF resized object','Photo, photographer and original-MD5 groups are isolated; physical furniture identity and near-duplicate captures across different authors remain unknown','Actual image hashes and ABO cross-dataset pixel/hash overlap checks require approved photographs','Current individual source licence and orientation verification remain pending','No taxonomy child labels substituted; native categories do not imply complete seven-class image annotation'])
    duplicates={k:dict(repeated_values=sum(n>1 for n in C.Counter(r[k] for r in all_source_identities).values()),excess_records=sum(n-1 for n in C.Counter(r[k] for r in all_source_identities).values() if n>1)) for k in ('source_identity','metadata_original_md5','photographer_identity')}
    room_counts={part:{c:dict(images=sum(r['source_partition']==part and r['indoor_evidence_rank']>=1 and r['eligible_box_counts'].get(c,0)>0 for r in all_records),boxes=sum(r['eligible_box_counts'].get(c,0) for r in all_records if r['source_partition']==part and r['indoor_evidence_rank']>=1)) for c in d.CLASSES} for part in ('validation','test')}
    report=dict(room_evidence_eligible_counts=room_counts,full_source_identity_records=len(all_source_identities),full_source_duplicate_evidence=duplicates,grouping_basis='All valid source-photo/photographer/original-MD5 records from both full image information tables, including non-furniture and excluded photos',eligible_counts=counts,eligible_unique_images=len(all_records),eligible_known_groups=len(known_groups),metadata_exclusion_counts=dict(excluded),
         candidate_known_multiphoto_groups=sum(len(v)>1 for v in known_groups.values()),
         related_native_classes_not_substituted=subtype_counts,selected_summary=summary,
         feasibility=dict(metadata_only_150=len(selected)==150,verified_real_room_pilot='not established without photograph and per-image rights/orientation checks'))
    out.mkdir(parents=True)
    d.write_new(out/'proposed-image-acquisition.json',result);d.write_new(out/'eligibility-report.json',report)
    with (out/'proposed-images.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','source_partition','proposed_split','selection_anchor','identity_group','source_identity','author','license_url','image_url','source_landing_url','metadata_original_size','indoor_evidence_rank'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in selected)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--metadata',type=Path,default=META);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();build(a.metadata,a.out)
