"""Audit source geometry on optimization caches; never activate new targets."""
import math
from collections import Counter,defaultdict
import numpy as np
from common14 import *
from common9 import registered_box
from common6 import candidates
from geometry6 import polygons,certify
from g5_common import overlap


def read_derived(path):
    import zipfile
    path=Path(path)
    if path.stat().st_size>2_000_000:raise ValueError('Oversized diagnostic cache')
    with zipfile.ZipFile(path) as z:
        if sum(a.file_size for a in z.infolist())>3_000_000 or len(z.infolist())>20:raise ValueError('Unsafe diagnostic NPZ')
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    masks={'target','legacy','used','added_available'};grads={'grad_positive','grad_native','grad_added','grad_classification'};flags={'geometry_size_bin','padding_anchor','known_Chair_IoU05','fragment_heuristic'}
    if set(data)!=masks|grads|flags|{'assignment','grad_box'}:raise ValueError('Unexpected diagnostic schema')
    n=len(data['assignment'])
    if not 0<n<=3600 or data['assignment'].shape!=(n,) or data['assignment'].dtype!=np.int16:raise ValueError('Invalid diagnostic assignment frame')
    for key in masks|grads:
        a=data[key]
        if a.shape!=(n,7) or a.dtype!=np.dtype('uint8' if key in masks else 'float32') or not np.isfinite(a).all():raise ValueError('Invalid diagnostic mask/gradient')
        if key in masks and ((a>1).any()):raise ValueError('Nonbinary diagnostic mask')
    for key in flags:
        a=data[key]
        if a.shape!=(n,) or a.dtype!=np.uint8 or (a>(3 if key=='geometry_size_bin' else 1)).any():raise ValueError('Invalid diagnostic geometry flag')
    if data['grad_box'].shape!=(n,4) or data['grad_box'].dtype!=np.float32 or not np.isfinite(data['grad_box']).all():raise ValueError('Invalid diagnostic box gradient')
    if (data['legacy']&~data['used']).any() or (data['target']&~data['used']).any():raise ValueError('Lost original supervised target')
    return data


def run():
    _,train,dev,photos,reg=training_context();byid={r['image_id']:r for r in train};summaries=read(RAW14/'query-summaries.json')['records'];bundle=read(SOURCE/'source-negative-evidence.json')['records'];source={r['annotation']['annotation_key']:r for r in candidates()};observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in candidates()}
    current_ids={r['candidate_id'] for r in bundle};polycache={};proposals={};reasons=Counter();context_examples=[];coverage=[]
    for summary in summaries:
        row=byid[summary['image_id']];require_training_row(row,train,dev);a=load_npz(ROOT/summary['query_file']);d=read_derived(ROOT/summary['derived_file']);box=registered_box(a['square_cxcywh'],row['width'],row['height'],clip=False);p=1/(1+np.exp(-a['logits'].astype(float)));actual_added=d['used'][:,1].astype(bool)&~d['target'][:,1].astype(bool)
        selected=[r for r in bundle if r['image_id']==row['image_id']];covered={e['candidate_id']:[] for e in selected}
        for e in selected:
            b=np.asarray(e['box']);inside=(box[:,:2]>=b[:2]).all(1)&(box[:,2:]<=b[2:]).all(1)&d['added_available'][:,1].astype(bool);covered[e['candidate_id']]=np.flatnonzero(inside).tolist()
        coverage.append(dict(image_id=row['image_id'],state=summary['state'],added_terms=int(actual_added.sum()),counterfactual_available_terms=int(d['added_available'][:,1].sum()),source_negative_scores=quantiles(p[actual_added,1]),available_source_scores=quantiles(p[d['added_available'][:,1].astype(bool),1]),actual_hard_015=int((actual_added&(p[:,1]>=.15)).sum()),actual_hard_05=int((actual_added&(p[:,1]>=.5)).sum()),region_query_indices=covered))
        if summary['state'] not in ('c8_corrected_e5','c11_e5','c13_e5'):continue
        protected=[a['box'] for a in row['annotations']]+[a['box'] for a in reg[row['image_id']]['protected']]+[a['box'] for a in photos[row['image_id']]['annotations'] if a['mapped_auxiliary_category']=='protected_furniture']+[r['annotation']['box'] for r in candidates() if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        high=np.flatnonzero((p[:,1]>=.15)&~d['used'][:,1].astype(bool))
        for q in high:
            b=box[q].tolist()
            if not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):reasons['query_outside_original_frame']+=1;continue
            if d['assignment'][q]>=0:reasons['matched_positive_query']+=1;continue
            # Known furniture boxes remain conservative blockers even if the
            # native visible foreground mask could in theory disambiguate them.
            if any(overlap(b,g) for g in protected):reasons['known_furniture_or_uncertain_overlap']+=1;continue
            pixels=[math.floor(b[0]*row['width']),math.floor(b[1]*row['height']),math.ceil(b[2]*row['width']),math.ceil(b[3]*row['height'])]
            if min(pixels[2]-pixels[0],pixels[3]-pixels[1])<32:reasons['below_existing32px_visual_region_minimum']+=1;continue
            found=False
            for annotation in photos[row['image_id']]['annotations']:
                bb=annotation['box']
                if not (bb[0]<=b[0]<b[2]<=bb[2] and bb[1]<=b[1]<b[3]<=bb[3]):continue
                if not annotation['metadata_candidate_eligible'] or annotation['mapped_auxiliary_category'] in ('protected_furniture','furniture_parts'):continue
                key=annotation['annotation_key']
                if key not in polycache:polycache[key]=polygons(annotation['original_annotation'],row['width'],row['height'])
                for pi,poly in enumerate(polycache[key]):
                    if not certify(pixels,poly):continue
                    found=True;candidate_key=digest(dict(image_id=row['image_id'],annotation_key=key,box_pixels=pixels))[:20];old=source.get(key);prior=observations.get(old['candidate_id']) if old else None
                    record=proposals.setdefault(candidate_key,dict(candidate_key=candidate_key,image_id=row['image_id'],source_sha256=row['sha256'],identity_group=row['identity_group'],native_label=annotation['native_label'],mapped_context=annotation['mapped_auxiliary_category'],annotation_key=key,native_dataset=annotation['dataset'],native_category_id=annotation['native_category_id'],annotation_sha256=digest(annotation['original_annotation']),annotation_source=annotation['annotation_source'],annotation_source_sha256=annotation['annotation_source_sha256'],box_pixels=pixels,box=[pixels[0]/row['width'],pixels[1]/row['height'],pixels[2]/row['width'],pixels[3]/row['height']],native_polygon_index=pi,native_polygon=[list(x) for x in poly],source_object_hypothesis_group=annotation['object_hypothesis_group'],prior_candidate_id=old['candidate_id'] if old else None,prior_visual_outcome=prior['visual_outcome'] if prior else None,prior_source_region_approved=bool(old and old['candidate_id'] in current_ids),source_geometry_certified=True,known_furniture_overlap=False,completeness='Visible native polygon only;no photograph-level Chair absence',evidence_tier='source_geometry_pending_query_local_visual_verification',human_reviewed=False,training_eligible=False,training_activated=False,query_observations=[]))
                    record['query_observations'].append(dict(state=summary['state'],query_index=int(q),Chair_probability=float(p[q,1]),all_category_probabilities=p[q].tolist(),query_box=b))
            if not found:reasons['no_certified_visible_nonchair_source_polygon']+=1
        # Bounded inspection suggestions use training/native context only.
        for group in ('people','plants_flowers','screens','windows_doors','furniture_parts','protected_furniture'):
            options=[]
            anns=[a for a in photos[row['image_id']]['annotations'] if a['mapped_auxiliary_category']==group]
            for q in high:
                b=box[q];area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
                if not area:continue
                for annotation in anns:
                    g=annotation['box'];inter=max(0,min(b[2],g[2])-max(b[0],g[0]))*max(0,min(b[3],g[3])-max(b[1],g[1]))
                    if inter/area>=.7:options.append((float(p[q,1]),int(q),annotation))
            if options:
                score,q,ann=max(options,key=lambda x:(x[0],-x[1]));context_examples.append(dict(image_id=row['image_id'],identity_group=row['identity_group'],photo_key=summary['photo_key'],state=summary['state'],query_index=q,Chair_probability=score,query_box=box[q].tolist(),native_label=ann['native_label'],native_box=ann['box'],annotation_key=ann['annotation_key'],context_group=group,evidence_tier='native_bbox_overlap_context_only',Chair_negative_proven=False,human_reviewed=False))
    save(RAW14/'negative-coverage.json',dict(records=coverage))
    save(RAW14/'new-hard-negative-candidates.json',dict(candidates=sorted(proposals.values(),key=lambda r:r['candidate_key']),newly_training_eligible=0,source_geometry_candidates=len(proposals),all_require_query_local_visual_check_and_separate_activation=True,filter_counts=dict(reasons),source_annotations_modified=False,validation_used=False))
    # At most12 examples, balanced across source groups/context families;
    # suggestions are not labels and no completeness decisions are invented.
    picked=[];seen=set();groups=set()
    for kind in ('people','plants_flowers','screens','windows_doors','furniture_parts','protected_furniture'):
        opts=sorted([r for r in context_examples if r['context_group']==kind],key=lambda r:(r['identity_group'] in groups,-r['Chair_probability'],r['image_id'],r['state'],r['query_index']))
        for r in opts:
            key=(r['image_id'],r['query_index'])
            if key in seen:continue
            picked.append(dict(r,sample_id='T'+str(len(picked)+1).zfill(2)));seen.add(key);groups.add(r['identity_group'])
            if sum(x['context_group']==kind for x in picked)>=2:break
    save(RAW14/'training-inspection-sample.json',dict(records=picked,selection='Optimization-only native context families and saved Chair confidence;not development errors,not a prevalence estimate',not_training_labels=True))
    summary={}
    for state in STATES:
        ss=[r for r in coverage if r['state']==state['key']];region=set(e for r in ss for e,v in r['region_query_indices'].items() if v);summary[state['key']]=dict(available_terms=sum(r['counterfactual_available_terms'] for r in ss),actual_applied_terms=sum(r['added_terms'] for r in ss),actual_hard_015=sum(r['actual_hard_015'] for r in ss),actual_hard_05=sum(r['actual_hard_05'] for r in ss),regions_with_query_coverage=len(region),regions_without_query_coverage=41-len(region))
    save(REPORT14/'hard-negative-audit.json',dict(existing_regions=41,by_state=summary,new_source_geometry_candidate_regions=len(proposals),newly_qualified_hard_negative_regions=0,awaiting_query_local_visual_evidence=len(proposals),candidate_filter_counts=dict(reasons),inventory_native_annotation_records_on_training=sum(len(p['annotations']) for p in photos.values()),source_records_not_independent_physical_objects=True,all_new_targets_inactive=True,source_annotations_modified=False,no_inference=True))
    print('C14 hard-negative search:',len(proposals),'new geometry-qualified proposals;0auto-approved;training-only',flush=True)


if __name__=='__main__':run()
