"""Training-only cached-query mining. Unmatched predictions never become labels."""
import math
import numpy as np
import torch
from common21 import *
from teacher21 import query_arrays
from geometry6 import polygons,certify
from g5_common import overlap
from common9 import registered_box


def protected_boxes(row,photo,registry,prior,observations):
    return [a['box'] for a in row['annotations']]+[a['box'] for a in photo['annotations'] if a['protect_furniture'] or a['native_ignored'] or a['part_or_semantics_unresolved']]+[r['box'] for r in registry[row['image_id']]['protected']]+[r['annotation']['box'] for r in prior if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']


def query_caches():
    records=[dict(state='original_C17',image_id=r['image_id'],path=r['cache'],sha256=r['cache_sha256']) for r in read(DATA21/'teacher-cache-index.json')['records']]
    for name,root,epoch in [('C8_B_E5',C8/'B-corrected',5),('C18_E1',RUN18,1),('C20_E3',RUN20,3),('C20_E5',RUN20,5)]:
        marker=read(root/'probes'/('epoch-'+str(epoch))/'complete.json')
        # These markers contain training-only receipts,not evaluation rows.
        for index,r in enumerate(marker.get('records',[]),1):
            path=root/'probes'/('epoch-'+str(epoch))/(str(index).zfill(2)+'.npz');records.append(dict(state=name,image_id=r['image_id'],path=str(path.relative_to(ROOT)),sha256=r['sha256']))
        if not marker.get('records'):
            ids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']]
            for index,ident in enumerate(ids,1):
                path=root/'probes'/('epoch-'+str(epoch))/(str(index).zfill(2)+'.npz');receipt=read(path.with_suffix('.json'));h=receipt.get('sha256',receipt.get('query_file_sha256'))
                if receipt.get('image_id')!=ident or h is None:raise ValueError('Training cache receipt mismatch')
                records.append(dict(state=name,image_id=ident,path=str(path.relative_to(ROOT)),sha256=h))
    return records


def run():
    cfg,prepared,train,dev,reg,old_inventory,old_photos=context21();byid={r['image_id']:r for r in train};native=read(DATA21/'full-local-native-inventory.json');photos={p['image_id']:p for p in native['photos'] if p['image_id'] in byid}
    from common6 import candidates
    prior=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in prior};existing=read(SOURCE/'source-negative-evidence.json')['records'];oldregions={(r['image_id'],tuple(r['box'])):r for r in existing};polycache={};filters=Counter();candidates_out={};contexts={};coverage=[];safe_existing=[]
    for cache in query_caches():
        row=byid.get(cache['image_id'])
        if row is None:raise PermissionError('Nonoptimization cache cannot enter negative mining')
        require_training_row(row,train,dev);photo=photos[row['image_id']];a=query_arrays(ROOT/cache['path'],cache['sha256']);boxes=registered_box(a['square_cxcywh'],row['width'],row['height'],clip=False);scores=1/(1+np.exp(-a['logits'][:,1].astype(float)));block=protected_boxes(row,photo,reg,prior,observations)
        high=np.flatnonzero(scores>=.15);blocked=np.zeros(len(boxes),dtype=bool)
        for b in block:blocked|=(np.minimum(boxes[:,2:],b[2:])-np.maximum(boxes[:,:2],b[:2])).clip(min=0).prod(1)>0
        safe=np.isfinite(boxes).all(1)&(boxes[:,:2]>=0).all(1)&(boxes[:,2:]<=1).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1)
        trained_only=cache['state']!='original_C17'
        coverage.append(dict(state=cache['state'],image_id=row['image_id'],source_group=row['identity_group'],Chair_queries_ge_015=len(high),Chair_queries_ge_05=int((scores>=.5).sum()),high_queries_blocked_by_protected=int(blocked[high].sum())))
        for q in high:
            b=boxes[q].tolist();score=float(scores[q]);context=[]
            area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
            for ann in photo['annotations']:
                g=ann['box'];inter=max(0,min(b[2],g[2])-max(b[0],g[0]))*max(0,min(b[3],g[3])-max(b[1],g[1]))
                if area and inter/area>=.7:context.append(ann)
            for ann in context:
                key=(row['image_id'],ann['context'],ann['annotation_key']);d=contexts.setdefault(key,dict(image_id=row['image_id'],source_group=row['identity_group'],context=ann['context'],native_label=ann['native_label'],annotation_key=ann['annotation_key'],box=ann['box'],max_Chair_score=score,states=[],Chair_negative_proven=False))
                d['max_Chair_score']=max(score,d['max_Chair_score']);d['states']=sorted(set(d['states']+[cache['state']]))
            if not safe[q]:filters['outside_original_frame']+=1;continue
            if blocked[q]:filters['known_furniture_ignored_or_uncertain_overlap']+=1;continue
            pixels=[math.floor(b[0]*row['width']),math.floor(b[1]*row['height']),math.ceil(b[2]*row['width']),math.ceil(b[3]*row['height'])]
            if min(pixels[2]-pixels[0],pixels[3]-pixels[1])<32:filters['below32px_query_local_verification_resolution']+=1;continue
            found=False
            for ann in photo['annotations']:
                g=ann['box']
                if not(g[0]<=b[0]<b[2]<=g[2] and g[1]<=b[1]<b[3]<=g[3]):continue
                if ann['native_ignored'] or ann['protect_furniture'] or ann['part_or_semantics_unresolved']:continue
                key=ann['annotation_key']
                if key not in polycache:polycache[key]=polygons(ann['original_annotation'],row['width'],row['height'])
                for index,poly in enumerate(polycache[key]):
                    if not certify(pixels,poly):continue
                    found=True;candidate_id='C21-'+digest(dict(image_id=row['image_id'],annotation_key=key,box_pixels=pixels))[:20];record=candidates_out.setdefault(candidate_id,dict(candidate_id=candidate_id,image_id=row['image_id'],source_group=row['identity_group'],source_sha256=row['sha256'],path=row['path'],width=row['width'],height=row['height'],box_pixels=pixels,box=[pixels[0]/row['width'],pixels[1]/row['height'],pixels[2]/row['width'],pixels[3]/row['height']],native_dataset=ann['dataset'],native_label=ann['native_label'],context=ann['context'],annotation_key=key,annotation_source=ann['source'],annotation_source_sha256=ann['source_sha256'],annotation_sha256=ann['original_annotation_sha256'],native_polygon_index=index,native_polygon=[list(p) for p in poly],source_geometry_certified=True,protected_overlap=False,completeness='Only native visible instance polygon;not whole-photo Chair absence',licence=photo['rights'],native_annotation_licence_as_recorded='https://creativecommons.org/licenses/by/4.0/',category='chair',other_furniture_negative_claim=False,query_observations=[],evidence_tier='B_pending_actual_query_local_visual_verification',training_eligible=False,training_activated=False,human_reviewed=False,human_confirmed=False,assistant_visual_reviewed=False))
                    record['query_observations'].append(dict(state=cache['state'],query_index=int(q),Chair_score=score,box=b,cache=cache['path'],cache_sha256=cache['sha256']))
            if not found:filters['no_safe_native_visible_polygon_for_exact_query']+=1
        for e in existing:
            if e['image_id']!=row['image_id']:continue
            b=e['box'];inside=(boxes[:,:2]>=b[:2]).all(1)&(boxes[:,2:]<=b[2:]).all(1)&safe&~blocked
            selected=np.flatnonzero(inside)
            safe_existing.append(dict(state=cache['state'],image_id=row['image_id'],source_group=row['identity_group'],region=e['candidate_id'],native_category=e['source_native_category'],covered_queries=len(selected),queries_ge_015=int((scores[selected]>=.15).sum()),queries_ge_05=int((scores[selected]>=.5).sum()),maximum_score=float(scores[selected].max()) if len(selected) else None))
        print('MINED',cache['state'],row['image_id'],'high',len(high),'proposals',len(candidates_out),flush=True)
    records=sorted(candidates_out.values(),key=lambda r:(-max(v['Chair_score'] for v in r['query_observations']),r['source_group'],r['candidate_id']))
    emit21(DATA21/'hard-negative-candidates.json',dict(records=records,mining_optimization_only=True,threshold_for_mining=.15,threshold_not_adopted_for_production=True,source_annotations_modified=False,all_unreviewed_proposals_inactive=True));emit21(DATA21/'native-confuser-context.json',dict(records=list(contexts.values()),overlap_is_not_a_negative_label=True));emit21(DATA21/'cached-query-coverage.json',dict(records=coverage,existing_regions=safe_existing,filter_counts=dict(filters)))
    emit21(REPORT21/'mining-summary-initial.json',dict(cached_photograph_states=len(coverage),unique_training_photographs=len({r['image_id'] for r in coverage}),source_geometry_proposals=len(records),new_trusted_negatives=0,all_new_regions_pending_visual_or_semantic_evidence=True,context_groups={g:dict(annotation_records=sum(r['context']==g for r in contexts.values()),photos=len({r['image_id'] for r in contexts.values() if r['context']==g}),source_groups=len({r['source_group'] for r in contexts.values() if r['context']==g})) for g in sorted({r['context'] for r in contexts.values()})},filter_counts=dict(filters),model_inference=0,development_or_reserved_cache_accessed_for_mining=False))
    print('C21mining complete;proposals',len(records),';no approvals invented',flush=True)


if __name__=='__main__':run()
