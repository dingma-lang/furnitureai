"""Optimization-only statistics and derived query masks from immutable caches."""
import math,os
from collections import Counter,defaultdict
import numpy as np
import torch
from common14 import *
from replay14 import replay
from masks6 import validate_bundle
from common6 import candidates
from common9 import geometry


def overlap_arrays(boxes,gt):
    if not len(gt):return np.zeros((len(boxes),0)),np.zeros((len(boxes),0)),np.zeros((len(boxes),0))
    a=np.asarray(boxes,dtype=float);b=np.asarray(gt,dtype=float);inter=(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).clip(0).prod(2);aa=(a[:,2:]-a[:,:2]).clip(0).prod(1);ba=(b[:,2:]-b[:,:2]).clip(0).prod(1)
    return inter/np.maximum(aa[:,None]+ba[None,:]-inter,1e-20),inter/np.maximum(aa[:,None],1e-20),inter/np.maximum(ba[None,:],1e-20)


def geometry_and_context(boxes,z,derived,row,native):
    scores=1/(1+np.exp(-z.astype(float)));lab=z.argmax(1);pos=[a for a in row['annotations'] if not a['ignore']];chairs=[a['box'] for a in pos if a['label']=='chair'];other=[a['box'] for a in pos if a['label']!='chair'];ci,cp,cg=overlap_arrays(boxes,chairs);oi,op,og=overlap_arrays(boxes,other)
    area=(boxes[:,2:]-boxes[:,:2]).clip(0).prod(1);sizes=np.select([area<.01,area<.05,area<.2],[0,1,2],default=3).astype(np.uint8);high=(scores[:,1]>=.15);ignored=~derived['used'][:,1].astype(bool);selected=high&ignored
    native_boxes=[a['box'] for a in native];ni,npred,ngt=overlap_arrays(boxes,native_boxes);context=npred>=.5;source_counts=Counter()
    for j,a in enumerate(native):
        if a['mapped_auxiliary_category']!='protected_furniture':source_counts[a['mapped_auxiliary_category']]+=int((context[:,j]&selected).sum())
    native_groups=defaultdict(set)
    for j,a in enumerate(native):
        for q in np.flatnonzero(context[:,j]&selected):native_groups[int(q)].add(a['native_label'])
    near_chair=(ci.max(1)>=.5) if len(chairs) else np.zeros(len(z),bool)
    fragments=((cp>=.8)&(ci<.5)&(area[:,None]<.6*np.asarray([(b[2]-b[0])*(b[3]-b[1]) for b in chairs])[None,:])).any(1) if chairs else np.zeros(len(z),bool)
    q=np.arange(len(z));pad=((q%60+.5)/60>=row['width']/max(row['width'],row['height']))|((q//60+.5)/60>=row['height']/max(row['width'],row['height']))
    summary=dict(high_Chair_all_queries=int(high.sum()),high_Chair_argmax_queries=int((high&(lab==1)).sum()),high_Chair_ignored_queries=int(selected.sum()),high_Chair_ignored_IoU05_known_Chair=int((selected&near_chair).sum()),high_Chair_ignored_fragment_heuristic=int((selected&fragments).sum()),high_Chair_ignored_IoU05_other_active_furniture=int((selected&(oi.max(1)>=.5)).sum()) if len(other) else 0,
        high_Chair_ignored_native_bbox_context_counts=dict(source_counts),high_Chair_ignored_query_size_bins=dict(Counter(str(int(i)) for i in sizes[selected])),high_Chair_ignored_padding_anchors=int((selected&pad).sum()),high_Chair_ignored_box_outside_frame=int((selected&((boxes<0).any(1)|(boxes>1).any(1))).sum()),
        notes='Raw queries,no NMS/no FP claims;native bbox overlaps and fragments are context/heuristics,not absence proofs;categories can overlap',score_quantiles={c:quantiles(scores[:,j]) for j,c in enumerate(CLASSES)},
        added_negative_size_bins=dict(Counter(str(int(i)) for i in sizes[derived['added_available'][:,1].astype(bool)])))
    return summary,dict(geometry_size_bin=sizes,padding_anchor=pad.astype(np.uint8),known_Chair_IoU05=near_chair.astype(np.uint8),fragment_heuristic=fragments.astype(np.uint8)),native_groups


def run():
    torch.set_num_threads(2);prepared,train,dev,photos,registry=training_context();byid={r['image_id']:r for r in train};frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,observations)
    records=read(RAW14/'cache-index.json')['records'];folder=RAW14/'queries';folder.mkdir(exist_ok=False);summaries=[];reconstruction_failures=[];missing_parameter_records=[]
    for index,record in enumerate(records):
        row=byid[record['image_id']];require_training_row(row,train,dev);path=ROOT/record['file'];saved=read(ROOT/record['receipt_file'])
        if sha(path)!=record['sha256'] or sha(ROOT/record['receipt_file'])!=record['receipt_sha256']:raise ValueError('Cached input changed')
        selected=[e for e in bundle['records'] if e['image_id']==row['image_id']];protected=[p['box'] for p in registry[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        arrays=load_npz(path);derived,summary,boxes=replay(arrays,row,selected,protected,record['corrected'],saved);context,flags,_=geometry_and_context(boxes,arrays['logits'],derived,row,photos[row['image_id']]['annotations']);derived.update(flags)
        out=folder/(digest(row['image_id'])[:16]+'-'+record['state']+'.npz')
        with out.open('xb') as f:np.savez_compressed(f,**derived)
        if not summary['saved_loss_and_mask_checks_passed']:reconstruction_failures.append(dict(image_id=row['image_id'],state=record['state'],warnings=summary['reconstruction_warnings']))
        parameter={}
        for key in ('class_head_additional_gradient_norms','class_head_used_gradient_norms'):
            if key in saved:parameter[key]=saved[key]
            else:missing_parameter_records.append(dict(image_id=row['image_id'],state=record['state'],missing=key))
        named=parameter.get('class_head_used_gradient_norms',{});extra=parameter.get('class_head_additional_gradient_norms',{})
        if named and all(named.get(n) is not None and extra.get(n) is not None for n in PROJECTION_KEYS):
            data=math.sqrt(sum(named[n]**2 for n in PROJECTION_KEYS));neg=math.sqrt(sum(extra[n]**2 for n in PROJECTION_KEYS));parameter['projection_data_norm']=data;parameter['projection_source_negative_norm']=neg;parameter['source_negative_to_data_norm_ratio']=neg/data if data else None
        summary.update(image_id=row['image_id'],photo_key=digest(row['image_id'])[:16],identity_group=row['identity_group'],state=record['state'],epoch=record['epoch'],corrected_source_negatives_applied=record['corrected'],query_file=record['file'],query_sha256=record['sha256'],receipt_file=record['receipt_file'],derived_file=str(out.relative_to(ROOT)),derived_sha256=sha(out),context=context,saved_parameter_gradient_norms=parameter)
        save(out.with_suffix('.json'),summary);summaries.append(summary)
        if (index+1)%24==0:print('REPLAY',index+1,'/',len(records),'cached training states',flush=True)
    save(RAW14/'query-summaries.json',dict(records=summaries,reconstruction_failures=reconstruction_failures,missing_parameter_gradient_records=missing_parameter_records))
    aggregate={}
    for state in STATES:
        ss=[s for s in summaries if s['state']==state['key']];keys=('high_Chair_all_queries','high_Chair_argmax_queries','high_Chair_ignored_queries','high_Chair_ignored_IoU05_known_Chair','high_Chair_ignored_fragment_heuristic','high_Chair_ignored_IoU05_other_active_furniture','high_Chair_ignored_padding_anchors','high_Chair_ignored_box_outside_frame')
        aggregate[state['key']]=dict(label=state['label'],photos=len(ss),queries=3600*len(ss),positive_terms_by_category=np.sum([s['positive_terms_by_category'] for s in ss],0).tolist(),negative_terms_by_category=np.sum([s['negative_terms_by_category'] for s in ss],0).tolist(),ignored_terms_by_category=np.sum([s['ignored_terms_by_category'] for s in ss],0).tolist(),context={key:sum(s['context'][key] for s in ss) for key in keys},mean_classification_loss=float(np.mean([s['stats']['classification_loss'] for s in ss])),mean_positive_loss=float(np.mean([s['stats']['positive_classification'] for s in ss])),mean_native_negative_loss=float(np.mean([s['stats']['legacy_negative_classification'] for s in ss])),mean_source_negative_loss=float(np.mean([s['stats']['added_Chair_negative_loss'] for s in ss])),
            uniform_logit_direction_derivative_quantiles=quantiles([s['gradient']['uniform_seven_logit_offset_derivative'] for s in ss]),photos_classification_decreases_along_uniform_positive_logit_direction=sum(s['gradient']['uniform_seven_logit_offset_derivative']<0 for s in ss),source_negative_projection_gradient_ratio_quantiles=quantiles([s['saved_parameter_gradient_norms']['source_negative_to_data_norm_ratio'] for s in ss if s['saved_parameter_gradient_norms'].get('source_negative_to_data_norm_ratio') is not None]),
            measured_source_negative_projection_gradient_zero_photos=sum(s['saved_parameter_gradient_norms'].get('projection_source_negative_norm')==0 for s in ss),mean_counterfactual_source_BCE_loss=float(np.mean([s['gradient']['counterfactual_source_BCE_loss'] for s in ss])),logit_positive_L1=sum(sum(s['gradient']['logit_positive_L1_by_category']) for s in ss),logit_source_negative_L1=sum(sum(s['gradient']['logit_added_negative_L1_by_category']) for s in ss),counterfactual_source_BCE_logit_gradient_L1=sum(s['gradient']['counterfactual_source_BCE_logit_gradient_L1'] for s in ss))
    save(REPORT14/'query-assignment-summary.json',dict(classes=list(CLASSES),state_summaries=aggregate,cached_optimization_photos=24,optimization_photos_without_queries=35,total_query_state_records=len(summaries)*3600,reconstruction_failures=reconstruction_failures,original_GPU_assignments_were_not_saved=True,model_forwards=0,optimizer_updates=0,all_stats_training_only=True))
    print('C14 cache audit complete;reconstruction discrepancies',len(reconstruction_failures),flush=True)


if __name__=='__main__':run()
