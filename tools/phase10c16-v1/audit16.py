"""Per-source/region cached output replay;no real-model gradient fabrication."""
from collections import Counter
import csv,io,subprocess
import numpy as np
import torch
from common16 import *
from balanced16 import catalog,branches,Policy


def run():
    if DATA16.exists() or REPORT16.exists():raise FileExistsError('Fresh C16 audit outputs required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT16:raise ValueError('Unexpected parent')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH16:raise ValueError('Dedicated experimental branch required')
    records,index,history=verify16_inputs();prepared,train,dev,_,reg=training_context()
    from common6 import candidates
    from masks6 import validate_bundle
    frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,observations)
    ctx=catalog(bundle['records'],train);byid={r['image_id']:r for r in train};reports=[];output_grads=[];total=0.;total_loss=0.;bands=[dict(label=k,count=0,raw_balanced_output_L1=0.) for k in ('p<.01','.01<=p<.05','.05<=p<.15','p>=.15')]
    torch.set_num_threads(2)
    for receipt,record in zip(index,records):
        row=byid[receipt['image_id']];require_training_row(row,train,dev);e=[x for x in bundle['records'] if x['image_id']==row['image_id']];protected=[p['box'] for p in reg[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        with np.load(ROOT/receipt['raw_file'],allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
        z=torch.from_numpy(a['logits'].copy()).requires_grad_();b=torch.from_numpy(a['square_cxcywh'].copy()).requires_grad_();r=branches(z,b,row,e,protected,ctx,Policy(enabled=True));g=torch.autograd.grad(r['source_raw'],z)[0].detach().numpy()
        if not np.array_equal(r['masks']['added'].numpy(),a['source_added']) or not np.array_equal(r['masks']['mask'].numpy(),a['mask']) or (g[~a['source_added'].astype(bool)]!=0).any():raise ValueError('C16 altered masks or ignored outputs')
        probs=1/(1+np.exp(-a['logits'][:,1].astype(float)));w=r['weights'].numpy();source_anns={x['candidate_id']:x for x in e};region_reports=[]
        for region in r['normalization']['regions']:
            qs=region['query_indices'];source=source_anns[region['region_id']];q=np.asarray(qs,dtype=int);native_key=source['source_annotation_key'];boxes=torch.from_numpy(a['square_cxcywh'][q])
            # Pairwise IoU measures query redundancy,not independent captures.
            pairs=overlap05=0
            if len(q)>1:
                from audit14 import overlap_arrays
                from expanded import cxcywh_to_xyxy
                bb=cxcywh_to_xyxy(boxes).numpy();ii=overlap_arrays(bb,bb)[0];upper=np.triu_indices(len(q),1);pairs=len(upper[0]);overlap05=int((ii[upper]>=.5).sum())
            oldf=float(np.abs(a['output_grad_source_focal'][q,1].astype(float)).sum());oldb=float(np.abs(a['output_grad_source_BCE'][q,1].astype(float)).sum());newg=float(np.abs(g[q,1].astype(float)).sum())
            region_reports.append(dict(region_id=region['region_id'],aliases=region['aliases'],source_annotation_key=native_key,native_dataset=source['source_native_dataset'],native_category=source['source_native_category'],source_annotation_sha256=source['source_annotation_sha256'],source_verified_under_existing_policy=source['source_verified_under_policy'],human_reviewed=source['human_reviewed'],source_polygon_sha256=digest(source['native_polygon']),queries=qs,unique_query_signatures=region['covered_signatures'],query_pairs=pairs,query_pairs_IoU_at_least05=overlap05,Chair_confidence=quantiles(probs[q].tolist()),focal_output_gradient_L1=oldf,BCE_output_gradient_L1=oldb,balanced_raw_output_gradient_L1=newg,balanced_raw_negative_loss=float((.75*torch.nn.functional.softplus(z.detach()[q,1])*r['weights'][q]).sum()),focal_negative_loss=float((.75*torch.sigmoid(z.detach()[q,1]).square()*torch.nn.functional.softplus(z.detach()[q,1])).sum()/r['stats']['positive_boxes']),BCE_negative_loss=float((.75*torch.nn.functional.softplus(z.detach()[q,1])).sum()/r['stats']['positive_boxes']),negative_parameter_norm=None,parameter_decomposition='Per-region signed parameter gradients/pre-head features were not saved;no apportionment of photo-level norms',independence='Each admitted source polygon retains provenance;different regions are not asserted independent physical objects'))
        signed=record['losses_and_gradients'];pg=dict(index=receipt['index'],image_id=row['image_id'],identity_group=row['identity_group'],positive_Chair_boxes=sum(not x['ignore'] and x['label']=='chair' for x in row['annotations']),all_positive_boxes=r['stats']['positive_boxes'],verified_regions=len(e),source_queries=record['counts']['source_negative'],Chair_confidence=quantiles(record['source_negative_scores']),focal_negative_loss=signed['source_focal']['loss'],BCE_negative_loss=signed['source_BCE']['loss'],balanced_raw_negative_loss=float(r['source_raw'].detach()),positive_projection_norm=signed['positive']['projection_L2'],focal_negative_projection_norm=signed['source_focal']['projection_L2'],BCE_negative_projection_norm=signed['source_BCE']['projection_L2'],positive_BCE_cosine=record['projection_cosines']['positive_vs_source_BCE'],new_balanced_parameter_norm=None,new_balanced_parameter_direction=None,cap_factor=None,cap_measurement_required='C8 all6class-head gradients on original initialization;C15R was adapted C11 with calibration frozen',normalization=r['normalization'],regions=region_reports,balanced_raw_output_gradient_L1=float(np.abs(g.astype(float)).sum()),same_target_and_ignore_masks=True,positive_loss_preserved=True,source_support='Existing41automatic source-polygon negatives,not new human labels')
        total+=pg['balanced_raw_output_gradient_L1'];total_loss+=pg['balanced_raw_negative_loss'];reports.append(pg)
        for k in range(4):
            cond=[probs<.01,(probs>=.01)&(probs<.05),(probs>=.05)&(probs<.15),probs>=.15][k]&a['source_added'][:,1].astype(bool);bands[k]['count']+=int(cond.sum());bands[k]['raw_balanced_output_L1']+=float(np.abs(g[cond,1].astype(float)).sum())
        output_grads.append(dict(index=receipt['index'],source_queries=record['counts']['source_negative'],balanced_raw_output_gradient_L1=pg['balanced_raw_output_gradient_L1']))
    DATA16.mkdir(parents=True);REPORT16.mkdir(parents=True);emit16(DATA16/'catalog.json',ctx);emit16(DATA16/'source-region-audit.json',dict(groups=reports,training_rows_sha256=digest(train),original_source_evidence_sha256=sha(SOURCE/'source-negative-evidence.json'),query_index=index,new_labels=0))
    old=read(REPORTR/'results.json');caps=dict(new_balanced_real_model_parameter_norm=None,new_balanced_real_model_parameter_direction=None,real_model_cap_factors_unavailable=True,all7_logits_available=True,prehead_features_unavailable=True,C15R_calibration_frozen_not_C8_baseline=True)
    emit16(REPORT16/'confirmed-history.json',history);emit16(REPORT16/'comparison.json',dict(focal=old['components']['source_focal'],BCE=old['components']['source_BCE'],source_balanced_raw=dict(sum_output_gradient_L1=total,mean_per_photo_loss=total_loss/24,output_strength_relative_focal=total/old['components']['source_focal']['sum_per_photo_output_logit_L1'],bands=bands),parameter_space_limitations=caps,groups=output_grads,ignored_and_positive_masks_unchanged=True,known_groups=24,source_regions=41,source_queries=281,new_real_model_forwards=0,coefficient_sweeps=0,threshold_tuning=0))
    emit16(REPORT16/'source-group-audit.json',dict(groups=reports,global_catalog=dict(source_records=ctx['source_records'],unique_supervision_cells=ctx['unique_supervision_cells'],group_region_counts=ctx['group_region_counts']),per_region_parameter_gradients_unavailable=True,new_negatives=0,source_photos_not_redecoded=True))
    print('C16 cached-output audit complete;raw balanced output mass',total,';no new parameter-gradient claims',flush=True)


if __name__=='__main__':run()
