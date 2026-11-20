"""Exact CPU mask comparison using synthetic boxes and immutable train rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from preflight import *

def run():
    import torch
    import hashlib
    torch.set_num_threads(2)
    from trusted_loss import negative_mask,load_configuration,validate_bundle
    from expanded import supervision,to_square,xyxy_to_cxcywh
    prepared=manifest();rows=[r for r in prepared['rows'] if optimizer_member(r)]
    evidence=read(RAW3/'supervision-manifest-v1.json')['accepted_negative_evidence']
    if evidence:raise PermissionError('This probe is pinned to the currently empty accepted bundle')
    records=[];positive_total=0;chair_negative=0
    for r in rows:
        y,x=torch.meshgrid((torch.arange(60)+.5)/60,(torch.arange(60)+.5)/60,indexing='ij')
        boxes=torch.stack([x.flatten(),y.flatten(),torch.full((3600,),1/60),torch.full((3600,),1/60)],dim=-1)
        anns=[a for a in r['annotations'] if not a['ignore']];ii=torch.arange(len(anns));labels=torch.tensor([('sofa','chair','table','bed','desk','cabinet','lamp').index(a['label']) for a in anns])
        boxes[:len(anns)]=xyxy_to_cxcywh(to_square(torch.tensor([a['box'] for a in anns]),r['width'],r['height']))
        logits=torch.zeros(3600,7);ta,ma=supervision(logits,boxes,r,ii,labels);tb,mb,added=negative_mask(logits,boxes,r,ii,labels,evidence)
        if not torch.equal(ta,tb) or not torch.equal(ma,mb) or added.any():raise ValueError('Unexpected target-mask contrast with no trusted evidence')
        positives=int((ma&ta.bool()).sum());negatives=ma&~ta.bool();positive_total+=positives;chair_negative+=int(negatives[:,1].sum())
        mask_sha=hashlib.sha256(ma.numpy().tobytes()).hexdigest();target_sha=hashlib.sha256(ta.numpy().tobytes()).hexdigest()
        records.append(dict(image_id=r['image_id'],queries=3600,matched_positive_targets=positives,negative_terms_by_class={c:int(negatives[:,j].sum()) for j,c in enumerate(('sofa','chair','table','bed','desk','cabinet','lamp'))},
            target_differences=0,validity_mask_differences=0,added_trusted_Chair_negatives=0,legacy_mask_sha256=mask_sha,corrected_mask_sha256=mask_sha,targets_sha256=target_sha))
    if positive_total!=182 or chair_negative!=0:raise ValueError('Training supervision counts changed')
    gate=''
    try:validate_bundle(load_configuration(REPORT2/'arm-B-corrected.json'),[],prepared)
    except PermissionError as e:gate=str(e)
    if not gate:raise ValueError('Empty evidence unexpectedly passed execution gate')
    result=dict(scope='CPU synthetic60x60 query boxes with native-positive boxes injected; not model inference or actual future optimizer-batch counts',
        optimization_photos=59,total_fixture_queries=59*3600,matched_native_positives=positive_total,legacy_chair_negative_terms=chair_negative,
        corrected_added_chair_negative_terms=0,all_target_and_mask_tensors_bitwise_equal=True,meaningful_AB_difference=False,
        algebra='With evidence=[] the permitted-region mask is identicallyFalse for any query boxes; added mask isFalse, hence both targets/masks are unchanged for every possible input, not only this fixture',
        execution_gate=gate,records=records,torch_version=torch.__version__,CUDA_initialized=torch.cuda.is_initialized(),optimizer_updates=0,model_forwards=0)
    save(RAW3/'mask-difference-audit.json',result)
    save(REPORT3/'mask-difference-summary.json',{k:v for k,v in result.items() if k!='records'})
    print('MASK PASS:59photos/182positive targets; zero Chair negatives and zero changed mask entries; A/B blocked')

if __name__=='__main__':run()
