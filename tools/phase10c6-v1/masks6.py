"""Opt-in automatic source-polygon masks; preserve legacy loss and human gates."""
import torch
from common6 import *
from evidence6 import validate_source
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from expanded import supervision,to_square,xyxy_to_cxcywh,cxcywh_to_xyxy
DEFAULT=dict(enabled=False,approved=False,mode='source_polygon_negative_v1')

def validate_bundle(bundle,rows,frozen,observations):
    if bundle.get('policy_sha256')!=digest(ADMISSION) or bundle.get('policy')!=ADMISSION:raise ValueError('Automatic-source evidence policy changed')
    index={r['candidate_id']:r for r in frozen};byrow={r['image_id']:r for r in rows};seen=set()
    for e in bundle['records']:
        if e.get('candidate_id') in seen:raise ValueError('Duplicate source evidence')
        seen.add(e['candidate_id']);r=index[e['candidate_id']];row=byrow.get(e['image_id'])
        if row is None or not optimizer_member(row) or any(e.get(k)!=row[k] for k in ('image_id','identity_group','historical_partition','internal_partition')) or e.get('source_photo_sha256')!=row['sha256']:raise PermissionError('Automatic evidence outside immutable training partition')
        if e.get('human_reviewed') is not False or e.get('human_confirmed') is not False or e.get('source_verified_under_policy') is not True:raise ValueError('Automatic evidence must never masquerade as human approval')
        if e.get('category')!='chair' or e.get('kind')!='source_polygon_visible_negative' or e.get('box')!=r['proposal']['box'] or e.get('box_pixels')!=r['proposal']['box_pixels'] or e.get('native_polygon')!=r['proposal']['polygon']:raise ValueError('Source target/geometry changed')
        if e.get('source_annotation_sha256')!=digest(r['annotation']['original_annotation']) or e.get('source_annotation_key')!=r['annotation']['annotation_key'] or e.get('automatic_visual_observation_sha256')!=digest(observations[e['candidate_id']]):raise ValueError('Native or visual provenance changed')
        if e.get('source_native_dataset')!=r['annotation']['dataset'] or e.get('source_native_category')!=r['annotation']['native_category_id'] or e.get('native_polygon_index')!=r['proposal']['polygon_index'] or e.get('policy_sha256')!=digest(ADMISSION):raise ValueError('Native taxonomy or per-region policy changed')
        validate_source(r,observations[e['candidate_id']],frozen,observations)
    for i,a in enumerate(bundle['records']):
        for b in bundle['records'][i+1:]:
            if a['image_id']==b['image_id'] and overlap(a['box'],b['box']):raise ValueError('Duplicate/overlapping source targets')
    return True

def authorized(config,bundle,prepared):
    if config.get('enabled') is not True or config.get('approved') is not True or not config.get('approval_reference'):raise PermissionError('Automatic-source training remains disabled pending separate authorization')
    if config.get('mode')!='source_polygon_negative_v1' or config.get('evidence_sha256')!=digest(bundle) or config.get('manifest_sha256')!=digest(prepared):raise ValueError('Training configuration/data binding mismatch')
    if config.get('legacy_loss')!=dict(classification=1,L1=5,GIoU=2,alpha=.25,gamma=2):raise ValueError('Do not mix coefficients/normalization with mask ablation')
    for p in (RAW/'corrections').glob('*.json'):
        note=read(p)
        if note.get('decision') in ('exclude','needs_recheck') and note.get('candidate_id') in {e['candidate_id'] for e in bundle['records']}:raise PermissionError('An optional correction blocks this frozen evidence; prepare a new version')
    return True

def source_supervision(logits,boxes,row,matched,labels,config,bundle,prepared,frozen,observations,additional_protected=()):
    """Future integration entry point: never enable from defaults or fake humans."""
    authorized(config,bundle,prepared);validate_bundle(bundle,prepared['rows'],frozen,observations)
    selected=[e for e in bundle['records'] if e['image_id']==row['image_id']]
    uncertain=[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
    return target_mask(logits,boxes,row,matched,labels,selected,list(additional_protected)+uncertain)

def target_mask(logits,boxes,row,matched,labels,selected,extra_protected=()):
    # Pure CPU tensor operation used by tests/coordinate probes, not inference.
    target,valid=supervision(logits,boxes,row,matched,labels);added=torch.zeros_like(valid);xy=cxcywh_to_xyxy(boxes.detach());frame=to_square(boxes.new_tensor([0.,0.,1.,1.]),row['width'],row['height'])
    safe=(xy[:,:2]>=frame[:2]).all(-1)&(xy[:,2:]<=frame[2:]).all(-1)&(xy[:,2:]>xy[:,:2]).all(-1)
    protect=[a['box'] for a in row['annotations'] if a['label']=='chair' or a['ignore']]+list(extra_protected)
    for b in protect:
        q=to_square(boxes.new_tensor(b),row['width'],row['height']);safe&=~((torch.minimum(xy[:,2:],q[2:])-torch.maximum(xy[:,:2],q[:2])).clamp(min=0).prod(-1)>0)
    permitted=torch.zeros(len(boxes),dtype=torch.bool,device=boxes.device)
    for e in selected:
        if e['image_id']!=row['image_id'] or e['category']!='chair':raise ValueError('Wrong source evidence row/category')
        q=to_square(boxes.new_tensor(e['box']),row['width'],row['height']);permitted|=(xy[:,:2]>=q[:2]).all(-1)&(xy[:,2:]<=q[2:]).all(-1)
    added[:,1]=safe&permitted&~valid[:,1];valid|=added
    if matched is not None:target[matched,labels]=1;valid[matched,labels]=True;added[matched,labels]=False
    return target,valid,added

def run():
    torch.set_num_threads(2);frozen=candidates();obs={r['candidate_id']:read(RAW/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(RAW/'source-negative-evidence.json');prepared=manifest();validate_bundle(bundle,prepared['rows'],frozen,obs)
    reg={p['image_id']:p for p in registry()['photos'] if p['role']=='optimization'};records=[];total=0;probes=0
    for row in prepared['rows']:
        if not optimizer_member(row):continue
        ev=[e for e in bundle['records'] if e['image_id']==row['image_id']];anns=[a for a in row['annotations'] if not a['ignore']]
        pos=xyxy_to_cxcywh(to_square(torch.tensor([a['box'] for a in anns]),row['width'],row['height']))
        ys,xs=torch.meshgrid((torch.arange(30)+.5)/30,(torch.arange(30)+.5)/30,indexing='ij');grid=torch.stack([xs.flatten(),ys.flatten(),torch.full((900,),1/60),torch.full((900,),1/60)],-1);parts=[pos,grid]
        for e in ev:
            x1,y1,x2,y2=e['box'];cx=(x1+x2)/2;cy=(y1+y2)/2;b=[cx-(x2-x1)/4,cy-(y2-y1)/4,cx+(x2-x1)/4,cy+(y2-y1)/4];parts.append(xyxy_to_cxcywh(to_square(torch.tensor([b]),row['width'],row['height'])))
        boxes=torch.cat(parts);logits=torch.zeros(len(boxes),7);ii=torch.arange(len(anns));labels=torch.tensor([CLASSES.index(a['label']) for a in anns]);a,am=supervision(logits,boxes,row,ii,labels)
        extra=[p['box'] for p in reg[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        b,bm,added=target_mask(logits,boxes,row,ii,labels,ev,extra)
        if not torch.equal(a,b) or not torch.equal(am[ii,labels],bm[ii,labels]) or (am&~bm).any() or added[:,[0,2,3,4,5,6]].any():raise ValueError('Legacy positives/masks or non-Chair terms changed')
        n=int(added.sum());total+=n;probes+=int(added[-len(ev):,1].sum()) if ev else 0
        records.append(dict(image_id=row['image_id'],added_Chair_negative_terms=n,source_regions=len(ev),positive_targets_preserved=True))
    if torch.cuda.is_initialized():raise ValueError('CPU mask analysis unexpectedly initialized CUDA')
    detailed=dict(scope='CPU synthetic grid plus strictly interior source-region coordinate probes; not actual OWLv2 predictions',records=records,changed_mask_terms=total,
        region_interior_probes_added=probes,meaningful_coordinate_fixture_difference=total>0,actual_GPU_query_coverage_unmeasured=True,loss_coefficients_unchanged=True,
        positives_preserved=True,only_Chair_negative_mask_added=True,training_enabled=False,model_forwards=0,optimizer_updates=0)
    save(RAW/'mask-comparison.json',detailed);save(REPORT/'mask-summary.json',{k:v for k,v in detailed.items() if k!='records'});print('ADDED CHAIR TERMS',total,';interior probes',probes,';real model forwards0')

if __name__=='__main__':run()
