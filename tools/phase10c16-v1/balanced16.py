"""Opt-in source-region BCE and bounded added-gradient composition.

No optimizer/model loading. The cap consumes actual unscaled branch gradients;
cached output derivatives cannot substitute for these parameter-space vectors.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from common16 import CLASSES,HEAD_KEYS
from expanded import cxcywh_to_xyxy,to_square
from loss8 import objective as baseline_objective
from loss15 import controlled_losses

CLASS_KEYS=frozenset(n for n in HEAD_KEYS if n.startswith('class_head.'))


@dataclass(frozen=True)
class Policy:
    enabled:bool=False
    rho:float=.25
    positive_norm_floor:float=1e-8
    negative_only_fraction_of_clip:float=.01
    global_clip_norm:float=1.


POLICY=Policy()


class PositiveAnchorUnavailable(ValueError):pass


def catalog(regions,rows):
    """Dedup supervision cells,not photograph/object identities;retain aliases."""
    byid={r['image_id']:r for r in rows};seen={};cells={}
    for e in regions:
        ident=e['image_id'];row=byid.get(ident)
        if row is None or row.get('internal_partition')!='optimization' or row.get('historical_partition')!='train' or e['category']!='chair' or e['identity_group']!=row['identity_group']:raise PermissionError('Only existing optimization Chair-negative evidence')
        if e.get('source_photo_sha256')!=row['sha256'] or e.get('source_verified_under_policy') is not True or e.get('human_reviewed') is not False:raise ValueError('Unverified or incorrect source frame')
        b=tuple(float(x) for x in e['box'])
        if len(b)!=4 or not all(math.isfinite(x) for x in b) or not(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):raise ValueError('Invalid source rectangle')
        sig=(ident,'chair',b);cid=e['candidate_id']
        if cid in seen and seen[cid]!=sig:raise ValueError('Conflicting duplicate region ID')
        seen[cid]=sig;v=cells.setdefault(sig,dict(image_id=ident,identity_group=row['identity_group'],box=list(b),aliases=set()))
        v['aliases'].add(cid)
    photos=defaultdict(list);totals=defaultdict(int)
    for v in cells.values():
        v=dict(v,aliases=sorted(v['aliases']));v['region_id']=v['aliases'][0];photos[v['image_id']].append(v);totals[v['identity_group']]+=1
    return dict(photos={i:sorted(v,key=lambda r:r['region_id']) for i,v in sorted(photos.items())},group_region_counts=dict(sorted(totals.items())),source_records=len(regions),unique_supervision_cells=len(cells),identity_merges=0)


def weights(z,b,row,masks,context,protected=()):
    """Fixed weights on admitted queries only;all geometry/logits detached."""
    if z.dtype!=torch.float32 or b.dtype!=torch.float32 or z.ndim!=2 or z.shape[1]!=7 or b.shape!=(len(z),4) or not torch.isfinite(z).all() or not torch.isfinite(b).all():raise ValueError('Invalid FP32 query arrays')
    added=masks['added'];target=masks['target'];legacy=masks['legacy']
    if added.dtype!=torch.bool or added.shape!=z.shape or target.shape!=z.shape or legacy.shape!=z.shape or (added&target.bool()).any() or (added&legacy).any() or added[:,[0,2,3,4,5,6]].any():raise ValueError('Invalid pre-existing additional-negative mask')
    out=torch.zeros(len(z),dtype=torch.float32,device=z.device);regions=context['photos'].get(row['image_id'],[]);R=context['group_region_counts'].get(row['identity_group'],0)
    if regions and (R<len(regions) or any(r['identity_group']!=row['identity_group'] for r in regions)):raise ValueError('Invalid source-group denominator')
    boxes=cxcywh_to_xyxy(b.detach()).cpu().tolist();scores=z.detach()[:,1].cpu().tolist();clusters={};block=[a['box'] for a in row.get('annotations',[]) if a['label']=='chair' or a['ignore']]+list(protected)
    # Match the executed mask's FP32 square frame exactly;no inverse-transform
    # rounding tolerance and no expansion/shrinkage of source geometry.
    block=[to_square(b.new_tensor(g),row['width'],row['height']).cpu().tolist() for g in block]
    region_boxes={r['region_id']:to_square(b.new_tensor(r['box']),row['width'],row['height']).cpu().tolist() for r in regions}
    frame=to_square(b.new_tensor([0.,0.,1.,1.]),row['width'],row['height']).cpu().tolist()
    for q in torch.nonzero(added[:,1],as_tuple=False).flatten().cpu().tolist():
        bb=boxes[q]
        if not(frame[0]<=bb[0]<bb[2]<=frame[2] and frame[1]<=bb[1]<bb[3]<=frame[3]):raise ValueError('Admitted query outside original frame')
        if any(min(bb[2],g[2])>max(bb[0],g[0]) and min(bb[3],g[3])>max(bb[1],g[1]) for g in block):raise ValueError('Admitted Chair-negative overlaps protected seating')
        owners=[rid for rid,g in region_boxes.items() if g[0]<=bb[0] and g[1]<=bb[1] and bb[2]<=g[2] and bb[3]<=g[3]]
        if not owners or not R:raise ValueError('Added query has no verified containing region')
        # Exact output signatures only;near-duplicates are reported,not merged.
        signature=(tuple(float(x) for x in b.detach()[q].cpu().tolist()),float(scores[q]));entry=clusters.setdefault(signature,dict(owner=min(owners),queries=[]))
        if entry['owner']!=min(owners):raise ValueError('Inconsistent duplicate ownership')
        entry['queries'].append(q)
    grouped=defaultdict(list)
    for c in clusters.values():grouped[c['owner']].append(c['queries'])
    for rid,items in grouped.items():
        for qs in items:out[qs]=1/(R*len(items)*len(qs))
    if (out[~added[:,1]]!=0).any() or not torch.isfinite(out).all():raise ValueError('Weight escaped admitted mask')
    meta=dict(group_region_count=R,admitted_queries=int(added.sum()),unique_query_signatures=len(clusters),duplicate_query_signatures_collapsed=int(added.sum())-len(clusters),regions=[dict(region_id=r['region_id'],aliases=r['aliases'],covered_signatures=len(grouped[r['region_id']]),query_indices=[q for qs in grouped[r['region_id']] for q in qs]) for r in regions],weights_sum=float(out.sum()),all_noncovered_regions_zero=True,overlapping_query_ownership='Lexicographically first canonical containing region;one query contributes once')
    return out.detach(),meta


def source_loss(z,w):
    if w.shape!=(len(z),) or w.requires_grad or (w<0).any() or not torch.isfinite(w).all():raise ValueError('Invalid detached region weights')
    return .75*(F.binary_cross_entropy_with_logits(z[:,1],torch.zeros_like(z[:,1]),reduction='none')*w).sum()


def branches(z,b,row,evidence,protected,context,policy=POLICY):
    """Default returns bit-identical C8 corrected objective;opt-in exposes branches."""
    if not policy.enabled:
        total,p,stats=baseline_objective(z,b,row,evidence,protected,True)
        return dict(baseline_total=total,baseline_parts=p,stats=stats,enabled=False)
    if policy!=Policy(enabled=True):raise ValueError('One fixed policy only;no weight sweep')
    old,oldparts,oldstats=baseline_objective(z,b,row,evidence,protected,False)
    losses,masks,stats=controlled_losses(z,b,row,evidence,protected)
    if not torch.equal(oldparts['positive'],losses['positive']) or not torch.equal(oldparts['legacy_negative'],losses['native_negative']):raise ValueError('Verified positive/native losses changed')
    w,meta=weights(z,b,row,masks,context,protected);raw=source_loss(z,w)
    return dict(enabled=True,base_total=old,positive_anchor=oldparts['positive'],native_negative=oldparts['legacy_negative'],source_raw=raw,existing_source_focal=losses['source_focal'],unweighted_source_BCE=losses['source_BCE'],box=losses['box'],masks=masks,weights=w,normalization=meta,stats=stats)


def norm(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))


def compose(base,positive,negative,positive_count,policy=POLICY):
    """Compose actual unscaled FP32 buffers after an entire accumulation group.

    No model mutation,optimizer,gradient projection or positive rescaling.
    Callers must measure class_head dense0+shift+scale gradients with C8 scope.
    """
    if not policy.enabled:raise PermissionError('Balanced gradient composition is disabled by default')
    if policy!=Policy(enabled=True):raise ValueError('Fixed policy only')
    if set(base)!=HEAD_KEYS or set(positive)!=CLASS_KEYS or set(negative)!=CLASS_KEYS or type(positive_count)is not int or positive_count<0:raise ValueError('Exact C8 twelve-head/six-class-head gradient scope required')
    for name,t in base.items():
        if t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('Finite unscaled FP32 gradients required')
        if name in CLASS_KEYS:
            for v in (positive[name],negative[name]):
                if v.shape!=t.shape or v.dtype!=torch.float32 or v.device!=t.device or not torch.isfinite(v).all():raise ValueError('Branch shape,dtype,device or finiteness mismatch')
    ordered=sorted(CLASS_KEYS)
    p=norm(positive[k] for k in ordered);n=norm(negative[k] for k in ordered);dot=sum(float((positive[k].detach().double()*negative[k].detach().double()).sum()) for k in ordered)
    if positive_count==0:
        if p:raise ValueError('Positive gradient present without verified positive targets')
        budget=policy.negative_only_fraction_of_clip*policy.global_clip_norm;mode='verified-negative-only:absolute1percent-of-original-clip'
    elif n==0:budget=policy.rho*p;mode='no-negative-gradient'
    elif p<=policy.positive_norm_floor:raise PositiveAnchorUnavailable('Verified positives exist but gradient anchor is absent/near zero;stop update,do not drop evidence or divide by tiny anchor')
    else:budget=policy.rho*p;mode='relative-to-all-verified-furniture-positives'
    scale=min(1.,budget/n) if n else 1.;used={k:negative[k].detach()*scale for k in ordered}
    if norm(used.values())>budget*(1+1e-6)+1e-12:raise ValueError('Numerical parameter-gradient budget violation')
    result={k:(base[k].detach().clone()+used[k] if k in CLASS_KEYS else base[k].detach().clone()) for k in base}
    return result,dict(scale=scale,positive_norm=p,raw_negative_norm=n,used_negative_norm=norm(used.values()),budget=budget,mode=mode,positive_negative_cosine=dot/(p*n) if p*n else None,positive_branch_unchanged=True,box_branch_unchanged=True,cap_space='Six trainable C8 classification-head parameter tensors,after accumulation,before unchanged global clipping/AdamW',optimizer_update_bound_not_claimed=True)


class Accumulator:
    """Preserve original scaled accumulation/unscale order,including last3.

    Each input VJP must differentiate (per-photo loss / microbatches) * scale,
    in that order,matching C8. Accumulate scaledFP32 buffers first;unscale once.
    """
    def __init__(self,microbatches,scale):
        if type(microbatches)is not int or not 1<=microbatches<=4 or not math.isfinite(scale) or scale<=0:raise ValueError('Invalid fixed accumulation/scaler contract')
        self.expected=microbatches;self.scale=scale;self.count=0;self.positive_count=0;self.base=None;self.positive=None;self.negative=None
    def add(self,base,positive,negative,positive_count):
        if self.count>=self.expected:raise ValueError('Extra microbatch would change ordering/budget')
        if set(base)!=HEAD_KEYS or set(positive)!=CLASS_KEYS or set(negative)!=CLASS_KEYS or type(positive_count)is not int or positive_count<0:raise ValueError('Gradient scope/count mismatch')
        for source in (base,positive,negative):
            if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in source.values()):raise ValueError('Scaled finiteFP32 buffers required')
        if self.base is None:
            self.base={k:v.detach().clone() for k,v in base.items()};self.positive={k:v.detach().clone() for k,v in positive.items()};self.negative={k:v.detach().clone() for k,v in negative.items()}
        else:
            for dest,source in ((self.base,base),(self.positive,positive),(self.negative,negative)):
                if any(dest[k].shape!=v.shape or dest[k].device!=v.device for k,v in source.items()):raise ValueError('Accumulation frame changed')
            for dest,source in ((self.base,base),(self.positive,positive),(self.negative,negative)):
                for k,v in source.items():dest[k].add_(v.detach())
        self.count+=1;self.positive_count+=positive_count
    def finish(self,policy=POLICY):
        if self.count!=self.expected:raise ValueError('Incomplete optimizer accumulation group')
        b={k:v/self.scale for k,v in self.base.items()};p={k:v/self.scale for k,v in self.positive.items()};n={k:v/self.scale for k,v in self.negative.items()};result,diag=compose(b,p,n,self.positive_count,policy)
        return result,dict(diag,actual_microbatches=self.count,scale=self.scale,positive_targets=self.positive_count,normalization_order='Each branch loss divided by actual group size beforeVJP;scaled buffers summed then unscaled once')
