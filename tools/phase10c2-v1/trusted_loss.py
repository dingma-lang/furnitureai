"""Opt-in trusted Chair negatives; default is byte-preserved Phase10B behavior.

No model constructor, forward, optimizer step or training worker is provided.
Only CPU synthetic tensors are used to validate this module in Phase10C.2.
"""
import sys
import math
import torch
import torch.nn.functional as F
from n2 import ROOT,RAW,CLASSES,digest,optimizer_member,read,sha
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from expanded import detection_loss as legacy_loss,supervision,to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou

DEFAULT=dict(mode='phase10b',enabled=False,validated=False,approved=False)

def box_valid(b):
    return len(b)==4 and all(isinstance(x,(float,int)) and math.isfinite(x) for x in b) and 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1

def evidence_valid(record,row):
    if not optimizer_member(row):raise PermissionError('Only immutable optimization members can acquire training negatives')
    if any(record.get(k)!=row.get(k) for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')):raise ValueError('Negative evidence identity/partition/hash binding')
    dataset=next((a['native_dataset'] for a in row['annotations']),row.get('source_native_dataset'))
    native={'OpenImages':'/m/01mzpv','COCO':62,'LVIS':232}.get(dataset)
    if native is None or record.get('category')!='chair' or record.get('native_category')!=native:raise ValueError('Native category lineage changed')
    if record.get('native_annotation_digest')!=digest(row['annotations']):raise ValueError('Annotation lineage changed after human review')
    if not(record.get('human_reviewed') is True and record.get('confirmed') is True and record.get('reviewer') and record.get('review_date') and record.get('evidence','').strip() and record.get('trust_basis')=='human_category_scope_review'):
        raise PermissionError('No assistant or missing human confirmations may activate negatives')
    from datetime import date
    date.fromisoformat(record['review_date'])
    if not record['reviewer'].strip() or len(record['evidence'].strip())<8:raise ValueError('Reviewer and concrete evidence required')
    kind=record.get('kind')
    if kind not in ('full_image_absent','complete_with_positives','negative_region'):raise ValueError('Unknown negative evidence kind')
    if kind in ('full_image_absent','complete_with_positives') and record.get('complete_for_chair') is not True:raise ValueError('Whole-image completeness not confirmed')
    potential=record.get('potential_unannotated_chair_regions',[])
    if any(not box_valid(b) for b in potential):raise ValueError('Malformed potential-positive region')
    if potential and kind!='negative_region':raise PermissionError('Potential missing Chairs contradict whole-image completeness')
    if kind=='full_image_absent' and any(a['label']=='chair' for a in row['annotations']):raise ValueError('Absence contradicts native Chair annotations')
    if kind=='negative_region' and (not record.get('regions') or not all(box_valid(b) for b in record['regions'])):raise ValueError('Safe normalized negative regions required')
    return True

def authorized(config,evidence,row):
    if not(config.get('enabled') is True and config.get('validated') is True and config.get('approved') is True and config.get('approval_reference')):raise PermissionError('Corrected mode disabled until validated and separately approved')
    if config.get('mode')!='trusted_chair_negative_v1' or config.get('legacy_weights')!=dict(classification=1,l1=5,giou=2,gamma=2,positive_alpha=.25,negative_alpha=.75):raise ValueError('No simultaneous coefficient/normalization change')
    if config.get('evidence_sha256')!=digest(evidence):raise ValueError('Evidence bundle hash changed')
    if config.get('row_bindings',{}).get(row['image_id'])!=digest(row):raise ValueError('Prepared immutable row changed')
    selected=[e for e in evidence if e.get('image_id')==row['image_id']]
    for e in selected:evidence_valid(e,row)
    return selected

def validate_bundle(config,evidence,prepared_manifest):
    """Mandatory future-worker preflight; no new photographs or group movement."""
    if config.get('dataset_sha256')!=digest(prepared_manifest):raise ValueError('Dataset content binding changed')
    index={r['image_id']:r for r in prepared_manifest['rows']}
    train=[r for r in index.values() if optimizer_member(r)];other=[r for r in index.values() if not optimizer_member(r)]
    for key in ('sha256','identity_group'):
        if {r[key] for r in train}&{r[key] for r in other}:raise PermissionError('Training/evaluation identity overlap')
    seen=set()
    for e in evidence:
        key=(e['image_id'],e['category'])
        if key in seen:raise ValueError('Conflicting/redundant category evidence')
        seen.add(key)
        if e['image_id'] not in index:raise PermissionError('Unapproved new source image')
        evidence_valid(e,index[e['image_id']])
        authorized(config,evidence,index[e['image_id']])
    if not evidence:raise PermissionError('No trusted Chair evidence: A/B would be an identity comparison')
    return True

def load_configuration(path):
    """Read a separately versioned future config; never edit or enable it."""
    c=read(path)
    if c.get('mode')=='trusted_chair_negative_v1':
        p=(ROOT/c['row_bindings_path']).resolve()
        if not p.is_relative_to(RAW) or sha(p)!=c['row_bindings_sha256']:raise ValueError('Row binding provenance changed')
        c=dict(c,row_bindings=read(p))
    return c

def negative_mask(logits,boxes,row,matched,labels,evidence):
    """Preserve old masks, then add only fully contained, protected Chair negatives.

    No regression of native positives; no global background interpretation.
    Any overlap with known Chair or ignored/uncertain native boxes prevents a
    new Chair-negative target. This intentionally leaves duplicates/fragments
    inside known positives unsupervised rather than declaring Chair absence.
    """
    target,valid=supervision(logits,boxes,row,matched,labels)
    added=torch.zeros_like(valid);chair=CLASSES.index('chair');xy=cxcywh_to_xyxy(boxes.detach())
    frame=to_square(boxes.new_tensor([0.,0.,1.,1.]),row['width'],row['height'])
    safe=(xy[:,:2]>=frame[:2]).all(-1)&(xy[:,2:]<=frame[2:]).all(-1)&(xy[:,2:]>xy[:,:2]).all(-1)
    for ann in row['annotations']:
        if ann['label']=='chair' or ann['ignore']:
            b=to_square(boxes.new_tensor(ann['box']),row['width'],row['height'])
            overlap=(torch.minimum(xy[:,2:],b[2:])-torch.maximum(xy[:,:2],b[:2])).clamp(min=0).prod(-1)>0
            safe&=~overlap
    permitted=torch.zeros(len(boxes),device=boxes.device,dtype=torch.bool)
    for e in evidence:
        evidence_valid(e,row)
        for b in e.get('potential_unannotated_chair_regions',[]):
            region=to_square(boxes.new_tensor(b),row['width'],row['height'])
            safe&=~((torch.minimum(xy[:,2:],region[2:])-torch.maximum(xy[:,:2],region[:2])).clamp(min=0).prod(-1)>0)
        regions=e['regions'] if e['kind']=='negative_region' else [[0.,0.,1.,1.]]
        for b in regions:
            region=to_square(boxes.new_tensor(b),row['width'],row['height'])
            permitted|=(xy[:,:2]>=region[:2]).all(-1)&(xy[:,2:]<=region[2:]).all(-1)
    added[:,chair]=safe&permitted&~valid[:,chair]
    valid|=added
    # Positive targets always win; a trusted absence contradiction was rejected above.
    if matched is not None:target[matched,labels]=1;valid[matched,labels]=True;added[matched,labels]=False
    return target,valid,added

def loss(logits,boxes,row,config=None,evidence=None):
    config=DEFAULT if config is None else config;evidence=[] if evidence is None else evidence
    if config.get('mode')=='phase10b':return legacy_loss(logits,boxes,row)
    selected=authorized(config,evidence,row)
    from scipy.optimize import linear_sum_assignment
    logits,boxes=logits.float(),boxes.float()
    if logits.ndim!=2 or logits.shape[1]!=7 or boxes.shape!=(len(logits),4) or not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():raise ValueError('Malformed/nonfinite detector tensors')
    anns=[a for a in row['annotations'] if not a['ignore']]
    if any(a['label'] not in ('sofa','chair','bed') for a in anns):raise ValueError('Ambiguous native class cannot enter loss')
    for a in anns:
        if not box_valid(a['box']):raise ValueError('Malformed native positive box')
    l1=boxes.sum()*0;overlap=boxes.sum()*0;ii=labels=None
    if anns:
        if len(logits)<len(anns):raise ValueError('Insufficient queries')
        gt=to_square(boxes.new_tensor([a['box'] for a in anns]),row['width'],row['height']);gc=xyxy_to_cxcywh(gt)
        native_labels=torch.tensor([CLASSES.index(a['label']) for a in anns],device=boxes.device)
        with torch.no_grad():
            cost=-logits.sigmoid()[:,native_labels]+5*torch.cdist(boxes,gc,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gt))
            a,b=linear_sum_assignment(cost.cpu().numpy());ii=torch.as_tensor(a,device=boxes.device);jj=torch.as_tensor(b,device=boxes.device);labels=native_labels[jj]
        l1=F.l1_loss(boxes[ii],gc[jj],reduction='sum')/len(anns);overlap=(1-giou(cxcywh_to_xyxy(boxes[ii]),gt[jj]).diagonal()).mean()
    target,valid,added=negative_mask(logits,boxes,row,ii,labels,selected)
    p=logits.sigmoid();pt=p*target+(1-p)*(1-target)
    elements=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(2)*(.25*target+.75*(1-target))
    denominator=max(len(anns),1);classification=(elements*valid).sum()/denominator
    total=classification+5*l1+2*overlap
    if not torch.isfinite(total):raise ValueError('Nonfinite loss')
    return total,dict(classification=float(classification.detach()),new_trusted_chair_negative=float((elements*added).sum().detach()/denominator),
        l1=float(l1.detach()),giou=float(overlap.detach()),normalization_native_GT=denominator,matched=len(anns),
        valid_positive_terms=int((target.bool()&valid).sum()),valid_negative_terms=int((~target.bool()&valid).sum()),
        added_chair_negative_terms=int(added.sum()),valid_terms_by_class={c:int(valid[:,i].sum()) for i,c in enumerate(CLASSES)},
        objectness_supervised=False,unknowns_as_background=False,mask_digest=digest(valid.detach().cpu().tolist()))
