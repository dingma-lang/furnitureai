"""Head/loss/loader building blocks only. No model construction or optimizer calls."""
import torch
import torch.nn.functional as F
from common import CLASSES, OUT, ROOT, read, sha
from policy import SUPPORTED

HEAD_PARAMETERS = 1_579_526
HEAD_KEYS = {f'{head}.{layer}.{suffix}'
             for head,layers in [('class_head',('dense0','logit_shift','logit_scale')),
                                 ('box_head',('dense0','dense1','dense2'))]
             for layer in layers for suffix in ('weight','bias')}

def configure_heads(model, expected_count=HEAD_PARAMETERS):
    model.requires_grad_(False)
    model.eval()                         # Frozen backbone remains in eval mode.
    for module in (model.class_head, model.box_head):
        module.requires_grad_(True); module.train()
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if set(names) != HEAD_KEYS:
        raise ValueError('Trainable scope violation')
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != expected_count:
        raise ValueError('Unexpected head parameter count')
    return names

def load_delta(model, payload):
    expected = {n:p for n,p in model.named_parameters() if p.requires_grad}
    weights = payload['weights']
    if not expected or set(weights)!=set(expected): raise ValueError('Head delta key mismatch')
    for n,v in weights.items():
        if v.shape!=expected[n].shape or not torch.isfinite(v).all(): raise ValueError('Invalid head tensor')
    with torch.no_grad():
        for n,v in weights.items(): expected[n].copy_(v)

def load_training_row(row, partition, manifest=None):
    if partition not in ('optimization','inner_development'): raise PermissionError('No external development/frozen image loader')
    manifest = read(OUT/'prepared-manifest.json') if manifest is None else manifest
    exact = {r['image_id']:r for r in manifest['rows']}
    if row != exact.get(row['image_id']) or row['internal_partition']!=partition or row['historical_partition']!='train':
        raise PermissionError('Not an exact prepared training-only member')
    p = (ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']: raise ValueError('Image provenance changed')
    # No image body is decoded by the preflight; a future separately authorized worker may use this path.
    return p

def to_square(box, width, height):
    if min(width,height)<=0: raise ValueError('Invalid coordinate frame')
    return box*box.new_tensor([width/max(width,height),height/max(width,height)]*2)

def from_square(box, width, height):
    if min(width,height)<=0: raise ValueError('Invalid coordinate frame')
    return box/box.new_tensor([width/max(width,height),height/max(width,height)]*2)

def xyxy_to_cxcywh(b): return torch.cat(((b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]),-1)
def cxcywh_to_xyxy(b): return torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)

def giou(a,b):
    area = lambda x:(x[...,2:]-x[...,:2]).clamp(min=0).prod(-1)
    overlap = (torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
    union = area(a)[:,None]+area(b)[None,:]-overlap
    enclosing = (torch.maximum(a[:,None,2:],b[None,:,2:])-torch.minimum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
    return overlap/union.clamp(min=1e-8)-(enclosing-union)/enclosing.clamp(min=1e-8)

def supervision(logits, boxes, row, match_pred=None, labels=None):
    target = torch.zeros_like(logits)
    negatives = row['explicit_native_negative']
    if any(negatives.get(c) for c in CLASSES if c not in SUPPORTED): raise ValueError('Unresolved class negative')
    valid = torch.tensor([negatives.get(c,False) for c in CLASSES],device=logits.device,dtype=torch.bool).expand_as(logits).clone()
    centers = boxes[:,:2].detach()
    wh = boxes.new_tensor([row['width']/max(row['width'],row['height']),row['height']/max(row['width'],row['height'])])
    valid[(centers<0).any(-1)|(centers>wh).any(-1)] = False
    for a in row['annotations']:
        if a['ignore']:
            b = to_square(boxes.new_tensor(a['box']),row['width'],row['height'])
            valid[((centers>=b[:2])&(centers<=b[2:])).all(-1)] = False
    if match_pred is not None:
        if any(negatives.get(CLASSES[int(c)]) for c in labels): raise ValueError('Positive/negative contradiction')
        target[match_pred,labels] = 1
        valid[match_pred,labels] = True
    return target,valid

def detection_loss(logits, boxes, row):
    """Synthetic CPU-testable loss; unknown classes/instances never become background."""
    from scipy.optimize import linear_sum_assignment
    logits,boxes = logits.float(),boxes.float()
    if logits.ndim!=2 or logits.shape[1]!=7 or boxes.shape!=(len(logits),4): raise ValueError('Prediction dimensions')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all(): raise ValueError('Nonfinite detector output')
    annotations = [a for a in row['annotations'] if not a['ignore']]
    if any(a['label'] not in SUPPORTED for a in annotations): raise ValueError('Unresolved project target entered loss')
    loss_box = boxes.sum()*0; loss_giou = boxes.sum()*0; ii = labels = None
    if annotations:
        if len(logits)<len(annotations): raise ValueError('Insufficient prediction slots')
        gxy = to_square(boxes.new_tensor([a['box'] for a in annotations]),row['width'],row['height'])
        gc = xyxy_to_cxcywh(gxy)
        gt_labels = torch.tensor([CLASSES.index(a['label']) for a in annotations],device=logits.device)
        with torch.no_grad():
            cost = -logits.sigmoid()[:,gt_labels]+5*torch.cdist(boxes,gc,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gxy))
            a,b = linear_sum_assignment(cost.cpu().numpy())
            ii = torch.as_tensor(a,device=logits.device); jj = torch.as_tensor(b,device=logits.device)
            labels = gt_labels[jj]
        loss_box = F.l1_loss(boxes[ii],gc[jj],reduction='sum')/len(annotations)
        loss_giou = (1-giou(cxcywh_to_xyxy(boxes[ii]),gxy[jj]).diagonal()).mean()
    target,valid = supervision(logits,boxes,row,ii,labels)
    p = logits.sigmoid(); pt = p*target+(1-p)*(1-target)
    cls = (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(2)*(.25*target+.75*(1-target))*valid).sum()/max(len(annotations),1)
    total = cls+5*loss_box+2*loss_giou
    if not torch.isfinite(total): raise ValueError('Nonfinite loss')
    return total,dict(matched=len(annotations),valid_class_terms=int(valid.sum()),objectness_supervised=False,
                     unknowns_as_background=False,instance_mask_loss=False)
