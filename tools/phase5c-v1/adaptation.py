"""Isolated OWLv2 adaptation components; no optimizer step in this module.

Published partial class coverage masks classification. Objectness remains frozen:
unknown object categories make global background supervision unsafe in this pilot.
"""
import hashlib
import math
import torch
from torch import nn
import torch.nn.functional as F
from common import CLASSES

def configure(model,mode='heads',checkpointing=False):
 if mode!='heads' or checkpointing:raise ValueError('Phase 5C permits prepared frozen-backbone heads only')
 model.requires_grad_(False)
 for module in (model.class_head,model.box_head):module.requires_grad_(True)
 names=[n for n,p in model.named_parameters() if p.requires_grad]
 if not names or any(not n.startswith(('class_head.','box_head.')) for n in names):raise ValueError('Trainable scope violation')
 model.train();model.owlv2.text_model.eval()
 return names

def xyxy_to_cxcywh(b):return torch.cat(((b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]),-1)
def cxcywh_to_xyxy(b):return torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
def to_square(b,width,height):return b*b.new_tensor([width/max(width,height),height/max(width,height)]*2)
def from_square(b,width,height):return b/b.new_tensor([width/max(width,height),height/max(width,height)]*2)
def box_area(b):return (b[...,2:]-b[...,:2]).clamp(min=0).prod(-1)
def pair_iou(a,b):
 inter=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
 union=box_area(a)[:,None]+box_area(b)[None,:]-inter
 return inter/union.clamp(min=1e-8),union
def giou(a,b):
 iou,union=pair_iou(a,b);enclosure=(torch.maximum(a[:,None,2:],b[None,:,2:])-torch.minimum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
 return iou-(enclosure-union)/enclosure.clamp(min=1e-8)

def detection_loss(logits,boxes,row):
 """One image. Boxes in model's padded-square normalized cxcywh, class logits N x 7."""
 from scipy.optimize import linear_sum_assignment
 logits=logits.float();boxes=boxes.float()
 if logits.ndim!=2 or logits.shape[-1]!=7 or boxes.shape!=(len(logits),4) or not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():raise ValueError('Malformed/nonfinite detector outputs')
 annotations=[g for g in row['annotations'] if not g['ignore'] and row['coverage'][g['label']]]
 device=logits.device;target=torch.zeros_like(logits);valid=torch.tensor([row['coverage'][c] for c in CLASSES],device=device).expand_as(logits).clone()
 pred_xy=cxcywh_to_xyxy(boxes);match_pred=match_gt=None
 loss_box=boxes.sum()*0;loss_giou=boxes.sum()*0
 if annotations:
  gxy=to_square(torch.tensor([g['box'] for g in annotations],device=device,dtype=torch.float32),row['width'],row['height']);gc=xyxy_to_cxcywh(gxy)
  labels=torch.tensor([CLASSES.index(g['label']) for g in annotations],device=device)
  if len(logits)<len(annotations):raise ValueError('Insufficient prediction slots for GT')
  with torch.no_grad():
   cost=-logits.sigmoid()[:,labels]+5*torch.cdist(boxes,gc,p=1)+2*(1-giou(pred_xy,gxy))
   ii,jj=linear_sum_assignment(cost.detach().cpu().numpy());match_pred=torch.as_tensor(ii,device=device);match_gt=torch.as_tensor(jj,device=device)
  target[match_pred,labels[match_gt]]=1
  loss_box=F.l1_loss(boxes[match_pred],gc[match_gt],reduction='sum')/len(annotations)
  loss_giou=(1-giou(pred_xy[match_pred],gxy[match_gt]).diagonal()).mean()
 # Group/depiction areas and padded regions are unknown background, never negatives.
 wh=boxes.new_tensor([row['width']/max(row['width'],row['height']),row['height']/max(row['width'],row['height'])])
 centers=boxes[:,:2].detach();outside=(centers<0).any(-1)|(centers>wh).any(-1);valid[outside]=False
 for g in row['annotations']:
  if not g['ignore']:continue
  ignored=to_square(torch.tensor(g['box'],device=device),row['width'],row['height']);inside=((centers>=ignored[:2])&(centers<=ignored[2:])).all(-1);valid[inside]=False
 if match_pred is not None:valid[match_pred,labels[match_gt]]=True
 prob=logits.sigmoid();ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none');pt=prob*target+(1-prob)*(1-target);alpha=.25*target+.75*(1-target)
 cls=(ce*(1-pt).pow(2)*alpha*valid).sum()/max(len(annotations),1)
 total=cls+5*loss_box+2*loss_giou
 if not torch.isfinite(total):raise ValueError('Nonfinite detection loss')
 return total,dict(classification=float(cls.detach()),l1=float(loss_box.detach()),giou=float(loss_giou.detach()),matched=len(annotations),known_classes=sum(row['coverage'].values()),objectness_supervised=False)

def trainable_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def trainable_state(model):return {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
def load_delta(model,state):
 expected={n:p for n,p in model.named_parameters() if p.requires_grad}
 if set(state)!=set(expected):raise ValueError('Adaptation checkpoint key mismatch')
 with torch.no_grad():
  for name,value in state.items():
   if value.shape!=expected[name].shape or not torch.isfinite(value).all():raise ValueError('Adaptation tensor invalid')
   expected[name].copy_(value)
