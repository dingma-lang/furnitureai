"""Conservative source-native comparison mapping and no-training loss-mask prototype."""
import math

OI={'/m/02crq1':'sofa','/m/01mzpv':'chair','/m/04bcr3':'table','/m/03ssj5':'bed','/m/01y9k5':'desk','/m/01s105':'cabinet','/m/0dtln':'lamp'}
COCO={62:'chair',63:'sofa',65:'bed',67:'table'}
LVIS={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp',
 19:'chair',170:'bed',285:'table',367:'table',438:'cabinet',464:'chair',610:'table',889:'chair',1051:'lamp'}
PARTS={390,628};OUTDOOR={627};AMBIGUOUS={301,346,1158,738,480,233}

def mapping(dataset,native_id,unit_confirmed=False):
 if dataset not in ('OpenImages','COCO','LVIS'):raise ValueError('Unknown native namespace')
 table={'OpenImages':OI,'COCO':COCO,'LVIS':LVIS}[dataset];group=table.get(native_id)
 if dataset=='LVIS' and native_id in PARTS:return dict(comparison_group=None,scope='part_not_whole_furniture',project_positive_allowed=False)
 if dataset=='LVIS' and native_id in OUTDOOR:return dict(comparison_group=None,scope='outdoor_not_indoor_lamp',project_positive_allowed=False)
 if dataset=='LVIS' and native_id in AMBIGUOUS:return dict(comparison_group=None,scope='ambiguous_subtype_kept_native',project_positive_allowed=False)
 if group is None:return dict(comparison_group=None,scope='unsupported_keep_native',project_positive_allowed=False)
 risky=group in ('table','desk','cabinet','lamp') or dataset=='LVIS' and native_id not in (982,232,1050,77,361,181,626)
 return dict(comparison_group=group,native_dataset=dataset,native_id=native_id,
  scope='native_only_unit_policy_required' if risky and not unit_confirmed else 'source_positive_comparison_group',
  project_positive_allowed=bool(unit_confirmed or not risky),
  floor_lamp_verified=False,project_absence_inferred=False,
  caveat='Grouping is descriptive. A confirmed native unit is not human review or a verified project-negative grant.')

def native_scope(dataset,native_id,has_positive,raw_scope):
 if dataset=='LVIS':
  neg=set(raw_scope.get('neg_category_ids',[]));nex=set(raw_scope.get('not_exhaustive_category_ids',[]))
  if native_id in neg and (has_positive or native_id in nex):raise ValueError('Contradictory native scope')
  if native_id in neg:return 'explicit_native_negative'
  if has_positive:return 'positive_nonexhaustive' if native_id in nex else 'positive_source_exhaustive'
  return 'unknown'
 if dataset=='OpenImages':
  labels=raw_scope.get('native_image_level_labels',{})
  if native_id in labels and labels[native_id]==0:
   if has_positive:raise ValueError('OI positive/negative contradiction')
   return 'explicit_native_negative'
  if has_positive:return 'positive_source_coverage' # Does not assert seven-class or hierarchy completeness.
  return 'unknown'
 if dataset=='COCO':return 'positive_completeness_unverified' if has_positive else 'unknown'
 raise ValueError('Unknown dataset')

def loss_mask(dataset,native_id,scope,matched=False,has_mask=False,ignored=False,group_of=False,
              unit_ambiguous=False,head_space='native',allow_exhaustive_native_negatives=False,
              all_native_targets_retained=False):
 if head_space not in ('native','project'):raise ValueError('Unknown head scope')
 known={'unknown','explicit_native_negative','positive_nonexhaustive','positive_source_exhaustive','positive_source_coverage','positive_completeness_unverified'}
 if scope not in known:raise ValueError('Unknown annotation-completeness state')
 plan=dict(positive_class=False,negative_class=False,box=False,mask=False,background_objectness=False,
  dataset=dataset,native_id=native_id,head_space=head_space,scope=scope)
 m=mapping(dataset,native_id)
 if ignored or group_of or unit_ambiguous or m['comparison_group'] is None:return plan
 if matched:
  if scope in ('explicit_native_negative','unknown'):raise ValueError('Matched positive conflicts with source scope')
  if head_space=='project' and not m['project_positive_allowed']:return plan
  plan.update(positive_class=True,box=True,mask=has_mask)
 elif scope=='explicit_native_negative' and head_space=='native':plan['negative_class']=True
 elif scope=='positive_source_exhaustive' and head_space=='native' and allow_exhaustive_native_negatives and all_native_targets_retained:
  plan['negative_class']=True
 # Opt-in exhaustive native negatives require an explicit complete-target assertion. Never broaden them to a project alias.
 return plan

def masked_binary_loss(logits,targets,enabled):
 """CPU scalar reference only; no model, gradients, optimiser or training invocation."""
 if not len(logits)==len(targets)==len(enabled):raise ValueError('Loss-vector dimensions differ')
 terms=[]
 for logit,target,use in zip(logits,targets,enabled):
  if not use:continue
  if target not in (0,1) or not math.isfinite(logit):raise ValueError('Invalid enabled loss input')
  terms.append(max(logit,0)-logit*target+math.log1p(math.exp(-abs(logit))))
 return sum(terms)/len(terms) if terms else 0.

def pixel_xyxy_view(dataset,native_box,width,height):
 if width<=0 or height<=0 or len(native_box)!=4 or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in native_box):raise ValueError('Invalid box metadata')
 if dataset=='OpenImages':
  x1,y1,x2,y2=native_box;result=[x1*width,y1*height,x2*width,y2*height]
 elif dataset in ('COCO','LVIS'):
  x,y,w,h=native_box;result=[x,y,x+w,y+h]
 else:raise ValueError('Unknown coordinate namespace')
 x1,y1,x2,y2=result
 if not (0<=x1<x2<=width+.02 and 0<=y1<y2<=height+.02):raise ValueError('Source box outside its declared coordinate frame')
 return result # Derived view only. Never clip/overwrite native coordinates.
