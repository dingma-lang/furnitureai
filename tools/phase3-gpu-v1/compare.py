"""One controlled comparison after checkpoint selection is frozen. Never tunes models."""
import argparse
import collections as C
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import train as t


def product_records(rows,logits):
 groups=C.defaultdict(list)
 for i,r in enumerate(rows):groups[r['abo_product_id']].append(i)
 result=[]
 for pid,ids in sorted(groups.items()):
  truth={rows[i]['weak_label'] for i in ids}
  if len(truth)!=1:raise ValueError('Frozen product labels conflict')
  result.append({'product_id':pid,'group':rows[ids[0]]['leakage_group_id'],'truth':t.CLASSES.index(next(iter(truth))),
   'prediction':int(logits[ids].mean(axis=0).argmax())})
 return result


def detailed(truth,pred):
 cm=np.zeros((7,7),dtype=int)
 for a,b in zip(truth,pred):cm[a,b]+=1
 report={}
 for i,cls in enumerate(t.CLASSES):
  tp=int(cm[i,i]);n=int(cm[i].sum());positive=int(cm[:,i].sum());precision=tp/positive if positive else 0.;recall=tp/n if n else None
  report[cls]={'support':n,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.}
 metrics=t.w.helper().label_metrics(list(truth),list(pred));metrics.update(confusion_matrix=cm.tolist(),per_class=report)
 return metrics


def paired_bootstrap(records,baseline='baseline',repeats=1000):
 keys=list(records);reference=records[baseline];ids=[r['product_id'] for r in reference]
 if any([r['product_id'] for r in records[k]]!=ids for k in keys):raise ValueError('Comparison examples differ')
 groups=sorted({r['group'] for r in reference});rng=np.random.default_rng(t.SEED);samples={k:[] for k in keys};deltas={k:[] for k in keys if k!=baseline};invalid=0
 for _ in range(repeats):
  drawn=rng.choice(groups,len(groups),replace=True);indices=[i for group in drawn for i,r in enumerate(reference) if r['group']==group]
  truth=[reference[i]['truth'] for i in indices]
  if set(truth)!=set(range(7)):invalid+=1;continue
  scores={k:t.w.helper().label_metrics(truth,[records[k][i]['prediction'] for i in indices])['macro_f1'] for k in keys}
  for k in keys:samples[k].append(scores[k])
  for k in deltas:deltas[k].append(scores[k]-scores[baseline])
 def interval(values):return list(map(float,np.quantile(values,[.025,.975]))) if len(values)>=100 else None
 return {'seed':t.SEED,'repeats':repeats,'invalid_missing_category':invalid,'valid':repeats-invalid,
  'macro_f1_intervals':{k:interval(v) for k,v in samples.items()},'paired_delta_vs_baseline_intervals':{k:interval(v) for k,v in deltas.items()},
  'scope':'conditional group-bootstrap uncertainty for weak labels on this previously exposed panel, not independent real-world accuracy'}


def compare(out):
 t.validate(out)
 if not (out/'queue-complete.json').exists() or not (out/'selection-frozen.json').exists():raise ValueError('Training and selection must be complete before panel access')
 registry=t.read(out/'comparison-checkpoints-frozen.json')
 if registry['selection_sha256']!=t.sha(out/'selection-frozen.json'):raise ValueError('Frozen selection changed')
 for arm in ('baseline','selected_expanded','expanded_cosine_control'):
  if t.sha(out/registry[arm]['path'])!=registry[arm]['sha256']:raise ValueError('Selected checkpoint changed')
 directory=out/'comparison';directory.mkdir(exist_ok=True)
 binding={'registry_sha256':t.sha(out/'comparison-checkpoints-frozen.json'),'selection_sha256':t.sha(out/'selection-frozen.json'),
  'code_sha256':t.sha(__file__),'protocol_sha256':t.sha(t.PREPARATION/'frozen-development-protocol.json'),
  'panel_sha256':t.read(out/'input-lock.json')['frozen_panel_sha256_from_prior_preparation']}
 if (directory/'result.json').exists():
  if t.read(directory/'result.json')['binding']!=binding:raise ValueError('Existing comparison bound to other inputs')
  print('One-shot comparison already complete; no scoring repeated');return
 intent=directory/'intent.json'
 if intent.exists():
  if t.read(intent)!=binding:raise ValueError('Comparison intent changed; no new tuning allowed')
 else:t.save(intent,binding)
 # First semantic access to the frozen panel matrix happens only here, after full selection freeze.
 panel=t.PREPARATION/'frozen-development-features.npy'
 if t.sha(panel)!=binding['panel_sha256']:raise ValueError('Frozen panel hash mismatch')
 array=np.load(panel,allow_pickle=False);rows=t.read(t.PREPARATION/'frozen-development-protocol.json')['development_rows']
 if array.shape!=(106,768) or len(rows)!=106 or len({r['abo_product_id'] for r in rows})!=45:raise ValueError('Comparison population mismatch')
 results={};records={};logits_by_checkpoint={}
 for arm in ('baseline','selected_expanded','expanded_cosine_control'):
  item=registry[arm];prediction_path=directory/(arm+'-logits.pt')
  if prediction_path.exists():
   saved=torch.load(prediction_path,map_location='cpu',weights_only=True)
   if saved['binding']!=binding or saved['checkpoint_sha256']!=item['sha256']:raise ValueError('Cached comparison predictions changed binding')
   logits=saved['logits'].numpy()
  elif item['sha256'] in logits_by_checkpoint:
   logits=logits_by_checkpoint[item['sha256']]
   t.w.save_tensor(prediction_path,{'binding':binding,'checkpoint_sha256':item['sha256'],'logits':torch.from_numpy(logits)})
  else:
   saved=torch.load(out/item['path'],map_location='cpu',weights_only=True);model=t.w.network(saved['config'],768);model.load_state_dict(saved['state_dict']);model.eval()
   if {r['abo_product_id'] for r in rows}&set(saved['train_product_ids']):raise ValueError('Comparison product appeared in model gradients')
   with torch.inference_mode():logits=model(torch.from_numpy(array)).numpy()
   t.w.save_tensor(prediction_path,{'binding':binding,'checkpoint_sha256':item['sha256'],'logits':torch.from_numpy(logits)})
  logits_by_checkpoint[item['sha256']]=logits
  rec=product_records(rows,logits);records[arm]=rec
  results[arm]={'checkpoint':item,'product_level':detailed([r['truth'] for r in rec],[r['prediction'] for r in rec]),
   'image_level':detailed([t.CLASSES.index(r['weak_label']) for r in rows],logits.argmax(axis=1).tolist()),'products':rec}
 result={'binding':binding,'scope':'Single frozen-panel comparison after training-only selection; provisional weak-label agreement, not verified accuracy',
  'panel_images':106,'panel_products':45,'arms':results,'uncertainty':paired_bootstrap(records),'official_holdout_scoring':False,
  'panel_used_for_current_tuning':False,'panel_previously_exposed_in_historical_work':True,'no_new_model_selection_after_this_result':True}
 t.save(directory/'result.json',result);t.preserve_check(out)
 print(json.dumps({k:v['product_level']['macro_f1'] for k,v in results.items()},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=t.OUT);args=parser.parse_args();compare(args.output.resolve())
