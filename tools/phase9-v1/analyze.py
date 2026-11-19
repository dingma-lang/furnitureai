"""Fixed recipe evaluation, precision/recall trade-offs and locked selection."""
import argparse
import time
from core import *
from data import exposed_rows,final_rows_metadata
from fusion import method,consensus
from metrics import evaluate
from infer import caches
from siglip import record,should_abstain

RECIPES=('baseline','scale_nms','scale_agreement','scale_weighted','tiled_guarded','flip_agreement')

def needed(arm,recipe):
 v=['full960']
 if recipe.startswith('scale_'):v+=['full768','full1152']
 elif recipe=='tiled_guarded':v+=['tile0','tile1','tile2','tile3']
 elif recipe=='flip_agreement':v+=['flip960']
 return {arm:v}

def sensitivity(m):
 c=[v for k,v in m['categories'].items() if k not in ('cabinet','desk')];tp=sum(v['tp'] for v in c);fp=sum(v['fp'] for v in c);fn=sum(v['fn'] for v in c)
 return dict(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,ap50=sum(v['ap50'] for v in c if v['ap50'] is not None)/sum(v['ap50'] is not None for v in c),ap75=sum(v['ap75'] for v in c if v['ap75'] is not None)/sum(v['ap75'] is not None for v in c))

def evaluate_arm(part,name,rows):
 predictions={};seconds=[];peaks=[];passes=[];processing=[];diagnostics={}
 for r in rows:
  stage=time.monotonic()
  if name=='crossmodel_agreement':
   a=caches(part,'original',r)['full960'];b=caches(part,'adapted',r)['full960'];p,d=consensus({'original':a['predictions'],'adapted':b['predictions']},weighted=True);used=[a,b]
  else:
   arm,recipe=name.split('-',1);cache=caches(part,arm,r)
   if recipe=='siglip_consistency':
    used=[cache['full960']];check=record(part,arm,r);byindex={v['index']:v for v in check['checks']};p=[];rejected=[]
    for j,q in enumerate(used[0]['predictions']):
     if should_abstain(q,byindex[j]):rejected.append(j)
     else:p.append(q)
    d=dict(abstained_indices=rejected,source_predictions_preserved=True,scored_crops=sum(x['status']=='category_scored' for x in check['checks']))
    used=used+[dict(seconds=check['seconds'],peak_allocated_bytes=check['peak_allocated_bytes'])]
   else:
    p,d=method(recipe,{k:v['predictions'] for k,v in cache.items()});used=[cache[v] for v in needed(arm,recipe)[arm]]
  processing.append(time.monotonic()-stage);predictions[r['image_id']]=p;diagnostics[r['image_id']]=d
  seconds.append(sum(v['seconds'] for v in used));peaks.append(max(v['peak_allocated_bytes'] for v in used));passes.append(len(used))
 m=evaluate(rows,predictions);m['exclude_desk_cabinet_sensitivity']=sensitivity(m)
 m.update(mean_inference_pipeline_seconds=sum(seconds)/len(seconds),mean_analysis_seconds=sum(processing)/len(processing),peak_allocated_bytes=max(peaks),
  mean_stage_passes=sum(passes)/len(passes),latency_scope='Sum of measured shared view/verification stages, excludes model loading and final gallery/scoring; analysis timer includes cache reads and fusion',
  name=name,method_diagnostics=diagnostics)
 sealed(OUT/'evaluations'/part/(name+'.json'),dict(metrics=m,predictions=predictions));return m

def qualifies(m,reference):
 failures=[]
 if (m['precision'] or 0)<reference['precision']+.03:failures.append('precision gain below3pp')
 if m['tp']<reference['tp']-1:failures.append('recall loss more than one annotated object')
 if m['ap75']<reference['ap75']-.02:failures.append('AP75 loss more than2pp')
 if any(m['categories'][c]['tp']<reference['categories'][c]['tp'] for c in ('sofa','chair','lamp')):failures.append('rare-category true-positive loss')
 return failures

def run(part='exposed'):
 rows=exposed_rows() if part=='exposed' else final_rows_metadata()
 if part=='exposed':names=[a+'-'+m for a in ('original','adapted') for m in RECIPES]+['crossmodel_agreement']+[a+'-siglip_consistency' for a in ('original','adapted')]
 else:
  selection=checked(OUT/'selection.json')
  if selection['code_hashes']!=code_hashes():raise ValueError('Code changed after final selection')
  names=list(dict.fromkeys(['original-baseline','adapted-baseline',selection['selected']]))
 results={}
 for name in names:
  p=OUT/'evaluations'/part/(name+'.json');prior=checked(p)
  results[name]=prior['metrics'] if prior else evaluate_arm(part,name,rows)
  m=results[name];print(part,name,'P/R/AP50/AP75',*[round(100*m[k],2) if m[k] is not None else None for k in ('precision','recall','ap50','ap75')], 'TP/FP/FN',m['tp'],m['fp'],m['fn'],flush=True)
 if part=='exposed':
  baseline=results['adapted-baseline'];decisions={name:qualifies(m,baseline) for name,m in results.items() if not name.endswith('baseline')}
  candidates=[m for n,m in results.items() if n in decisions and not decisions[n]]
  winner=max(candidates,key=lambda m:(m['precision'],m['ap75'],m['f1'],m['name'])) if candidates else baseline
  name=winner['name'];required={'original':['full960'],'adapted':['full960']}
  if name!='crossmodel_agreement':
   arm,recipe=name.split('-',1)
   for a,v in needed(arm,recipe).items():required[a]=list(dict.fromkeys(required[a]+v))
  selected=dict(selected=name,qualification_failures=decisions,robust_improvement_on_exposed_protocol=bool(candidates),reference='adapted-baseline',
   required_views=required,SigLIP_final_required=name.endswith('siglip_consistency'),code_hashes=code_hashes(),
   fresh_manifest_sha256=sha(OUT/'fresh-metadata-lock.json'),final_comparisons_max=1,
   selected_before_any_fresh_pixels=True,no_public_or_human_verified_accuracy_claim=True,selection_utc=now())
  if not (OUT/'selection.json').exists():sealed(OUT/'selection.json',selected)
 else:
  if not (OUT/'final-comparison-complete.json').exists():sealed(OUT/'final-comparison-complete.json',dict(models=names,images=36,selection_sha256=sha(OUT/'selection.json'),retuning_prohibited=True,only_one_locked_comparison=True))
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['exposed','final'],default='exposed');a=p.parse_args();run(a.part)
