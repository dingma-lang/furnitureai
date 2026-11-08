"""Source-group bootstrap from saved development predictions; no inference."""
import copy
import json
import numpy as np
from common import *
from metrics import evaluate

def run():
 rows=load_rows('development');pred={r['image_id']:read(OUT/'original-development-v1'/(r['image_id']+'.json'))['predictions'] for r in rows};rng=np.random.default_rng(20261105)
 groups={r['identity_group'] for r in rows}
 if len(groups)!=len(rows):raise ValueError('Bootstrap expects current one-photo-per-group development lock')
 stats={k:[] for k in ('precision','recall','macro_ap50','macro_ap75')};per={c:{k:[] for k in ('ap50','ap75','precision','recall')} for c in CLASSES};valid_macro=0
 for b in range(500):
  sample=[];ps={}
  for j,i in enumerate(rng.integers(0,len(rows),len(rows))):
   row=copy.deepcopy(rows[int(i)]);old=row['image_id'];row['image_id']=old+'-bootstrap-'+str(j);sample.append(row);ps[row['image_id']]=pred[old]
  m=evaluate(sample,ps)
  for k in ('precision','recall'):
   if m['micro_operating'][k] is not None:stats[k].append(m['micro_operating'][k])
  if all(m['categories'][c]['gt_boxes']>0 for c in CLASSES):
   valid_macro+=1
   for suffix in ('50','75'):stats['macro_ap'+suffix].append(m['macro_ap'+suffix+'_supported'])
  for c in CLASSES:
   for k in per[c]:
    if m['categories'][c][k] is not None:per[c][k].append(m['categories'][c][k])
  if b%100==0:print('BOOTSTRAP',b,'/500',flush=True)
 interval=lambda vals:dict(lower=float(np.quantile(vals,.025)),upper=float(np.quantile(vals,.975)),valid_replicates=len(vals)) if vals else None
 save(OUT/'original-development-v1/bootstrap-uncertainty.json',dict(seed=20261105,resamples=500,unit='known source group; 18 unique development groups',level=.95,
   overall={k:interval(v) for k,v in stats.items()},categories={c:{k:interval(v) for k,v in ks.items()} for c,ks in per.items()},
   caveats='Conditional macro intervals retain only resamples with all seven GT classes; per-class undefined values omitted. Tiny-support, bootstrap conditional intervals do not cover annotation error, scope bias or pretraining contamination. Analytic resampling only; no new data or partitions.'))
 print('COMPLETE',valid_macro,'valid all-class macro resamples',flush=True)

if __name__=='__main__':run()
