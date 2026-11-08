"""Freeze a metadata-only stratified proposal before image-acquisition approval."""
import collections as C
from core import *

def split(rows,dev_count,seed=SEED):
 if len({r['identity_group'] for r in rows})!=len(rows):raise ValueError('Expected one photo per known source group')
 dev=[];remaining=list(rows)
 counts={c:sum(bool(r['eligible_box_counts'].get(c)) for r in rows) for c in CLASSES}
 # Ensure every available class has development and training support where
 # >=2 independent groups exist. No image content or model scores consulted.
 for cat in sorted(CLASSES,key=lambda c:(counts[c],c)):
  if counts[cat]<2 or any(r['eligible_box_counts'].get(cat) for r in dev):continue
  options=[r for r in remaining if r['eligible_box_counts'].get(cat) and all(not r['eligible_box_counts'].get(c) or sum(bool(x['eligible_box_counts'].get(c)) for x in remaining)>1 for c in CLASSES if counts[c]>=2)]
  if options and len(dev)<dev_count:
   selected=min(options,key=lambda r:hashlib.sha256((seed+'|dev|'+r['image_id']).encode()).hexdigest());dev.append(selected);remaining.remove(selected)
 for r in sorted(list(remaining),key=lambda r:hashlib.sha256((seed+'|dev|'+r['image_id']).encode()).hexdigest()):
  if len(dev)>=dev_count:break
  if all(not r['eligible_box_counts'].get(c) or sum(bool(x['eligible_box_counts'].get(c)) for x in remaining)>1 for c in CLASSES if counts[c]>=2):dev.append(r);remaining.remove(r)
 ids={r['image_id'] for r in dev}
 return [dict(r,proposed_split='development' if r['image_id'] in ids else 'train') for r in rows]

def run():
 original=OUT/'acquisition-manifest.json';m=read(original);rows=split(m['images'],math.ceil(len(m['images'])*.2))
 isolation(rows,set(read(OUT/'protected-identities.json')))
 m.update(images=rows,preceding_proposal_sha256=sha(original),partition_policy='Deterministic 80/20 source-group stratification on native metadata only; seed recorded; every class with >=2 groups retains train/dev support if feasible. No pixel/model/human decisions used.')
 m['categories']={p:{c:dict(images=sum(bool(r['eligible_box_counts'].get(c)) for r in rows if r['proposed_split']==p),ordinary_eligible_boxes=sum(r['eligible_box_counts'].get(c,0) for r in rows if r['proposed_split']==p),native_boxes=sum(r['native_box_counts'].get(c,0) for r in rows if r['proposed_split']==p)) for c in CLASSES} for p in ('train','development')}
 write_new(MANIFEST,m);print('FROZEN PROPOSAL V2',len(rows),'images',dict(C.Counter(r['proposed_split'] for r in rows)),sha(MANIFEST))

if __name__=='__main__':run()
