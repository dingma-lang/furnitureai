"""Keep related furniture/part vocabularies separate rather than forcing seven classes."""
import collections as C
from core import *
from reader import stream

ANCILLARY={90:'bench',128:'bookcase',233:'chaise_longue',390:'drawer',395:'dresser',468:'footstool',480:'futon',710:'music_stool',738:'ottoman',795:'pew_(church_bench)',878:'recliner',1013:'step_stool',1018:'stool',1158:'wardrobe'}

def run():
 result={}
 for split in ('train','val'):
  cats={};images={c:set() for c in ANCILLARY};instances=C.Counter();masks=C.Counter();neg=C.Counter();nex=C.Counter()
  for kind,r in stream(OUT/f'lvis_v1_{split}.json.zip'):
   if kind=='categories' and r['id'] in ANCILLARY:cats[r['id']]=r
   elif kind=='images':neg.update(set(r['neg_category_ids'])&set(ANCILLARY));nex.update(set(r['not_exhaustive_category_ids'])&set(ANCILLARY))
   elif kind=='annotations' and r['category_id'] in ANCILLARY:
    cid=r['category_id'];images[cid].add(r['image_id']);instances[cid]+=1;masks[cid]+=bool(r.get('segmentation'))
  rows=[]
  for cid,name in ANCILLARY.items():
   if cats[cid]['name']!=name or cats[cid]['image_count']!=len(images[cid]) or cats[cid]['instance_count']!=instances[cid]:raise ValueError('Ancillary source count inconsistency')
   rows.append(dict(native_id=cid,name=name,definition=cats[cid]['def'],images=len(images[cid]),instances=instances[cid],boxes=instances[cid],masks=masks[cid],
    explicit_negative_images=neg[cid],not_exhaustive_flag_images=nex[cid],seven_class_mapping=None,
    policy='Part, unit, subtype or seating semantics need an explicit separate mapping; never silently merge into Cabinet/Chair/Sofa. Drawer is a part, not a whole cabinet.'))
  result[split]=rows
 commit(OUT/'additional-unmapped-furniture-categories.json',dict(splits=result,counts_verified_by_annotation_stream=True,no_effect_on_existing_core_or_proposed_selection=True))
 print('Ancillary native categories independently counted',len(ANCILLARY),flush=True)

if __name__=='__main__':run()
