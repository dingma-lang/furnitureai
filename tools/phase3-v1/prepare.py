"""No-network audit, ranked acquisition and frozen train-derived comparison protocol."""
import argparse
import collections as C
import csv
import gzip
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
import numpy as np
import torch
from common import *

QUOTAS={'bed':90,'desk':90,'cabinet':85,'lamp':70,'sofa':55,'chair':55,'table':55}
LIMITS={'products':500,'images':1500,'download_bytes':2_000_000_000,'object_bytes':512_000,
 'attempts_per_object':2,'new_allocated_storage_bytes':2*1024**3}
SOURCE='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/small/'
BAD_TITLE=re.compile(r'\b(bunk|loft|canopy|headboard|footboard|bed frame only|mattress|dresser|nightstand|ottoman|bench|vanity|set of|[2-9][ -]piece|dining set|modular|sectional piece|replacement|swatch|slipcover|wall mount|shade only|bulb only|base only|leg only)\b',re.I)


def reconcile(oldrows,newmeta,candidates):
 old={r['image_id']:r for r in oldrows};new={r['image_id']:r for r in newmeta['rows']}
 reasons={r['image_id']:r for r in newmeta['rejected']}
 removed=[{'image_id':i,'abo_product_id':old[i]['abo_product_id'],'class':old[i]['furniture_class'],**reasons.get(i,{'reason':'not_in_model_screen_candidates'})} for i in sorted(old.keys()-new.keys())]
 additions=[{'image_id':i,'abo_product_id':new[i]['abo_product_id'],'class':new[i]['weak_label']} for i in sorted(new.keys()-old.keys())]
 if len(new)!=len(old)-len(removed)+len(additions):raise ValueError('Reconciliation failed')
 # Labels/identities cannot silently change on retained rows.
 for i in old.keys()&new.keys():
  for field in ('abo_product_id','sha256','furniture_class'):
   if old[i][field]!=new[i][field]:raise ValueError('Retained row changed: '+i)
 return {'original_images':len(old),'original_retained':len(old.keys()&new.keys()),'removed_originals':removed,'added_views':additions,'final_images':len(new),
  'arithmetic':f'{len(old)} - {len(removed)} + {len(additions)} = {len(new)}','all_candidate_rejections':newmeta['rejected'],
  'candidate_screen_count':len(candidates),'replacements_are_capture_representatives_not_deleted_files':True}


def audit(out):
 original=read(w.CACHE/'train/metadata.json');summary=read(w.CACHE/'summary.json');current=read(WEAK/'features-v1/metadata.json');candidates=read(WEAK/'manifests-v1/candidates.json')['rows']
 if sha(w.CACHE/'train/metadata.json')!=summary['partitions']['train']['metadata_sha256'] or sha(w.CACHE/'train/features.npy')!=original['feature_sha256']:raise ValueError('Original feature cache mismatch')
 if sha(WEAK/'features-v1/features.npy')!=current['feature_sha256']:raise ValueError('Expanded cache mismatch')
 matrix=np.load(WEAK/'features-v1/features.npy',allow_pickle=False)
 if matrix.shape!=(len(current['rows']),768) or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4):raise ValueError('Feature shape/normalization failure')
 if w.make_folds(current['rows'])!=current['folds']:raise ValueError('Frozen grouping failure')
 checkpoints=read(WEAK/'checkpoint-registry.json')['current']
 for entry in checkpoints:
  if sha(WEAK/entry['path'])!=entry['sha256']:raise ValueError('Checkpoint changed')
 for directory in (w.MODEL,ROOT/'models/owlv2-original-baseline-20261103'):
  if not directory.is_dir():raise ValueError('Original model missing')
 preserved=read(WEAK/'preservation-before.json');bad=[r for r,h in preserved.items() if sha(ROOT/r)!=h]
 if bad:raise ValueError('Historical artifact mismatch: '+str(bad[:3]))
 inventory=read(PREP/'eligibility-inventory.json')['images'];identity=read(META/'identity-review.json');groups=w.group_inventory(inventory,identity)
 w.check_scope(current['rows'],[r for r in inventory if r.get('proposed_split') in ('validation','test')],groups)
 result=reconcile(original['rows'],current,candidates)
 result.update({'checkpoints_verified':len(checkpoints),'historical_hashes_verified':len(preserved),'feature_hashes_verified':True,'known_group_isolation_verified':True,'no_official_holdout_scoring':True})
 save(out/'expansion-audit.json',result)
 return current,matrix,inventory,identity,groups


def subtype(product):
 text=product['name'].lower(); cls=product['category']
 keywords={'bed':['platform','storage','upholstered','metal','wood','queen','twin','king'],
 'desk':['standing','writing','computer','corner','office','folding'],
 'cabinet':['storage','file','kitchen','bathroom','media','wood'],
 'lamp':['floor','table','desk','pendant','ceiling','wall'],
 'sofa':['sleeper','loveseat','reclining','chesterfield','sofa'],
 'chair':['office','dining','accent','recliner','rocking','outdoor','kids'],
 'table':['dining','coffee','side','console','outdoor','folding']}
 return next((k for k in keywords[cls] if k in text),'unspecified')


def choose_products(pool,target=500,quotas=QUOTAS):
 by=C.defaultdict(list)
 for product in pool:by[product['category']].append(product)
 selected=[];used_groups=set();used_images=set();counts=C.Counter();subtypes=C.Counter()
 # Greedy deterministic subtype diversity, then metadata/image quality evidence, then seeded ID.
 def pick(cls):
  options=[x for x in by[cls] if x['family_group'] not in used_groups and not used_images.intersection(x['image_ids'])]
  if not options:return False
  item=min(options,key=lambda x:(subtypes[(cls,x['subtype'])],-len(x['image_ids']),fingerprint([SEED,x['product_id']])))
  selected.append(item);used_groups.add(item['family_group']);used_images.update(item['image_ids']);counts[cls]+=1;subtypes[(cls,item['subtype'])]+=1
  return True
 for cls in ('bed','desk','cabinet','lamp','sofa','chair','table'):
  for _ in range(quotas[cls]):
   if len(selected)>=target or not pick(cls):break
 while len(selected)<target:
  order=sorted(CLASSES,key=lambda c:(counts[c]/quotas[c],('bed','desk','cabinet','lamp','sofa','chair','table').index(c)))
  if not any(pick(c) for c in order):break
 return selected


def raw_main_images(wanted):
 result=C.defaultdict(set)
 with tarfile.open(META/'archives/abo-listings.tar','r:') as archive:
  for member in sorted(archive.getmembers(),key=lambda m:m.name):
   if not re.fullmatch(r'listings/metadata/listings_[0-9a-f]\.json\.gz',member.name):continue
   with gzip.GzipFile(fileobj=archive.extractfile(member)) as stream:
    for line in stream:
     record=json.loads(line)
     if record['item_id'] in wanted and record.get('main_image_id'):result[record['item_id']].add(record['main_image_id'])
 return result


def acquisition(out,inventory,identity,groups):
 source=read(WEAK/'manifests-v1/metadata-only-expansion.json')['products']
 local_ids={r['abo_product_id'] for r in inventory};existing_groups={groups.find(i) for i in local_ids}
 reserved_ids={r['abo_product_id'] for r in inventory if r.get('proposed_split') in ('validation','test')}
 reserved_groups={groups.find(i) for i in reserved_ids}
 reserved_images={r['image_id'] for r in inventory};shared={r['image_id'] for r in identity['shared_image_ids']}
 known_paths={r['local_path'] for r in inventory}
 image_meta={}
 wanted={i for product in source for i in product['metadata_qualified_image_ids']}
 with gzip.open(META/'images/metadata/images.csv.gz','rt') as stream:
  for r in csv.DictReader(stream):
   if r['image_id'] in wanted:image_meta[r['image_id']]=r
 pool=[];excluded=[]
 for product in source:
  pid=product['product_id'];group=groups.find(pid);reasons=[]
  if pid in local_ids or group in existing_groups:reasons.append('existing_product_or_known_family')
  if group in reserved_groups:reasons.append('official_holdout_family')
  if not product['metadata_screen_pass'] or len(product['raw_types'])!=1 or product['raw_types'][0].lower()!=product['category']:reasons.append('category_ambiguous')
  if BAD_TITLE.search(product['name']):reasons.append('accessory_set_component_or_unclear_whole_object')
  candidates=[]
  for image_id in product['metadata_qualified_image_ids']:
   m=image_meta.get(image_id)
   if not m or image_id in reserved_images or image_id in shared:continue
   if not re.fullmatch(r'[0-9a-f]{2}/[0-9a-f]+\.(jpg|png)',m['path']):raise ValueError('Unsafe metadata image path')
   ratio=max(int(m['width']),int(m['height']))/min(int(m['width']),int(m['height']))
   if ratio>2.5:continue
   candidates.append(image_id)
  if len(candidates)<2:reasons.append('fewer_than_two_exclusive_suitable_links')
  if reasons:excluded.append({'product_id':pid,'reasons':reasons});continue
  pool.append({**product,'family_group':group,'subtype':subtype(product),'image_ids':sorted(candidates),
    'label_source':'official ABO product_type/category/title, metadata-only screen','label_confidence':'provisional_metadata_consistency_not_calibrated',
    'physical_identity_independence':'unverified; one listing per conservative known family',
    'capture_independence':'unverified; IDs and main/additional roles do not prove separate captures',
    'image_quality':'metadata-only aspect/dimensions; actual bytes/quality pending acquisition'})
 selected=choose_products(pool,LIMITS['products'])
 if len(selected)!=LIMITS['products']:raise ValueError(f'Only {len(selected)} eligible products; do not relax criteria')
 mains=raw_main_images({p['product_id'] for p in selected});objects=[]
 for rank,product in enumerate(selected,1):
  ids=sorted(product['image_ids'],key=lambda i:(i not in mains[product['product_id']],fingerprint([SEED,i])))[:3]
  product['rank']=rank;product['selected_image_ids']=ids
  for image_id in ids:
   m=image_meta[image_id]
   objects.append({'image_id':image_id,'product_id':product['product_id'],'category':product['category'],'family_group':product['family_group'],'metadata_path':m['path'],
    'url':SOURCE+m['path'],'destination':'images/'+m['path'],'original_metadata_dimensions':[int(m['width']),int(m['height'])],
    'role':'main' if image_id in mains[product['product_id']] else 'additional',
    'remote_size_bytes':None,'remote_exists':'not_checked_no_network','sha256':None,'capture_independence':'unverified'})
 if len(objects)>LIMITS['images'] or len({x['image_id'] for x in objects})!=len(objects) or len({x['destination'] for x in objects})!=len(objects):raise ValueError('Acquisition duplicate/image limit')
 actual_sizes=[r['actual_bytes'] for r in read(ROOT/'data/abo/image-pilots/phase1b-v1/acquisition.json')['objects'] if r.get('status')=='downloaded']
 mean=statistics.mean(actual_sizes);ordered=sorted(actual_sizes);p95=ordered[int(.95*(len(ordered)-1))]
 plan={'schema':1,'seed':SEED,'products':selected,'objects':objects,'limits':LIMITS,'destination':str(IMAGE_DEST.relative_to(ROOT)),
  'product_count':len(selected),'image_count':len(objects),'class_product_counts':dict(C.Counter(p['category'] for p in selected)),
  'family_group_count':len({p['family_group'] for p in selected}),'source_prefix':SOURCE,'documentation_sources':['https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/LICENSE-CC-BY-4.0.txt','https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/README.md'],'documentation_body_reservation_bytes':131072,'licence':'CC BY 4.0 per existing official ABO README/LICENSE; current remote availability/licensing must be confirmed before execution',
  'attribution_source':'data/abo/metadata/phase1-v1/README.md','licence_sha256':sha(META/'LICENSE-CC-BY-4.0.txt'),
  'estimated_transfer_bytes':round(mean*len(objects)),'estimate_method':f'Historical mean of {len(actual_sizes)} Phase 1B small-image objects, not remote HEAD sizes',
  'empirical_p95_per_object':p95,'p95_sum_scenario_bytes':p95*len(objects),'single_attempt_max_bytes':len(objects)*LIMITS['object_bytes'],
  'max_two_attempt_charged_bytes':len(objects)*LIMITS['object_bytes']*LIMITS['attempts_per_object'],
  'expected_allocated_image_bytes':sum(math.ceil(mean/4096)*4096 for _ in objects),
  'additional_feature_bytes_estimate':len(objects)*768*4,'request_count_without_retries':2*len(objects)+2,
  'price_note':'No paid API is configured. Provider/network charges are not quoted; exact remote sizes and egress cost are not known offline.',
  'no_images_downloaded':True,'no_source_probe_performed':True,'selection_is_not_guarantee_of_500_independent_physical_products':True}
 save(out/'acquisition-plan.json',plan)
 save(out/'identity-exclusions.json',{'source_candidates':len(source),'eligible_before_family_sampling':len(pool),'exclusion_counts':dict(C.Counter(r for x in excluded for r in x['reasons'])),'excluded':excluded,
  'remaining_eligible_not_selected':[{ 'product_id':p['product_id'],'family_group':p['family_group']} for p in pool if p['product_id'] not in {x['product_id'] for x in selected}],
  'generic_shared_assets_excluded_not_identity_merged':True,'known_family_one_listing_policy':True,'unknown_alias_risk_remaining':True})
 return plan


def frozen_protocol(out,current,matrix,inventory,identity,groups):
 rows=current['rows'];dev_indices=current['folds'][0];devset=set(dev_indices);fit=[i for i in range(len(rows)) if i not in devset]
 # This is a frozen INTERNAL development panel, already exposed in past experiments.
 # Historical trained heads cannot be used as comparable baselines because they saw it.
 dev=[rows[i] for i in dev_indices];training=[rows[i] for i in fit]
 if {groups.find(r['abo_product_id']) for r in dev}&{groups.find(r['abo_product_id']) for r in training}:raise ValueError('Frozen-panel family leakage')
 w.save_array(out/'frozen-development-features.npy',matrix[dev_indices])
 protocol={'schema':1,'scope':'FROZEN PROVISIONAL WEAK-LABEL INTERNAL DEVELOPMENT; previously exposed, never an independent test',
  'seed':SEED,'source_cache_sha256':current['feature_sha256'],'model_processor_fingerprint':current['model_processor_fingerprint'],
  'development_rows':dev,'baseline_training_rows':training,'development_feature_sha256':sha(out/'frozen-development-features.npy'),
  'development_product_count':len({r['abo_product_id'] for r in dev}),'development_image_count':len(dev),
  'baseline_training_product_count':len({r['abo_product_id'] for r in training}),'baseline_training_image_count':len(training),
  'class_development_products':dict(C.Counter({r['abo_product_id']:r['weak_label'] for r in dev}.values())),
  'label_policy':'frozen existing ABO weak labels; do not rescreen, relabel or replace development examples after acquisition',
  'fit_policy':'new products only add to FIT; official validation/test and all known related groups excluded',
  'fair_comparison':'refit baseline and expanded heads from scratch with identical fixed development panel, feature model, seeds and queue; never compare historical in-sample heads as leakage-free baselines',
  'selection':'product macro-F1 on this exact panel; epoch selection uses same internal panel and is optimistic',
  'uncertainty':'per-class product support and family-group bootstrap intervals, seed fixed, 1000 resamples; intervals conditional on this panel, not independent generalization guarantees',
  'configs':w.CONFIGS,'max_epochs':80,'patience':12,'cuda_allocator_fraction':.75,'head_batch_size':32,'queue_seconds_per_invocation':1800,
  'encoder_adaptation':False,'official_holdout_scoring':False}
 save(out/'frozen-development-protocol.json',protocol)
 return protocol


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();out=safe_output(args.output,ROOT/'data/abo/experiments')
 if any(out.iterdir()) if out.exists() else False:raise FileExistsError('Use a fresh preparation root')
 out.mkdir(exist_ok=True)
 save(out/'preservation-before.json',w.preservation(out))
 current,matrix,inventory,identity,groups=audit(out);plan=acquisition(out,inventory,identity,groups);protocol=frozen_protocol(out,current,matrix,inventory,identity,groups)
 sources=[w.CACHE/'train/metadata.json',w.CACHE/'train/features.npy',WEAK/'features-v1/metadata.json',WEAK/'features-v1/features.npy',PREP/'eligibility-inventory.json',META/'identity-review.json',META/'furniture-metadata-audit.json',META/'images/metadata/images.csv.gz',META/'archives/abo-listings.tar',META/'README.md',META/'LICENSE-CC-BY-4.0.txt',WEAK/'manifests-v1/metadata-only-expansion.json',ROOT/'tools/weak-training-v1/pipeline.py',ROOT/'tools/auto-training-v1/auto_train.py']
 sources += [WEAK/'features-v1/text-prompts.npy',w.MODEL/'checkpoint-manifest.json']
 for rel in read(w.MODEL/'checkpoint-manifest.json')['files']:sources.append(w.MODEL/rel)
 sources += list(HERE.glob('*.py'))
 artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='preservation-before.json'}
 save(out/'preparation-lock.json',{'source_files':{str(p.relative_to(ROOT)):sha(p) for p in sources},'artifacts':artifacts})
 save(out/'approval-template.json',{'approved':False,'approved_by':'','acquisition_plan_sha256':sha(out/'acquisition-plan.json'),'limits':LIMITS,'current_official_licence_rechecked':False})
 print(json.dumps({'products':plan['product_count'],'images':plan['image_count'],'classes':plan['class_product_counts'],'estimated_bytes':plan['estimated_transfer_bytes'],'development_images':protocol['development_image_count'],'development_products':protocol['development_product_count']},indent=2))

if __name__=='__main__':main()
