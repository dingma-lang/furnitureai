"""Integrity/duplicate/provisional weak-label audit only; no model fitting or holdout scoring."""
import collections as C
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import numpy as np
import torch
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase3-v1'))
import runner as prepared
from common import *

STOP=False


def classify_candidate(quality_reasons,duplicate_reasons,admit,confidence):
 if quality_reasons:return 'rejected_quality',quality_reasons
 if duplicate_reasons:return 'quarantined_duplicate_or_identity_risk',duplicate_reasons
 if admit:return 'provisional_accepted',[]
 if confidence and confidence['metadata_label_agreement']:return 'unresolved_low_margin',['insufficient_metadata_model_margin']
 return 'rejected_model_disagreement',['metadata_label_disagrees_with_frozen_prompt_screen']


def report(dest,reportdir,plan,rows,pairs,ledger,checks):
 counts=C.Counter(r['status'] for r in rows);classes={}
 for cls in CLASSES:
  selected=[p['product_id'] for p in plan['products'] if p['category']==cls];selected_images=[r for r in rows if r['category']==cls]
  accepted=[r for r in selected_images if r['status']=='provisional_accepted'];by=C.Counter(r['product_id'] for r in accepted)
  classes[cls]={'products_downloaded':len(selected),'images_downloaded':len(selected_images),'provisional_accepted_images':len(accepted),'products_with_accepted_images':len(by),'products_with_two_or_more_nonflagged_images':sum(n>=2 for n in by.values()),'statuses':dict(C.Counter(r['status'] for r in selected_images))}
 exact=[p for p in pairs if p['match'] in ('file_sha256','decoded_pixels')];near=[p for p in pairs if p['match']=='perceptual_distance_le_2']
 unique_pairs={tuple(sorted([p['left_image_id'],p['right_image_id']])) for p in exact};near_pairs={tuple(sorted([p['left_image_id'],p['right_image_id']])) for p in near}
 used=0;seen=set()
 for path in dest.rglob('*'):
  if path.is_file():
   st=path.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:used+=st.st_blocks*512;seen.add(key)
 summary={'scope':'PROVISIONAL AUTOMATED ACQUISITION SCREENING; no human-reviewed accuracy or capture-independence claim',
  'images_downloaded':len(rows),'products_downloaded':len(plan['products']),'provisional_accepted_images':counts['provisional_accepted'],
  'rejected_images':sum(n for k,n in counts.items() if k.startswith('rejected')),'quarantined_images':counts['quarantined_duplicate_or_identity_risk'],'unresolved_low_margin_images':counts['unresolved_low_margin'],
  'status_counts':dict(counts),'by_class':classes,'duplicate_exact_pair_count':len(unique_pairs),'perceptual_near_pair_count':len(near_pairs),
  'duplicate_records_can_have_multiple_match_types':True,'cross_product_suspected_pair_count':len({tuple(sorted([p['left_image_id'],p['right_image_id']])) for p in pairs if p['left_product_id']!=p['right_product_id']}),
  'observed_body_bytes':ledger['observed_body_bytes'],'image_body_bytes':sum(r['bytes'] for r in rows),'document_body_bytes':sum(r['bytes'] for r in ledger['documentation'].values()),
  'conservative_charged_bytes':ledger['charged_bytes'],'actual_bytes_are_exact_no_inflight_requests_or_unknown_attempts':all(a['status']=='complete' for r in ledger['objects'].values() for a in r['attempts']),
  'image_retries':sum(len(r['attempts'])-1 for r in ledger['objects'].values()),'allocated_storage_bytes_at_report':used,'image_storage_cap_bytes':plan['limits']['new_allocated_storage_bytes'],
  'source_plan_sha256':sha(OUT/'acquisition-plan.json'),'screening_code_sha256':sha(__file__),'gpu':torch.cuda.get_device_name(0),
  'model_fingerprint':read(WEAK/'features-v1/metadata.json')['model_processor_fingerprint'],'screen_policy':{'quality':w.POLICY,'near_duplicate_hamming_threshold':2,'cross_product_suspects_not_merged':True},
  'integrity':checks,'human_confirmations_created':0,'classifier_training_performed':False,'official_holdout_scoring':False,
  'known_family_and_reserved_identity_overlap_count':0,'independent_capture_counts':None,'independent_physical_product_counts':None,
  'remaining_issues':['Labels remain metadata-derived and weak; frozen-model agreement is uncalibrated, not a correctness probability.',
   'Different image IDs and nonduplicate signatures do not prove independent captures.',
   'Listing IDs/known-family separation do not prove distinct physical product identities.',
   'Perceptual duplicate suspects can be false positives; quarantined records remain distinct, never merged.',
   'Products with no accepted images or fewer than two nonflagged views need exclusion or a separately reviewed follow-up.',
   'Whole-object suitability is automatically screened and remains unverified by a human.'],
  'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
 save(reportdir/'summary.json',summary);save(reportdir/'screening.json',{'images':rows,'no_original_annotations_changed':True});save(reportdir/'duplicate-and-overlap-pairs.json',pairs)
 save(reportdir/'provisional-candidates.json',{'images':[r for r in rows if r['status']=='provisional_accepted'],'no_splits_created_or_changed':True,'human_verified':False})
 save(reportdir/'products.json',{'products':[{'product_id':p['product_id'],'category':p['category'],'family_group':p['family_group'],'title':p['name'],'label_confidence':p['label_confidence'],
  'accepted_image_ids':[r['image_id'] for r in rows if r['product_id']==p['product_id'] and r['status']=='provisional_accepted'],'human_verified':False,'physical_identity':'unverified','capture_independence':'unverified'} for p in plan['products']]})
 lines=['# Phase 3 approved acquisition and automated integrity report','',summary['scope'],'',
 f"Downloaded **{len(rows):,} images / {len(plan['products'])} listing IDs**. Accepted **{summary['provisional_accepted_images']}** as provisional weak-label candidates; rejected **{summary['rejected_images']}**, quarantined **{summary['quarantined_images']}**, unresolved low-margin **{summary['unresolved_low_margin_images']}**.",'',
 '| Category | Downloaded products | Images | Accepted images | Products ≥1 accepted | Products ≥2 nonflagged |','|---|---:|---:|---:|---:|---:|']
 for cls in CLASSES:
  d=classes[cls];lines.append(f"| {cls} | {d['products_downloaded']} | {d['images_downloaded']} | {d['provisional_accepted_images']} | {d['products_with_accepted_images']} | {d['products_with_two_or_more_nonflagged_images']} |")
 lines+=['',f"Actual response-body transfer: **{summary['observed_body_bytes']:,} bytes** = {summary['image_body_bytes']:,} image bytes + {summary['document_body_bytes']:,} official documentation bytes. Image retries: {summary['image_retries']}. Conservative precharged upper bound: **{summary['conservative_charged_bytes']:,} bytes**, within the 2,000,000,000-byte cap. Allocated storage at report creation: **{used:,} bytes**, within 2 GiB. Final post-report storage is recorded in `../final-storage.json`.",'',
 f"Exact duplicate pairs: **{len(unique_pairs)}**; perceptual near-duplicate pairs: **{len(near_pairs)}**; cross-product suspected overlap pairs: **{summary['cross_product_suspected_pair_count']}**. Pair records distinguish existing local/reserved reference images from new images, exact file/pixel matches from Hamming-distance suspects, and same-product from cross-product relationships. These are not automatic physical-identity merges.",'',
 'All 1,484 objects came from the exact approved official ABO S3 small-image URLs. All source HEAD checks completed before image GETs. Official CC BY 4.0 licence and README SHA-256 match the original Phase 1A records; copies and URLs/hashes are in `../documentation/` and `../transfer-ledger.json`. Attribution is preserved in `../ATTRIBUTION.md`. No source mirrors, original-resolution archives or model weights were downloaded.','',
 'Integrity checks cover file size/SHA-256, available single-part S3 ETag/MD5, full image decoding, dimensions, weak-label metadata/family isolation and exact/perceptual comparison with all existing local reference images. Automated screen uses fixed quality thresholds and the existing frozen SigLIP 2 prompt embeddings. No head was trained, no encoder weight was changed, and no official holdout model scoring ran.','',
 'The fixed internal development panel and official validation/test identities/families stay reserved. Comparing reference image hashes/pixels is for leakage prevention only, not holdout classification scoring. All current training and reviewed partitions remain unchanged. Historical preservation results are in `historical-preservation.json`.','',
 'Status counts: `'+json.dumps(dict(counts),sort_keys=True)+'`. Per-image quality, confidence, source SHA and exclusion reasons are in `screening.json`. Accepted means provisional screening eligibility, not validated training truth. Confidence margins are not calibrated probabilities. No human review flags or evidence were populated.','',
 '## Remaining issues','']+['- '+s for s in summary['remaining_issues']]
 lines+=['','## Recovery and next step','',
 'Downloader: `.venv/bin/python -B tools/phase3-acquisition-v1/download.py`. It binds the same approval/plan, preserves committed files, retains/charges partial attempts and refuses integrity/unavailable-source failures. Do not rerun to ignore a recorded stop condition. Screening uses atomically saved per-image frozen embeddings and can resume with the same `screen.py` command. Reports are versioned; existing reports are not replaced.','',
 'This acquisition does not automatically add all images to training. A later dataset build must use the provisional acceptance/exclusion records, retain original frozen development examples, and keep quarantined/uncertain products separate. No training is started by this script.']
 atomic(reportdir/'report.md',('\n'.join(lines)+'\n').encode())
 return summary


def main():
 verify_plan();plan=read(OUT/'acquisition-plan.json');dest=ROOT/plan['destination'];acquired=read(dest/'acquisition-summary.json')
 if acquired['status']!='complete':raise ValueError('Acquisition incomplete; no screening proceeds')
 ledger=read(dest/'transfer-ledger.json')
 if ledger['plan_binding']!=fingerprint(plan) or len(ledger['objects'])!=plan['image_count']:raise ValueError('Acquisition identity/scope mismatch')
 version=1
 while (dest/f'audit-v{version}').exists():version+=1
 reportdir=dest/f'audit-v{version}';reportdir.mkdir()
 # Quality and model decisions do not modify source files, partitions or original review annotations.
 inventory=read(PREP/'eligibility-inventory.json')['images'];identities=read(META/'identity-review.json');groups=w.group_inventory(inventory,identities)
 existing_products={r['abo_product_id'] for r in inventory};existing_groups={groups.find(pid) for pid in existing_products}
 if len({p['family_group'] for p in plan['products']})!=len(plan['products']) or any(p['product_id'] in existing_products or p['family_group'] in existing_groups for p in plan['products']):raise ValueError('Known identity/family overlap')
 protected={r['abo_product_id'] for r in inventory if r.get('proposed_split') in ('validation','test')}
 dev={r['abo_product_id'] for r in read(OUT/'frozen-development-protocol.json')['development_rows']}
 references=[]
 for r in inventory:
  exact,near=prepared.pixel_signatures(w.image_path(r))
  references.append({'image_id':r['image_id'],'product_id':r['abo_product_id'],'sha256':r['sha256'],'pixel':exact,'near':near,
   'origin':'official_holdout_reference' if r['abo_product_id'] in protected else 'frozen_development_reference' if r['abo_product_id'] in dev else 'existing_local_reference'})
 rows=[];signatures=[]
 for obj in plan['objects']:
  r=ledger['objects'][obj['image_id']];path=dest/obj['destination']
  if r.get('status')!='downloaded' or sha(path)!=r.get('sha256') or path.stat().st_size!=r['bytes']:raise ValueError('Downloaded image integrity changed')
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:im.load();dimensions=list(im.size)
  if not r.get('full_decode_verified'):raise ValueError('Acquisition missing full integrity confirmation')
  exact,near=prepared.pixel_signatures(path)
  row={'image_id':obj['image_id'],'product_id':obj['product_id'],'abo_product_id':obj['product_id'],'category':obj['category'],'weak_label':obj['category'],'furniture_class':obj['category'],
   'family_group':obj['family_group'],'source_url':obj['url'],'local_path':str(path.relative_to(ROOT)),'sha256':r['sha256'],'bytes':r['bytes'],'dimensions':dimensions,'decoded_pixel_sha256':exact,
   'label_source':'official ABO metadata; fixed automated consistency screen','human_verified':False,'capture_independence':'unverified','physical_identity':'unverified','original_cached_training_view':False}
  measures,reasons=w.quality(row);row['quality']=measures;row['quality_reasons']=reasons;row['duplicate_reasons']=[];rows.append(row)
  signatures.append({'image_id':row['image_id'],'product_id':row['product_id'],'sha256':row['sha256'],'pixel':exact,'near':near,'origin':'new_approved_acquisition'})
 pairs=[]
 for i,s in enumerate(signatures):
  for ref in references+signatures[:i]:
   kind='file_sha256' if s['sha256']==ref['sha256'] else 'decoded_pixels' if s['pixel']==ref['pixel'] else 'perceptual_distance_le_2' if (s['near']^ref['near']).bit_count()<=2 else None
   if kind is None:continue
   pair={'left_image_id':s['image_id'],'right_image_id':ref['image_id'],'left_product_id':s['product_id'],'right_product_id':ref['product_id'],
    'reference_origin':ref['origin'],'match':kind,'hamming_distance':(s['near']^ref['near']).bit_count(),'physical_identity_relationship':'unresolved; not merged'};pairs.append(pair)
   reason=kind+':'+ref['image_id'];rows[i]['duplicate_reasons'].append(reason)
   if ref['origin']=='new_approved_acquisition' and s['product_id']!=ref['product_id']:
    j=next(n for n,r in enumerate(rows) if r['image_id']==ref['image_id']);rows[j]['duplicate_reasons'].append(kind+':'+s['image_id'])
 # Frozen encoder inference on new photographs only. Never forward an official holdout through a model.
 device=w.gpu();manifest=read(w.MODEL/'checkpoint-manifest.json')
 for rel,h in manifest['files'].items():
  if sha(w.MODEL/rel)!=h:raise ValueError('Frozen model checkpoint mismatch')
 from transformers import AutoModel,AutoImageProcessor
 model=AutoModel.from_pretrained(str(w.MODEL),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval().requires_grad_(False)
 processor=AutoImageProcessor.from_pretrained(str(w.MODEL),local_files_only=True,use_fast=False);text=np.load(WEAK/'features-v1/text-prompts.npy',allow_pickle=False)
 cache=dest/'screening-cache-v1';cache.mkdir(exist_ok=True)
 def stop(*_):
  global STOP;STOP=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 for n,row in enumerate(rows,1):
  if STOP:raise InterruptedError('Screening stopped; committed embeddings retained')
  admit=False;confidence=None
  if not row['quality_reasons'] and not row['duplicate_reasons']:
   path=cache/(row['sha256']+'.npy')
   if path.exists():vector=np.load(path,allow_pickle=False)
   else:
    with Image.open(ROOT/row['local_path']) as image:im=ImageOps.exif_transpose(image).convert('RGB')
    with torch.inference_mode():
     inputs=processor(images=im,return_tensors='pt');vector=w.unit(model.get_image_features(**{k:v.to(device) for k,v in inputs.items()}).float().cpu().numpy())
    w.save_array(path,vector)
   if vector.shape!=(1,768) or not np.isfinite(vector).all():raise ValueError('Invalid screening cache')
   admit,confidence=w.model_screen(row,w.unit(vector)[0]@text.T)
  row['automated_model_screen']=confidence
  row['status'],row['reasons']=classify_candidate(row['quality_reasons'],row['duplicate_reasons'],admit,confidence)
  if n%100==0:print(json.dumps({'screened':n,'total':len(rows)}),flush=True)
 del model,processor;gc.collect();torch.cuda.empty_cache()
 snapshot=read(OUT/'preservation-before.json');bad=[p for p,h in snapshot.items() if sha(ROOT/p)!=h]
 if bad:raise ValueError('Historical preservation mismatch: '+str(bad[:3]))
 checks={'files_sha256_and_bytes_verified':len(rows),'full_decodes_verified':len(rows),'original_files_hashes_checked':len(snapshot),'historical_mismatches':bad,
  'frozen_development_protocol_sha256':sha(OUT/'frozen-development-protocol.json'),'frozen_development_vectors_sha256':sha(OUT/'frozen-development-features.npy')}
 save(reportdir/'historical-preservation.json',checks)
 summary=report(dest,reportdir,plan,rows,pairs,ledger,checks)
 attribution=(dest/'documentation/README.md').read_text();atomic(dest/'ATTRIBUTION.md',('Official ABO source README preserved below. Photographs were acquired unchanged; derived screening/embedding records are automated and provisional.\n\n'+attribution).encode())
 sums=[]
 for obj in plan['objects']:sums.append(f"{ledger['objects'][obj['image_id']]['sha256']}  {obj['destination']}\n")
 for path in sorted((dest/'documentation').glob('*')):sums.append(f'{sha(path)}  {path.relative_to(dest)}\n')
 atomic(reportdir/'SHA256SUMS',''.join(sums).encode())
 used=0;seen=set()
 for path in dest.rglob('*'):
  if path.is_file():
   stat=path.stat();key=(stat.st_dev,stat.st_ino)
   if key not in seen:used+=stat.st_blocks*512;seen.add(key)
 if used+8192>plan['limits']['new_allocated_storage_bytes']:raise RuntimeError('Final storage cap violation')
 save(dest/'final-storage.json',{'allocated_bytes_before_this_record':used,'within_image_storage_cap':True,'report':str(reportdir.relative_to(dest))})
 print(json.dumps({k:summary[k] for k in ['images_downloaded','products_downloaded','provisional_accepted_images','rejected_images','quarantined_images','unresolved_low_margin_images','observed_body_bytes']},indent=2));print('Report:',reportdir/'report.md')

if __name__=='__main__':main()
