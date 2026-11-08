"""Conservative product-wide eligibility guard; preserves the original image audit unchanged."""
import collections as C
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase3-v1'))
from common import *


def guard(rows,pairs):
 result=copy.deepcopy(rows);ids={r['product_id'] for r in rows};risks=C.defaultdict(list)
 for pair in pairs:
  if pair['left_product_id']==pair['right_product_id']:continue
  for pid in (pair['left_product_id'],pair['right_product_id']):
   if pid in ids:risks[pid].append(pair)
 for row in result:
  row['image_status_before_product_guard']=row['status'];row['product_identity_risk']=row['product_id'] in risks
  if row['product_identity_risk'] and row['status']=='provisional_accepted':
   row['status']='quarantined_duplicate_or_identity_risk';row['reasons']=['product_has_unresolved_cross_product_image_similarity']
 return result,dict(risks)


def main():
 verify_plan();dest=IMAGE_DEST;source=dest/'audit-v1';version=2
 while (dest/f'audit-v{version}').exists():version+=1
 target=dest/f'audit-v{version}'
 first=read(source/'summary.json');rows=read(source/'screening.json')['images'];pairs=read(source/'duplicate-and-overlap-pairs.json');plan=read(OUT/'acquisition-plan.json')
 filtered,risks=guard(rows,pairs);accepted=[r for r in filtered if r['status']=='provisional_accepted'];before=[r for r in rows if r['status']=='provisional_accepted']
 target.mkdir();summary=copy.deepcopy(first)
 summary.update({'parent_image_audit_sha256':sha(source/'screening.json'),'product_guard_code_sha256':sha(__file__),
  'product_guard_policy':'Any cross-product exact/perceptual similarity suspicion quarantines all otherwise accepted views for each new affected product; no identity merging, suspects can be false positives',
  'image_level_provisional_candidates_before_guard':len(before),'provisional_accepted_images':len(accepted),'products_with_provisional_accepted_images':len({r['product_id'] for r in accepted}),
  'products_quarantined_for_possible_identity_overlap':len(risks),'previously_accepted_images_quarantined_by_product_guard':len(before)-len(accepted),
  'status_counts':dict(C.Counter(r['status'] for r in filtered)),
  'quarantined_images':sum(r['status']=='quarantined_duplicate_or_identity_risk' for r in filtered),
  'image_integrity_errors':0,'no_new_downloads_or_inference_for_product_guard':True})
 for cls,data in summary['by_class'].items():
  r=[x for x in filtered if x['category']==cls];a=[x for x in r if x['status']=='provisional_accepted'];counts=C.Counter(x['product_id'] for x in a)
  data.update({'provisional_accepted_images':len(a),'products_with_accepted_images':len(counts),'products_with_two_or_more_nonflagged_images':sum(n>=2 for n in counts.values()),
    'statuses':dict(C.Counter(x['status'] for x in r)),'products_with_identity_suspicions':len({x['product_id'] for x in r if x['product_identity_risk']})})
 snapshot=read(OUT/'preservation-before.json');bad=[rel for rel,h in snapshot.items() if sha(ROOT/rel)!=h]
 if bad:raise ValueError('Historical files changed')
 summary['integrity']['historical_mismatches']=bad;summary['integrity']['original_files_hashes_checked']=len(snapshot)
 save(target/'summary.json',summary);save(target/'screening.json',{'images':filtered,'no_original_annotations_changed':True})
 save(target/'provisional-candidates.json',{'images':accepted,'human_verified':False,'splits_changed':False,'must_not_admit_product_risk_rows':True})
 save(target/'product-identity-quarantine.json',{'products':risks,'relationships':'possible similarity only; no products merged'})
 save(target/'products.json',{'products':[{'product_id':p['product_id'],'category':p['category'],'family_group':p['family_group'],'accepted_image_ids':[r['image_id'] for r in accepted if r['product_id']==p['product_id']],
  'identity_risk':p['product_id'] in risks,'physical_identity':'unverified','capture_independence':'unverified','human_verified':False} for p in plan['products']]})
 lines=['# Phase 3 final acquisition and product-aware screening report','',summary['scope'],'',
 f"All **1,484 approved images / 500 listing IDs** downloaded, passed size/SHA-256/full decode and available ETag/MD5 checks, with **zero retries or integrity failures**. Official CC BY 4.0 documents match Phase 1A. Image transfer **9,012,843 bytes**, documentation **18,659 bytes**, total **9,031,502 bytes**. Conservative charge **759,939,072 bytes**, below the 2,000,000,000-byte cap. Largest image **22,689 bytes**, below 512,000 bytes.",'',
 f"Final conservative subset: **{len(accepted)} images / {summary['products_with_provisional_accepted_images']} listing IDs**. Rejected **{summary['rejected_images']}** (568 model disagreements, 37 quality flags); quarantined **{summary['quarantined_images']}**; unresolved low-margin **{summary['unresolved_low_margin_images']}**. No classifier/encoder was trained, no original model was changed and no official holdout scoring ran.",'',
 '| Category | Products downloaded | Images downloaded | Provisional images | Products ≥1 candidate | Products ≥2 nonflagged |','|---|---:|---:|---:|---:|---:|']
 for cls in CLASSES:
  d=summary['by_class'][cls];lines.append(f"| {cls} | {d['products_downloaded']} | {d['images_downloaded']} | {d['provisional_accepted_images']} | {d['products_with_accepted_images']} | {d['products_with_two_or_more_nonflagged_images']} |")
 lines += ['',
 f"Image-only audit v1 yielded 434 candidates. Product guard {target.name} quarantined another **{len(before)-len(accepted)}** otherwise accepted images because their products had unresolved cross-product similarity. **{len(risks)} new products** have such suspicions. Audit v1 remains unchanged. Use **{target.name}/provisional-candidates.json**, not the image-only v1 list, for subsequent dataset preparation.",'',
 '**Duplicates and identity:** zero exact file/pixel duplicate pairs; 475 perceptual near-pairs, including 449 cross-product suspected pairs. Reference pair counts: 76 official holdout, 31 frozen internal development, 86 other existing local images, 282 between new images. A new product with any such cross-product suspicion is excluded as a whole from the provisional subset. These signatures can give false positives; they do not establish identity, and no products were merged. Known metadata family/reserved ID overlap is zero. Unknown aliases and independent capture/physical identity remain unresolved.','',
 '**Licence/provenance:** originals are unchanged under `../images/`; official documents and source/hash records are in `../documentation/`, `../ATTRIBUTION.md`, `../remote-preflight-v1.json` and `../transfer-ledger.json`. The immutable approved plan remains `data/abo/experiments/phase3-preparation-v1/acquisition-plan.json`. Individual image SHA-256 sums are in `../audit-v1/SHA256SUMS`.','',
 f"**Preservation:** all **{len(snapshot)} historical files** still match their snapshot, including original models, checkpoints and holdouts. Frozen development labels/vectors and official validation/test partitions remain unchanged. No manual evidence or human approvals were invented. Read-only frozen SigLIP inference was limited to new photographs.",'',
 '**Unresolved issues:** model screening scores are uncalibrated and can reject correctly labelled images; low-margin rows remain excluded. Perceptual quarantine needs later disambiguation rather than automatic merging. Nonflagged images are not proven independent captures. Whole-object suitability is not human-verified. Some categories, especially table/chair, have low conservative yield; rules were not weakened to reach a target. This is an acquisition audit, not a trained dataset or accuracy result.','',
 '**Files:** `screening.json` retains original image status and new product-risk status; `product-identity-quarantine.json` records pair evidence; `products.json` records category/family/provisional views; `provisional-candidates.json` is the conservative candidate subset. Actual final allocated disk use is in `../final-storage-v2.json`. No training or permanent split changes were performed.']
 atomic(target/'report.md',('\n'.join(lines)+'\n').replace('final-storage-v2.json',f'final-storage-v{version}.json').encode())
 sums=''.join(f'{sha(p)}  {p.relative_to(dest)}\n' for p in sorted(target.glob('*')) if p.is_file())
 atomic(target/'artifact-SHA256SUMS',sums.encode())
 # Retain the old storage report and add a final, fully accounted snapshot.
 used=sum(p.stat().st_blocks*512 for p in dest.rglob('*') if p.is_file())
 final={'allocated_bytes_before_this_new_record':used,'final_report':str(target.relative_to(dest)/'report.md'),'cap_bytes':plan['limits']['new_allocated_storage_bytes']}
 if used+8192>=final['cap_bytes']:raise RuntimeError('Storage cap')
 save(dest/f'final-storage-v{version}.json',final)
 print(json.dumps({'accepted_images':len(accepted),'accepted_products':summary['products_with_provisional_accepted_images'],'rejected_images':summary['rejected_images'],'quarantined_images':summary['quarantined_images'],'unresolved_images':summary['unresolved_low_margin_images'],'identity_risk_products':len(risks)},indent=2))

if __name__=='__main__':main()
