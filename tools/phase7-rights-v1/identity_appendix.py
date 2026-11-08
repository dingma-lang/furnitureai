"""Explain protected-group exclusions by concrete historical IDs, without changing decisions."""
import collections as C
import importlib.util
import sys
from audit_core import *

def run():
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase7_rights_group_explanation',ROOT/'tools/phase7-acquisition-v1/cross_reference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 uf,protected,history,prior=mod.identity_graph();records=read(OUT/'rights-audit.json')['records']
 for r in records:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_photo_id'],alias)
 acquired={r['coco_image_id']:r for r in read(ROOT/'data/coco/image-pilots/phase6d-v1/screening-v2.json')['images']};members=C.defaultdict(list)
 for relative in ('data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json','data/open-images/preparation/phase6a-v1/acquisition-manifest-v2.json'):
  for r in read(ROOT/relative)['images']:
   members[uf.find(r['source_identity'])].append(dict(source='historical_Open_Images_slot',image_id=r['image_id'],partition=r['proposed_split'],source_manifest=relative))
 for r in prior:
  photo=acquired.get(r['coco_image_id'])
  members[uf.find(r['source_identity'])].append(dict(source='previous_COCO_rights_audit',coco_image_id=r['coco_image_id'],prior_provisional_partition=r['proposed_split'],
   acquired_in_phase6d=photo is not None,actual_photo_status=photo.get('status') if photo else None,actual_photo_partition=photo.get('proposed_split') if photo else None,
   note='A previous rights-audit record is not automatically a photograph used in training. The approved conservative exclusion also protects prior audit groups.'))
 details=[]
 for r in read(OUT/'identity-audit.json')['records']:
  if 'historical_source_or_creator_group' not in r['identity_flags']:continue
  matches=members[uf.find(r['source_photo_id'])]
  if not matches:raise ValueError('Historical exclusion has no resolvable protected anchor')
  details.append(dict(coco_image_id=r['coco_image_id'],source_photo_id=r['source_photo_id'],provisional_partition=r['provisional_partition'],
   creator_aliases=r['creator_aliases'],historical_matches=matches,includes_frozen_relationship=any(m.get('partition')=='frozen_evaluation' for m in matches),
   acquisition_excluded=True,physical_scene_identity_verified=False))
 if len(details)!=7 or sum(r['includes_frozen_relationship'] for r in details)!=1:raise ValueError('Final protected-group explanation does not reconcile')
 commit(OUT/'historical-group-match-details-v1.json',dict(records=details,historical_group_excluded_photos=7,newly_resolved_frozen_related_candidate_ids=[r['coco_image_id'] for r in details if r['includes_frozen_relationship']],
  original_five_frozen_related_exclusions_still_preserved=True,decisions_changed=False,proposed_eligible_photos_remain=66,new_source_requests=0))
 lines=[]
 for r in details:
  labels=[f"OpenImages {m['image_id']} ({m['partition']})" if 'image_id' in m else f"prior COCO rights audit {m['coco_image_id']} ({m['prior_provisional_partition']}; acquired={m['acquired_in_phase6d']})" for m in r['historical_matches']]
  lines.append(f"- New COCO {r['coco_image_id']:012d}: "+'; '.join(labels)+'.')
 text='''# Final coverage and historical identity appendix

This appendix supplements report.md without changing any sealed source decision, licence finding, image ID, partition, annotation or acquisition proposal.

The host429 stopped source requests after104 candidate attempts. The attempted selection anchors were100 Desk and4 Cabinet; the296 never-attempted candidates therefore include most Cabinet/Lamp and all later anchor blocks. Observed rights yield and category distribution are biased by that deterministic queue order and must not be extrapolated as a representative seven-class rights yield. The eligible66 photos still contain native co-occurring categories (108 core boxes/masks), but are primarily Desk-labelled. A separately approved future never-attempted queue should consider deterministic category interleaving while preserving every approved ID and original partition.

Seven rights-verified photographs were excluded for protected historical source/creator relationships:

'''+ '\n'.join(lines)+'''

In particular, newly resolved COCO000000154613 is linked by source-bound creator aliases to frozen OpenImages8b71f5a78d0f2fe7. It is quarantined; this is an additional creator relationship beyond the five frozen-related IDs already excluded during metadata preparation. A shared creator does not prove the same physical scene, but is conservatively isolated. No frozen photograph pixels were read or scored.

Two further rights-verified photographs, COCO000000526734 and000000506283, share a creator/source component across provisional development and training. Both remain quarantined; no membership was moved. Historical rights-only records in the list above are not falsely described as photographs used in training.

The report's remaining296 records were not individually requested or denied; only one candidate actually received429. The provider circuit remains suspended even after its fallback cooldown. The exact66-photo proposal remains55 training/11 development, expected object bytes8,368,471, with905 original core source annotations preserved across the full400-ID parent manifest and108 core annotations among the66 survivors. Additional subtype/part annotations among those survivors bring their native source total to122, not122 whole core furniture objects.
'''
 with (OUT/'review-addendum-v1.md').open('x') as f:f.write(text)
 implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()};commit(OUT/'implementation-sha256-final-v2.json',implementation)
 resource=OUT/'resource-usage-final-v2.json';commit(resource,dict(status='measuring'))
 seal=OUT/'FINAL-ARTIFACT-SHA256SUMS-v2'
 def seals():return ''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p!=seal)
 with seal.open('x') as f:f.write(seals())
 for _ in range(2):
  usage=dict(total_new_allocated_bytes=allocated()+OUT.parent.lstat().st_blocks*512,cap_bytes=DISK,self_and_both_seals_included=True,
   actual_downloaded_response_body_bytes=read(OUT/'summary.json')['actual_body_bytes'],new_source_requests_in_appendix=0,accounting='st_blocks*512 over all new rights/code files and new parent directory')
  resource.write_text(json.dumps(usage,indent=2)+'\n');seal.write_text(seals())
 if allocated()+OUT.parent.lstat().st_blocks*512!=usage['total_new_allocated_bytes'] or usage['total_new_allocated_bytes']>DISK:raise ValueError('Final storage accounting mismatch')
 for line in seal.read_text().splitlines():
  digest,name=line.split('  ',1)
  if sha(OUT/name)!=digest:raise ValueError('Final audit artifact seal mismatch')
 print('Concrete historical links verified; original decisions unchanged; final allocated bytes',usage['total_new_allocated_bytes'],flush=True)

if __name__=='__main__':run()
