"""Build source-native candidates from existing official metadata only."""
import base64
import collections as C
import csv
import sys
from core import *
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
import prepare_pilot as prior
from metadata_dictionary import read_dictionary

def build():
 if OUT.exists():raise FileExistsError('Fresh Phase 6A preparation directory required')
 names=read_dictionary(META/'oidv7-class-descriptions-boxable.csv')
 for c,mid in MIDS.items():
  expected={'sofa':'Couch','cabinet':'Cabinetry'}.get(c,c.capitalize())
  if names[mid]!=expected:raise ValueError('Native class dictionary changed')
 # Full source tables, including excluded photos, prevent bridge leakage.
 full=[];infos={}
 for part in ('validation','test'):
  with (META/(part+'-images-with-rotation.csv')).open(newline='') as f:
   for r in csv.DictReader(f):
    try:
     identity=prior.photo_identity(r);author=prior.normalized_url(r['AuthorProfileURL'])
     if len(base64.b64decode(r['OriginalMD5'],validate=True))!=16:raise ValueError()
    except (ValueError,TypeError):continue
    full.append(dict(image_id=r['ImageID'],source_identity=identity,photographer_identity=author,metadata_original_md5=r['OriginalMD5']))
    if part=='validation':infos[r['ImageID']]=r
 grouped=prior.grouped(full);lookup={r['image_id']:r for r in grouped}
 old_manifest=read(META/'pilot-proposal-v4/proposed-image-acquisition.json')
 old_ids={r['image_id'] for r in old_manifest['images']}
 forbidden_groups={lookup[i]['identity_group'] for i in old_ids if i in lookup}
 protected=prior.d.reserved_identities()|old_ids|forbidden_groups
 audited=read(META/'audit-v1/furniture-candidates.json')['validation']
 anns=C.defaultdict(list);labels=C.defaultdict(dict);contexts=C.defaultdict(set)
 for r in audited['annotations']:box(r['box']);anns[r['image_id']].append(r)
 for r in audited['image_label_coverage']:labels[r['image_id']][r['label']]=r['value']
 with (META/'validation-annotations-bbox.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   if r['ImageID'] in anns and r['IsDepiction']=='0':contexts[r['ImageID']].add(names[r['LabelName']])
 exclusions=[];candidates=[]
 for ident,gs in sorted(anns.items()):
  reasons=[];info=infos.get(ident);identity=lookup.get(ident)
  if not info or not identity:exclusions.append(dict(image_id=ident,reasons=['missing_source_identity']));continue
  if ident in old_ids or identity['identity_group'] in forbidden_groups:reasons.append('historical_image_or_known_source_group')
  try:lic=license_url(info['License'])
  except ValueError:lic=None;reasons.append('restricted_or_missing_metadata_licence')
  try:
   if float(info['Rotation'])!=0:reasons.append('nonzero_rotation')
   if int(info['OriginalSize'])<=0:raise ValueError()
  except (ValueError,TypeError):reasons.append('invalid_size_or_rotation')
  if prior.OUTDOOR.search(info['Title']):reasons.append('outdoor_title')
  indoor=sorted(contexts[ident]&prior.INDOOR_OBJECTS);title=prior.INDOOR.findall(info['Title'])
  if not indoor and not title:reasons.append('no_strong_indoor_metadata_evidence')
  eligible=[g for g in gs if prior.pilot_box(g) and labels[ident].get(g['label'])==1]
  if not eligible:reasons.append('no_usable_native_box')
  if any({g['label'],h['label']} in ({'sofa','bed'},{'table','desk'}) and prior.d.iou(g['box'],h['box'])>=.8 for j,g in enumerate(gs) for h in gs[j+1:]):reasons.append('conflicting_native_boxes')
  if reasons:exclusions.append(dict(image_id=ident,reasons=reasons));continue
  coverage={c:labels[ident].get(c)==0 or (labels[ident].get(c)==1 and any(g['label']==c for g in gs)) for c in CLASSES}
  candidates.append(dict(**identity,source_partition='validation',image_url=image_url(dict(image_id=ident,source_partition='validation')),
   source_landing_url=info['OriginalLandingURL'],source_original_url=info['OriginalURL'],author=info['Author'],author_profile_url=info['AuthorProfileURL'],title=info['Title'],license_url=lic,metadata_original_size=int(info['OriginalSize']),
   annotations=gs,native_box_counts=dict(C.Counter(g['label'] for g in gs)),eligible_box_counts=dict(C.Counter(g['label'] for g in eligible)),coverage=coverage,coverage_provenance={c:'published_negative' if labels[ident].get(c)==0 else 'native_positive_with_boxes' if coverage[c] else 'unknown' for c in CLASSES},
   annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_creator='Google LLC',annotation_source='https://storage.googleapis.com/openimages/web/download_v7.html',
   indoor_evidence_rank=3 if title else 2,indoor_evidence=dict(title_matches=title,context_objects=indoor),scene_verified=False,human_verified=False,license_verified=False,image_sha256=None,
   taxonomy_risks=[c+': native source scope requires whole-object/unit check' for c in ('desk','cabinet') if c in {g['label'] for g in gs}],
   optimization_status='pending image screening and licence verification',physical_identity_status='unknown beyond full source-photo/MD5/photographer grouping'))
 # One proposal per entire known family. Rarity balanced, deterministic.
 pool=list(candidates);ranked=[];used=set();counts=C.Counter()
 support={c:sum(bool(r['eligible_box_counts'].get(c)) for r in pool) for c in CLASSES}
 while pool:
  available=[r for r in pool if r['identity_group'] not in used]
  if not available:break
  viable=[c for c in CLASSES if any(r['eligible_box_counts'].get(c) for r in available)]
  anchor=min(viable,key=lambda c:(counts[c],support[c],c))
  r=min((r for r in available if r['eligible_box_counts'].get(anchor)),key=lambda r:(-r['indoor_evidence_rank'],-len(r['eligible_box_counts']),hashlib.sha256((SEED+r['image_id']).encode()).hexdigest()))
  ranked.append(dict(r,selection_rank=len(ranked)+1,selection_anchor=anchor));used.add(r['identity_group'])
  counts.update(r['eligible_box_counts'].keys());pool.remove(r)
 # Native dictionary/hierarchy preserved. COCO has no local official instances file.
 audit=dict(total_validation_native_furniture_images=len(anns),eligible_metadata_candidates=len(candidates),independent_known_groups=len(ranked),historical_source_groups_excluded=len(forbidden_groups),full_source_identity_records=len(full),
  exclusions_by_reason=dict(C.Counter(reason for r in exclusions for reason in r['reasons'])),category_support={c:dict(images=sum(bool(r['eligible_box_counts'].get(c)) for r in candidates),boxes=sum(r['eligible_box_counts'].get(c,0) for r in candidates)) for c in CLASSES},
  coco_local_annotation_files=[str(p.relative_to(ROOT)) for p in (ROOT/'data').rglob('instances*.json')],coco_counts='unavailable without approved official annotation acquisition',local_images_reused=0,
  local_reuse_reason='All prior 150 Open Images memberships and source groups excluded; ABO catalogue images lack room detection boxes; existing authorised rooms have no ground-truth boxes. Frozen photographs not opened.',source_test_split_excluded_entirely=True)
 OUT.mkdir(parents=True)
 write_new(OUT/'candidate-manifest.json',dict(seed=SEED,images=ranked,selection_limit=400,license_verified=False,photographs_downloaded=0,metadata_sha256={p.name:sha(p) for p in META.iterdir() if p.suffix=='.csv' or p.name=='bbox_labels_600_hierarchy.json'}))
 write_new(OUT/'metadata-audit.json',audit);write_new(OUT/'exclusions.json',exclusions)
 write_new(OUT/'protected-identities.json',sorted(protected))
 write_new(OUT/'taxonomy.json',dict(open_images={c:dict(mid=mid,source_name=names[mid],policy='exact native only; no descendant substitution') for c,mid in MIDS.items()},
  coco={str(i):dict(source_name=a,target=b,policy='dining-table subtype only' if i==67 else 'separate dataset-native scoring stratum') for i,(a,b) in COCO.items()},coco_unavailable=['desk','cabinet','lamp'],
  ambiguous_native_classes=['desk','cabinet'],unknown_classes_are_background=False,source_gt_boxes_preserved=True))
 print(json.dumps(audit,indent=2));print('RANKED',len(ranked),'known-source groups')

if __name__=='__main__':build()
