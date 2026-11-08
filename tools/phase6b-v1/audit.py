"""Source-native furniture/license audit. No photos, inferred negatives or models."""
import collections as C
import csv
import math
import re
from urllib.parse import urlparse
from common import *
from streaming import entries

def flickr_id(url):
 u=urlparse(url)
 if u.scheme not in ('http','https'):return None
 if u.hostname in ('www.flickr.com','flickr.com'):
  m=re.fullmatch(r'/photos/[^/]+/(\d+)/?',u.path)
  return m[1] if m else None
 if u.hostname and (u.hostname.endswith('.staticflickr.com') or u.hostname.endswith('.static.flickr.com')):
  m=re.fullmatch(r'/(?:[^/]+/)*(\d+)_[^/]+\.(?:jpg|jpeg|png)',u.path,re.I)
  return m[1] if m else None
 return None

class Union:
 def __init__(self,keys):self.parent={k:k for k in keys}
 def find(self,k):
  while self.parent[k]!=k:self.parent[k]=self.parent[self.parent[k]];k=self.parent[k]
  return k
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def historic_groups():
 base=ROOT/'data/open-images/metadata/phase5-v1';records=[]
 for part in ('validation','test'):
  with (base/(part+'-images-with-rotation.csv')).open(newline='') as f:
   for r in csv.DictReader(f):
    photo=flickr_id(r['OriginalLandingURL']);u=urlparse(r['AuthorProfileURL']);segments=u.path.strip('/').split('/')
    author=segments[1].casefold() if u.hostname in ('www.flickr.com','flickr.com') and len(segments)==2 and segments[0] in ('people','photos') else r['AuthorProfileURL']
    records.append(dict(image_id=r['ImageID'],photo=photo,author=author,md5=r['OriginalMD5']))
 uf=Union(r['image_id'] for r in records);owners={}
 for r in records:
  for key in ('photo','author','md5'):
   if not r[key]:continue
   tag=(key,r[key])
   if tag in owners:uf.join(r['image_id'],owners[tag])
   owners[tag]=r['image_id']
 old=read(base/'pilot-proposal-v4/proposed-image-acquisition.json')['images']+read(ROOT/'data/open-images/preparation/phase6a-v1/acquisition-manifest-v2.json')['images']
 protected_roots={uf.find(r['image_id']) for r in old}
 byphoto={r['photo']:'oi-source-group:'+uf.find(r['image_id']) for r in records if r['photo']}
 protected={'oi-source-group:'+r for r in protected_roots}
 return byphoto,protected,dict(source_metadata_records=len(records),historical_pilot_slots=len(old),protected_source_groups=len(protected),grouping='Full image metadata source-photo/normalized photographer/original MD5 union; includes excluded bridge records')

def allowed_license(url):
 u=urlparse(url);p=u.path.rstrip('/')
 return u.scheme in ('http','https') and u.hostname=='creativecommons.org' and not (u.port or u.username or u.query or u.fragment) and (p=='/publicdomain/zero/1.0' or re.fullmatch(r'/licenses/by/(1\.0|2\.0|2\.5|3\.0|4\.0)',p) is not None)

def box_status(a,im):
 b=a.get('bbox');good=isinstance(b,list) and len(b)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b)
 if not good:return 'invalid_structure'
 x,y,w,h=b
 if w<=0 or h<=0:return 'nonpositive_extent'
 if x<0 or y<0 or x+w>im['width']+1e-6 or y+h>im['height']+1e-6:return 'outside_image'
 return 'valid'

def segmentation_status(a,im):
 s=a.get('segmentation')
 if isinstance(s,list):
  if not s:return 'missing'
  return 'polygon' if all(isinstance(p,list) and len(p)>=6 and len(p)%2==0 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) for p in s) else 'invalid'
 if isinstance(s,dict):
  if s.get('size')!=[im['height'],im['width']]:return 'invalid'
  counts=s.get('counts')
  if isinstance(counts,list):return 'rle_uncompressed' if all(type(v)==int and v>=0 for v in counts) and sum(counts)==im['height']*im['width'] else 'invalid'
  if isinstance(counts,str) and counts:return 'rle_compressed_not_raster_decoded'
  return 'invalid'
 return 'missing'

def native_partition(path,part):
 images={};licenses={};cats={};stats={c:C.Counter() for c in MAP};ids={c:set() for c in MAP};byimage=C.defaultdict(list);contexts=C.defaultdict(set);seen=set();bad=[];total_annotations=0;info=None
 for key,r in entries(path):
  if key=='info':info=r
  elif key=='licenses':
   if r['id'] in licenses:raise ValueError('Duplicate licence ID')
   licenses[r['id']]=r
  elif key=='categories':
   if r['id'] in cats:raise ValueError('Duplicate category ID')
   cats[r['id']]=r
  elif key=='images':
   if r['id'] in images or type(r['id'])!=int or r['width']<=0 or r['height']<=0 or r['file_name']!=f"{r['id']:012d}.jpg":raise ValueError('Invalid/duplicate image schema')
   if r['license'] not in licenses:raise ValueError('Unknown image licence')
   expected=f"http://images.cocodataset.org/{part}/{r['file_name']}"
   if r['coco_url']!=expected:raise ValueError('Unexpected official image source metadata')
   images[r['id']]=r
  elif key=='annotations':
   if r['id'] in seen or r['image_id'] not in images:raise ValueError('Duplicate/broken annotation reference')
   seen.add(r['id']);total_annotations+=1;contexts[r['image_id']].add(r['category_id'])
   if r['category_id'] not in MAP:continue
   im=images[r['image_id']];c=r['category_id'];ids[c].add(r['image_id']);s=stats[c];s['boxes']+=1
   if r.get('iscrowd') not in (0,1):raise ValueError('Invalid source crowd flag')
   s['crowd_boxes' if r['iscrowd'] else 'noncrowd_boxes']+=1
   geometry=box_status(r,im);seg=segmentation_status(r,im);s['bbox_'+geometry]+=1;s['segmentation_'+seg]+=1
   if geometry!='valid' or seg in ('invalid','missing'):bad.append(dict(partition=part,annotation_id=r['id'],image_id=r['image_id'],bbox_status=geometry,segmentation_status=seg))
   usable=geometry=='valid' and seg=='polygon' and not r['iscrowd'] and r['bbox'][2]/im['width']>=.05 and r['bbox'][3]/im['height']>=.05 and r['bbox'][2]*r['bbox'][3]/im['width']/im['height']>=.005
   byimage[r['image_id']].append(dict(annotation_id=r['id'],category_id=c,bbox=r['bbox'],segmentation_kind=seg,iscrowd=r['iscrowd'],geometry=geometry,large_noncrowd_polygon=usable))
  else:raise ValueError('Unexpected top-level schema '+key)
 if len(cats)!=80 or any(cats[c]['name']!=source for c,(source,target) in MAP.items()):raise ValueError('COCO category definitions changed')
 for midset in contexts.values():
  if not midset<=cats.keys():raise ValueError('Unknown annotation category')
 category_names={r['name'] for r in cats.values()}
 if any(c in category_names for c in ('desk','cabinet','lamp')):raise ValueError('Unsupported category unexpectedly present')
 licence_counts=C.Counter(im['license'] for im in images.values());class_summary={}
 for c,(name,target) in MAP.items():
  class_summary[name]=dict(source_category_id=c,target_category=target,images=len(ids[c]),**dict(stats[c]),metadata_allowed_license_images=sum(allowed_license(licenses[images[i]['license']]['url']) for i in ids[c]),individually_rights_verified_images=0)
 return images,byimage,contexts,cats,licenses,dict(images=len(images),annotations=total_annotations,info=info,furniture_images=len(byimage),categories=class_summary,license_image_counts={str(k):v for k,v in sorted(licence_counts.items())}),bad

def rank_select(rows,count=400):
 remaining=list(rows);chosen=[];groups=set();counts=C.Counter()
 support={c:sum(c in r['usable_category_ids'] for r in rows) for c in MAP}
 while len(chosen)<count:
  pool=[r for r in remaining if r['identity_group'] not in groups]
  if not pool:break
  viable=[c for c in MAP if any(c in r['usable_category_ids'] for r in pool)]
  c=min(viable,key=lambda c:(counts[c],0 if c in (63,65) else 1,support[c],c))
  r=min((r for r in pool if c in r['usable_category_ids']),key=lambda r:(-r['indoor_evidence_rank'],-min(r['scene_context_category_count'],8),-len(r['usable_category_ids']),hashlib.sha256((SEED+'|rank|'+str(r['coco_image_id'])).encode()).hexdigest()))
  chosen.append(dict(r,selection_anchor=MAP[c][1],selection_rank=len(chosen)+1));groups.add(r['identity_group']);counts[c]+=1;remaining.remove(r)
 return chosen

def group_split(rows):
 # Twenty percent of each assigned selection-anchor stratum. One photo/group
 # is selected, and creator-group discovery may change this draft later.
 result=[]
 for cat in ('sofa','bed','chair','table'):
  group=sorted((r for r in rows if r['selection_anchor']==cat),key=lambda r:hashlib.sha256((SEED+'|split|'+r['identity_group']).encode()).hexdigest())
  dev=math.ceil(len(group)*.2)
  result.extend(dict(r,proposed_split='development' if n<dev else 'train') for n,r in enumerate(group))
 return sorted(result,key=lambda r:r['selection_rank'])

def run():
 space(DERIVED_MAX)
 if (OUT/'candidate-inventory.jsonl').exists():raise FileExistsError('Fresh audit outputs required')
 sources,protected,source_audit=historic_groups();summaries={};candidates=[];all_coco=set();all_photo=C.Counter();exclusion=C.Counter();bad_rows=[];definitions=None;license_definition=None;furniture_license_counts={};inventory_count=0;inventory_bytes=0
 inventory=OUT/'candidate-inventory.jsonl'
 with inventory.open('x') as output:
  for part in ('train2017','val2017'):
   images,byimage,context,cats,licenses,summary,bad=native_partition(OUT/'extracted/annotations'/('instances_'+part+'.json'),part)
   if all_coco&images.keys():raise ValueError('Official train/val image ID overlap')
   all_coco.update(images)
   if definitions is not None and (definitions!=cats or license_definition!=licenses):raise ValueError('Category/licence definitions differ across source partitions')
   definitions=cats;license_definition=licenses;summaries[part]=summary;bad_rows.extend(bad)
   names={i:r['name'] for i,r in cats.items()};furniture_license_counts[part]=dict(C.Counter(images[i]['license'] for i in byimage))
   indoor={'bed','couch','dining table','oven','refrigerator','microwave','sink','toilet','tv'};outdoor={'traffic light','stop sign','fire hydrant','surfboard','skis','snowboard','boat'}
   for ident,boxes in sorted(byimage.items()):
    im=images[ident];lic=licenses[im['license']];photo=flickr_id(im.get('flickr_url',''));source='flickr-photo:'+photo if photo else None
    if photo:all_photo[photo]+=1
    group=sources.get(photo,'coco-flickr:'+photo) if photo else None
    context_names={names[i] for i in context[ident]};indoor_context=sorted(context_names&indoor);outside_context=sorted(context_names&outdoor)
    usable=sorted({b['category_id'] for b in boxes if b['large_noncrowd_polygon']});reasons=[]
    if not allowed_license(lic['url']):reasons.append('restricted_or_unapproved_metadata_licence')
    if not photo:reasons.append('missing_supported_original_photo_identity')
    if group in protected:reasons.append('historical_protected_source_group_overlap')
    if min(im['width'],im['height'])<256:reasons.append('small_source_dimensions')
    if not usable:reasons.append('no_large_valid_noncrowd_polygon_box')
    if not indoor_context:reasons.append('no_indoor_metadata_context')
    if outside_context and not ({'bed','couch','oven','refrigerator','microwave','sink','toilet','tv'}&context_names):reasons.append('outdoor_context_without_strong_indoor_anchor')
    if part!='train2017':reasons.append('official_source_validation_reserved_not_training')
    for reason in reasons:exclusion[part+':'+reason]+=1
    row=dict(image_id='coco2017:'+f'{ident:012d}',coco_image_id=ident,source_partition=part,file_name=im['file_name'],width=im['width'],height=im['height'],source_identity=source,identity_group=group,
     official_metadata_photo_url=im['coco_url'],proposed_secure_image_url=f"https://s3.amazonaws.com/images.cocodataset.org/{part}/{im['file_name']}",original_flickr_asset_url=im.get('flickr_url'),
     proposed_source_page_resolver='https://www.flickr.com/photo.gne?id='+photo if photo else None,date_captured_metadata=im.get('date_captured'),
     metadata_license_id=im['license'],metadata_license_name=lic['name'],metadata_license_url=lic['url'],metadata_commercial_policy_potential=allowed_license(lic['url']),
     license_verified=False,rights_status='unverified_individual_source',creator=None,creator_group_verified=False,image_sha256=None,scene_verified=False,human_verified=False,
     usable_category_ids=usable,native_category_box_counts=dict(C.Counter(MAP[b['category_id']][1] for b in boxes)),native_annotation_ids=[b['annotation_id'] for b in boxes],
     indoor_evidence_rank=3 if {'bed','couch'}&context_names else 2 if indoor_context else 0,indoor_metadata_context=indoor_context,scene_context_category_count=len(context_names),outside_context=outside_context,
     metadata_eligible_for_rights_audit=not reasons,exclusion_reasons=reasons,occlusion_or_viewpoint_annotation='not supplied by COCO; unknown',
     source_attribution_status='Original asset URL and licence identifier only; creator name/profile absent from COCO instances schema')
    encoded=json.dumps(row,separators=(',',':'))+'\n';inventory_bytes+=len(encoded.encode())
    if inventory_bytes>70000000:raise RuntimeError('70MB inventory ceiling; stop before derived storage cap')
    output.write(encoded);inventory_count+=1
    if not reasons:candidates.append(row)
   print('AUDIT',part,summary['images'],'images',summary['annotations'],'annotations; furniture',len(byimage),'rights-audit candidates so far',len(candidates),flush=True)
 # Deterministic deduplicated known-source selection. Actual photos and creator
 # verification must precede any final acquisition/split freeze.
 selected=group_split(rank_select(candidates));selected_ids={r['coco_image_id'] for r in selected};details=C.defaultdict(list)
 for key,a in entries(OUT/'extracted/annotations/instances_train2017.json'):
  if key=='annotations' and a['image_id'] in selected_ids and a['category_id'] in MAP:details[a['image_id']].append(a)
 for row in selected:
  row['original_furniture_annotations']=details[row['coco_image_id']]
  row['annotation_source']='annotations/instances_train2017.json';row['annotation_source_sha256']=next(r['sha256'] for r in read(OUT/'extraction-complete.json')['members'] if r['file'].endswith('instances_train2017.json'))
  row['annotation_license']='https://creativecommons.org/licenses/by/4.0/'
  row['scope_policy']='Dining table subtype only; Desk/Cabinet/Lamp absent, unknown. Missing image labels are not confirmed negatives.'
  row['rights_verification_required_before_photo_approval']=True
 # The inventory is a rights-audit queue, not permission to fetch these images.
 plan=dict(schema='phase6b-unverified-rights-candidate-proposal-v1',seed=SEED,images=selected,image_count=len(selected),proposed_partition_counts=dict(C.Counter(r['proposed_split'] for r in selected)),selection_anchor_counts=dict(C.Counter(r['selection_anchor'] for r in selected)),
  selection_policy='Official train2017 only; CC0/CC BY metadata claims; minimum256px; large positive valid noncrowd polygon furniture boxes; indoor context heuristic; rarity/sofa-bed priority; one photo per known source group; deterministic seeded ties and anchor-stratified80/20 proposal.',
  photograph_acquisition_approved=False,ready_for_photograph_acquisition=False,individually_verified_rights=0,partition_status='Draft only: creator account grouping, current licensing, byte/pixel duplicates, actual scene/quality and capture relationships require later verification; no historical split changes.',
  expected_photo_transfer_scenario_bytes=len(selected)*250000,expected_photo_transfer_is_head_verified=False,suggested_future_photo_transfer_cap=1073741824,suggested_future_photo_storage_cap=2147483648,suggested_per_photo_max=3145728,suggested_max_attempts=2,
  next_approval='A bounded individual-source rights/creator audit and official image-object HEAD preflight; no photograph bodies. Then freeze a rights-verified exact manifest and request separate photo approval.')
 write(OUT/'proposed-image-candidates-v1.json',plan)
 write(OUT/'category-and-license-definitions.json',dict(categories=list(definitions.values()),licenses=list(license_definition.values()),unsupported=['desk','cabinet','lamp'],source_partitions_agree=True))
 write(OUT/'annotation-issues.json',bad_rows)
 write(OUT/'metadata-audit.json',dict(source_partitions=summaries,inventory_rows=inventory_count,inventory_bytes=inventory_bytes,metadata_rights_audit_candidate_images=len(candidates),unique_eligible_known_groups=len({r['identity_group'] for r in candidates}),
  exclusion_reason_counts=dict(exclusion),furniture_license_image_counts=furniture_license_counts,rights_verified=0,source_identity_group_audit=source_audit,duplicate_flickr_photo_id_values=sum(v>1 for v in all_photo.values()),
  source_confirmation='Archive hash+ZIP CRC+schema/geometry checks; no photograph content or source-page rights verification',annotation_issue_count=len(bad_rows),dataset_negative_policy='No absent furniture class is converted to confirmed negative; unsupported categories remain unknown.',
  mask_interpretation='Polygons are instance-mask encodings; RLE crowd regions are separate from ordinary single-instance masks; no raster masks/images downloaded or synthesized.',
  missing_attribution='COCO image entries do not provide creator name/profile or independent current licence evidence; all records remain individually rights-unverified.',
  viewpoint_and_occlusion='No per-object occlusion/truncation/viewpoint flags in this source schema; overlap/context/size are heuristics and independent captures remain unverified.'))
 print('PROPOSED',len(selected),'metadata-only candidates',plan['proposed_partition_counts'],'inventorybytes',inventory_bytes,flush=True)

if __name__=='__main__':run()
