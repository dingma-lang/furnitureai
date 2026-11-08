"""Known source/creator grouping without photograph bodies or partition moves."""
import collections as C
import csv
from common import *

class Union:
 def __init__(self):self.parent={}
 def find(self,x):
  self.parent.setdefault(x,x)
  while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
  return x
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def source_id(url):
 u=urlparse(url)
 if u.scheme in ('https','http') and u.hostname in ('flickr.com','www.flickr.com'):
  m=re.fullmatch(r'/photos/[^/]+/(\d+)/?',u.path)
  if m:return 'flickr-photo:'+m[1]
 return None

def build_graph(records):
 uf=Union();metadata=ROOT/'data/open-images/metadata/phase5-v1';photo_by_oi={};url_owners=C.defaultdict(list);count=0
 for part in ('validation','test'):
  with (metadata/(part+'-images-with-rotation.csv')).open(newline='') as f:
   for r in csv.DictReader(f):
    count+=1;photo=source_id(r['OriginalLandingURL']);node='openimages:'+r['ImageID']
    if photo:uf.join(node,photo);photo_by_oi[r['ImageID']]=photo
    for alias in (account(r['AuthorProfileURL']),account(r['OriginalLandingURL'])):
     if alias:uf.join(node,alias)
    if r['OriginalMD5']:uf.join(node,'original-md5:'+r['OriginalMD5'])
    url_owners[r['OriginalURL']].append(r['ImageID']);url_owners[r['OriginalLandingURL']].append(r['ImageID'])
 protected_ids=[]
 for manifest_path,rights_path in [(metadata/'pilot-proposal-v4/proposed-image-acquisition.json',ROOT/'data/open-images/image-pilots/phase5-v1/licensing-audit.json'),(ROOT/'data/open-images/preparation/phase6a-v1/acquisition-manifest-v2.json',ROOT/'data/open-images/preparation/phase6a-v1/license-attribution-ledger.json')]:
  images=read(manifest_path)['images'];protected_ids.extend(r['image_id'] for r in images)
  evidence={r['image_id']:r for r in read(rights_path)['records']}
  for r in images:
   photo=r['source_identity'];uf.join('openimages:'+r['image_id'],photo)
   lic=evidence.get(r['image_id'],{})
   for key in ('creator_url','author_url','source_url'):
    alias=account(lic.get(key,''))
    if alias:uf.join(photo,alias)
   for key in ('author_profile_url','source_landing_url'):
    alias=account(r.get(key,''))
    if alias:uf.join(photo,alias)
 for r in records:
  for alias in r.get('creator_aliases',[]):uf.join(r['source_identity'],alias)
 protected={uf.find('openimages:'+i) for i in protected_ids}
 return uf,protected,dict(full_historical_source_records=count,protected_pilot_ids=len(protected_ids),protected_known_source_groups=len(protected),history_metadata_only=True),url_owners

def finalize(records,rows):
 uf,protected,history,known_urls=build_graph(records);groups=C.defaultdict(list);byid={r['image_id']:r for r in rows}
 for r in records:groups[uf.find(r['source_identity'])].append(r)
 existing_names=C.defaultdict(list)
 for base in (ROOT/'data',ROOT/'reports',ROOT/'assets',ROOT/'samples'):
  if base.exists():
   for p in base.rglob('*'):
    if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):existing_names[p.name].append(str(p.relative_to(ROOT)))
 decisions=[];overlaps=[]
 for r in records:
  original=byid[r['image_id']];group=uf.find(r['source_identity']);members=groups[group];parts={x['proposed_split'] for x in members};flags=[]
  if group in protected:flags.append('historical_known_source_or_creator_group')
  if len(parts)>1:flags.append('current_creator_or_known_source_group_crosses_original_partitions')
  if original['file_name'] in existing_names:flags.append('existing_local_filename_collision_unresolved')
  # URL matches to unrelated unconsumed OI metadata are recorded, not silently
  # treated as training leakage or proof of an independent physical identity.
  matches=known_urls.get(original['original_flickr_asset_url'],[])
  available=r['image_object_status']=='available'
  eligible=r['licensing_status']==STATUSES[0] and available and not flags
  result=dict(image_id=r['image_id'],source_identity=r['source_identity'],proposed_split=r['proposed_split'],known_source_group='phase6c-group:'+hashlib.sha256(group.encode()).hexdigest()[:24],
   creator_aliases=r.get('creator_aliases',[]),source_group_members=len(members),group_spans_original_partitions=len(parts)>1,licensing_status=r['licensing_status'],image_object_status=r['image_object_status'],
   eligible_for_image_acquisition_proposal=eligible,quarantined=not eligible,identity_flags=flags,known_exact_original_url_metadata_matches=matches,local_filename_matches=existing_names.get(original['file_name'],[]),
   physical_product_identity='Unverified; a shared creator/source group is conservative partition isolation, not a claim these photos show the same physical product.')
  decisions.append(result)
  if flags:overlaps.append(result)
 # Equal image ETag/size is only a header-level suspicion, not a body hash or
 # perceptual result. Quarantine suspicious cross-partition acquisitions.
 header_groups=C.defaultdict(list)
 for r in records:
  obj=r.get('image_object',{})
  if r['image_object_status']=='available':header_groups[(obj['etag'],obj['bytes'])].append(r['image_id'])
 header_suspicions=[ids for ids in header_groups.values() if len(ids)>1]
 for ids in header_suspicions:
  for d in decisions:
   if d['image_id'] in ids:d['identity_flags'].append('equal_etag_and_size_unverified_asset_overlap');d['eligible_for_image_acquisition_proposal']=False;d['quarantined']=True
 commit(OUT/'identity-audit.json',dict(records=decisions,historical_scope=history,overlap_records=overlaps,header_asset_suspicions=header_suspicions,
  original_allocation_preserved=True,partition_reassignments=0,photo_body_hash_or_perceptual_screening_performed=False,visual_quality_checked=False,
  limitations='Current creator aliases are linked only through matching same-photo source evidence; unresolved aliases or cross-author physical relationships remain unknown. All mixed-source groups are quarantined without shifting members.'))
 return decisions
