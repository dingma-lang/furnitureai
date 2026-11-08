"""Individual rights evidence audit and final manifest, never photograph GETs."""
import argparse
import collections as C
from html.parser import HTMLParser
from core import *
from network import Budget,head

class LD(HTMLParser):
 def __init__(self):super().__init__();self.active=False;self.parts=[];self.docs=[]
 def handle_starttag(self,tag,attrs):
  if tag=='script' and dict(attrs).get('type')=='application/ld+json':self.active=True;self.parts=[]
 def handle_data(self,data):
  if self.active:self.parts.append(data)
 def handle_endtag(self,tag):
  if tag=='script' and self.active:self.docs.append(json.loads(''.join(self.parts)));self.active=False

def verify(text,row):
 p=LD();p.feed(text);found=[]
 def visit(x):
  if isinstance(x,dict):
   if x.get('@type')=='ImageObject' and photo_id(x.get('acquireLicensePage',''))==row['source_identity'].removeprefix('flickr-photo:'):found.append(x)
   for v in x.values():visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 for doc in p.docs:visit(doc)
 if len(found)!=1:raise ValueError('Missing or ambiguous individual photo evidence')
 evidence=found[0];lic=license_url(evidence.get('license',''));author=evidence.get('author',{})
 # Licence changes are quarantined rather than replacing historical metadata.
 if lic!=license_url(row['license_url']):raise ValueError('Licence differs from source metadata')
 if not author.get('name') or not author.get('url'):raise ValueError('Missing creator credit')
 source=urlparse(evidence['acquireLicensePage']);owner=urlparse(author['url'])
 if owner.scheme!='https' or owner.hostname!=source.hostname or owner.path.rstrip('/')!=source.path.rstrip('/').rsplit('/',1)[0]:raise ValueError('Creator account identity mismatch')
 return dict(status='verified_individual_source_license',creator=author['name'],creator_url=author['url'],source_url=evidence['acquireLicensePage'],license_url=lic,
  checked_utc=now(),verification_method='Matching original Flickr ImageObject photo identity, creator account and allowed license; no embedded photographs fetched',
  attribution='Credit creator and photo title; link original photo and exact CC licence; record CVDF resize, rotation and all subsequent crop/annotation changes.',
  other_rights='No assertion of model/property/privacy releases; commercially permissive copyright licence evidence only')

def run(max_checks=500):
 manifest=read(OUT/'candidate-manifest.json');folder=OUT/'rights-audit';folder.mkdir(exist_ok=True)
 b=Budget(folder,cap=192*2**20,disk_cap=512*2**20);records=[];accepted=[]
 try:
  for row in manifest['images'][:max_checks]:
   f=folder/(row['image_id']+'.json')
   if f.exists():record=read(f)
   else:
    try:
     rec=b.request(row['source_landing_url'],'responses/'+row['image_id']+'-a1.html',1048576,'rights')
     record=verify(b.decoded(rec),row)
     record.update(response_file=rec['file'],response_sha256=rec['sha256'],response_body_bytes=rec['actual_bytes'])
     record['remote_object']=head(row['image_url'])
    except (ValueError,OSError,RuntimeError,json.JSONDecodeError) as exc:record=dict(status='quarantined',reason=str(exc),checked_utc=now())
    record.update(image_id=row['image_id'],source_identity=row['source_identity'],photograph_downloaded=False)
    write_new(f,record)
   records.append(record)
   if record['status']=='verified_individual_source_license':accepted.append(dict(row,license_verified=True,license_record=str(f.relative_to(ROOT)),license_record_sha256=sha(f),remote_object=record['remote_object']))
   if len(records)%10==0:print('RIGHTS',len(records),'eligible',len(accepted),'body bytes',b.state['actual_bytes'],flush=True)
   if len(accepted)>=400:break
  if not (OUT/'license-attribution-ledger.json').exists():write_new(OUT/'license-attribution-ledger.json',dict(records=records,actual_bytes=b.state['actual_bytes'],conservatively_charged_bytes=b.state['charged'],photographs_downloaded=0,eligible_creators_individually_verified=bool(accepted),all_checked_sources_verified=len(accepted)==len(records)))
 finally:b.close()
 if not accepted:print('No source-verified photographs; acquisition remains blocked');return
 # Frozen metadata-derived group assignment precedes visual verification; no later
 # rebalancing using model outputs is allowed. 80/20 deterministic group split.
 for row in accepted:
  n=int(hashlib.sha256((SEED+'|partition|'+row['identity_group']).encode()).hexdigest(),16)%100
  row.update(proposed_split='development' if n<20 else 'train',acquisition_status='awaiting_separate_approval',image_sha256=None)
 isolation(accepted,set(read(OUT/'protected-identities.json')))
 expected=sum(r['remote_object']['bytes'] for r in accepted)
 if 2*expected+read(OUT/'license-attribution-ledger.json')['conservatively_charged_bytes']>TRANSFER:raise RuntimeError('Two-attempt transfer budget cannot be guaranteed')
 proposal=dict(schema='phase6a-finalized-acquisition-v1',seed=SEED,images=accepted,image_count=len(accepted),expected_photo_bytes=expected,
  maximum_two_attempt_photo_bytes=2*expected,max_cumulative_transfer=TRANSFER,max_allocated_storage=STORAGE,per_image_max=IMAGE_MAX,max_attempts=2,destination=str(DEST),
  rights_audit_body_bytes=read(OUT/'license-attribution-ledger.json')['actual_bytes'],rights_conservative_charge=read(OUT/'license-attribution-ledger.json')['conservatively_charged_bytes'],
  categories={p:{c:dict(images=sum(bool(r['eligible_box_counts'].get(c)) for r in accepted if r['proposed_split']==p),ordinary_eligible_boxes=sum(r['eligible_box_counts'].get(c,0) for r in accepted if r['proposed_split']==p),native_boxes=sum(r['native_box_counts'].get(c,0) for r in accepted if r['proposed_split']==p)) for c in CLASSES} for p in ('train','development')},
  screening_pending=['actual hashes and decoding','exact/perceptual duplicates including historical signatures','indoor scene suitability','Desk/Cabinetry scope and hierarchy coverage','unannotated regions are unknown'],training_authorized=False,image_acquisition_authorized=False)
 write_new(OUT/'acquisition-manifest.json',proposal)
 print('FINAL MANIFEST',len(accepted),'expected bytes',expected,'partitions',dict(C.Counter(r['proposed_split'] for r in accepted)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--verify-source-pages',action='store_true');ap.add_argument('--max-checks',type=int,default=500);a=ap.parse_args()
 if not a.verify_source_pages:raise SystemExit('No network dry run: pass --verify-source-pages for individual rights HTML + object HEAD only. Photographs are never fetched by this program.')
 if not 0<a.max_checks<=500:raise SystemExit('Bounded to 500 source pages')
 run(a.max_checks)
