"""Run the exact400-ID rights audit. Only document GETs and photo HEADs."""
import argparse
import concurrent.futures as futures
from urllib.parse import urljoin
from common import *
from network import Transport,ResourceLimit,SourceRateLimited
from evidence import classify

def audit_one(transport,row,revision=1):
 target=OUT/('records' if revision==1 else 'records-v'+str(revision))/(row['image_id'].replace(':','-')+'.json')
 if target.exists():return read(target)
 previous=None
 if revision==3:
  old=OUT/'records-v2'/(row['image_id'].replace(':','-')+'.json');previous=read(old)
  if 'HTTP 429' not in previous.get('reason','') or transport.source_blocked:
   reused=dict(previous,revision3_note='Preserved original finding; not retried because not rate-limited or global source cooldown stop')
   commit(target,reused);return reused
 result=dict(image_id=row['image_id'],coco_image_id=row['coco_image_id'],source_identity=row['source_identity'],proposed_split=row['proposed_split'],manifest_sha256=PIN,
  metadata_license_url=row['metadata_license_url'],photograph_body_downloaded=False,checked_utc=now(),evidence_requests=[],image_object_status='not_checked',licensing_status=STATUSES[1])
 try:
  url=row['proposed_source_page_resolver'];photo=row['source_identity'].removeprefix('flickr-photo:')
  for hop in range(4):
   head=transport.request(row,url,'HEAD');result['evidence_requests'].append(dict(url=url,method='HEAD',status=head['status'],utc=head['checked_utc']))
   if head['status'] in (301,302,303,307,308):
    new=urljoin(url,head['headers'].get('Location',''))
    if not allowed_page(new,photo):raise ValueError('Redirect not an approved HTTPS same-photo canonical/resolver page')
    url=new;continue
   if head['status']!=200:
    result.update(licensing_status=STATUSES[3],reason='Original source unavailable/access restricted: HTTP '+str(head['status']));break
   if head['headers'].get('Content-Type','').split(';')[0] not in ('text/html','application/xhtml+xml'):
    result.update(licensing_status=STATUSES[1],reason='Source HEAD does not confirm an HTML document; no GET permitted');break
   page=transport.request(row,url,'GET');result['evidence_requests'].append(dict(url=url,method='GET',status=page['status'],utc=page['checked_utc'],file=page.get('file'),sha256=page.get('sha256')))
   if page['status']!=200:result.update(licensing_status=STATUSES[3],reason='Source document unavailable: HTTP '+str(page['status']));break
   result.update(classify(transport.decoded(page),row,url));break
  else:result.update(licensing_status=STATUSES[1],reason='Source redirect chain exceeds approved bounded resolution')
 except SourceRateLimited:
  if previous:
   reused=dict(previous,revision3_note='Source rate limit persisted; no further request attempted')
   commit(target,reused);return reused
  raise
 except ResourceLimit as exc:result.update(licensing_status=STATUSES[1],reason=str(exc),resource_limited=True)
 except OSError as exc:result.update(licensing_status=STATUSES[3],reason='Source unavailable or HTTPS validation/connection failed: '+str(exc))
 except (ValueError,RuntimeError,RecursionError) as exc:result.update(licensing_status=STATUSES[1],reason='Evidence unresolved: '+str(exc))
 # Object availability never upgrades or downgrades the copyright finding.
 try:
  image=transport.request(row,row['proposed_secure_image_url'],'HEAD');h=image['headers'];size=int(h.get('Content-Length','-1'))
  result['image_object']=dict(url=image['url'],status=image['status'],bytes=size,etag=h.get('ETag'),content_type=h.get('Content-Type'),checked_utc=image['checked_utc'],body_bytes=0,method='HEAD')
  result['image_object_status']='available' if image['status']==200 and 0<size<=PHOTO_MAX and h.get('ETag') and h.get('Content-Type','').split(';')[0]=='image/jpeg' else 'unavailable_or_outside_photo_limit'
 except (OSError,ValueError,RuntimeError) as exc:result.update(image_object_status='unavailable_or_unverified',image_object_error=str(exc))
 commit(target,result);return result

def run(resume=False,revision=1):
 m=manifest();transport=Transport(resume=resume,retry_429=revision==3)
 try:
  auth=OUT/'authorization.json'
  if not auth.exists():commit(auth,dict(user_approved_rights_audit=True,authorization='Explicit user approval: Furniture AI Phase6C COCO individual photograph rights audit',manifest_sha256=PIN,proposal_sha256=sha(PROPOSAL),recorded_utc=now(),source_ids=400,train_slots=320,development_slots=80,max_transfer=CAP,max_storage=DISK,photo_bodies_authorized=False,models_authorized=False))
  rows=m['images'];results={}
  with futures.ThreadPoolExecutor(max_workers=1 if revision==3 else 4) as pool:
   pending={pool.submit(audit_one,transport,row,revision):row for row in rows}
   for job in futures.as_completed(pending):
    result=job.result();results[result['image_id']]=result
    if len(results)%20==0:print('AUDITED',len(results),'/400','verified',sum(r['licensing_status']==STATUSES[0] for r in results.values()),'body bytes',transport.state['actual_body_bytes'],'charge',transport.state['conservative_charge'],flush=True)
  commit(OUT/('rights-audit.json' if revision==1 else 'rights-audit-v'+str(revision)+'.json'),dict(records=[results[r['image_id']] for r in rows],manifest_sha256=PIN,actual_body_bytes=transport.state['actual_body_bytes'],conservative_charge=transport.state['conservative_charge'],photo_bodies_downloaded=0,training_or_inference=0))
 finally:transport.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--revision',type=int,choices=(1,2,3),default=1);a=p.parse_args();run(a.resume,a.revision)
