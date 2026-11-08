"""One new evidence attempt per exact candidate; host429 leaves others unattempted."""
import argparse
import fcntl
from urllib.parse import urljoin
from audit_core import *
from transport import Transport,allowed_request,retry_not_before
from source_evidence import allowed_page,classify

def base_record(row):
 return dict(coco_image_id=row['coco_image_id'],lvis_image_id=row['lvis_image_id'],source_photo_id=row['source_photo_id'],
  provisional_partition=row['proposed_partition'],manifest_sha256=PIN,licensing_status=STATUSES[1],
  attempted=False,evidence_attempts=0,requests=[],checked_utc=now(),creator=None,license_url=None,
  annotation_license=row['annotation_license'],annotation_archive_sha256=row['annotation_archive_sha256'],
  human_reviewed=False,photo_body_downloaded=False,image_object_status='not_checked')

def one(transport,row):
 result=base_record(row);transport.begin(row);result.update(attempted=True,evidence_attempts=1)
 url=row['source_page_resolver'];photo=row['source_photo_id'].split(':')[1]
 try:
  for hop in range(4):
   head=transport.request(row,url,'HEAD');result['requests'].append(head)
   if head['status'] in (301,302,303,307,308):
    new=urljoin(url,head['headers'].get('Location',''))
    if not allowed_page(new,photo):raise ValueError('Redirect does not preserve the exact approved photo ID/source')
    url=new;continue
   if head['status']!=200:
    result.update(licensing_status=STATUSES[3],reason=f"Source HEAD HTTP{head['status']}; no GET or retry");break
   if head['headers'].get('Content-Type','').split(';')[0] not in ('text/html','application/xhtml+xml'):raise ValueError('Source does not identify an HTML document')
   page=transport.request(row,url,'GET');result['requests'].append(page)
   if page['status']!=200:
    result.update(licensing_status=STATUSES[3],reason=f"Source GET HTTP{page['status']}; no retry");break
   result.update(classify(transport.decoded(page),row,url));result.update(evidence_url=url,evidence_file=page['file'],evidence_sha256=page['sha256'],evidence_checked_utc=page['checked_utc']);break
  else:result.update(licensing_status=STATUSES[4],reason='Bounded same-photo redirect resolution exhausted')
  # Object HEAD does not establish copyright; only check after independent rights evidence passes.
  if result['licensing_status']==STATUSES[0]:
   try:
    h=transport.request(row,row['official_image_object_url'],'HEAD');result['requests'].append(h)
    size=int(h['headers'].get('Content-Length','-1'))
    result['image_object']=dict(url=h['url'],bytes=size,etag=h['headers'].get('ETag'),content_type=h['headers'].get('Content-Type'),status=h['status'],checked_utc=h['checked_utc'])
    result['image_object_status']='available' if h['status']==200 and 0<size<=3145728 and h['headers'].get('ETag') and h['headers'].get('Content-Type','').split(';')[0]=='image/jpeg' else 'unverified_or_outside_proposed_photo_bound'
   except (OSError,ValueError,RuntimeError) as e:result.update(image_object_status='not_available_or_verified',image_object_error=str(e))
 except ResourceLimit:raise
 except (OSError,ValueError,RuntimeError,RecursionError) as e:
  result.update(licensing_status=STATUSES[3] if isinstance(e,OSError) else STATUSES[4],reason=str(e))
 transport.state['candidates'][str(row['coco_image_id'])].update(complete=True,ended_utc=now());transport.save();return result

def run(resume=False):
 rows=manifest()['images'];auth=read(OUT/'authorization.json')
 if auth.get('manifest_sha256')!=PIN or not auth.get('rights_audit_approved'):raise PermissionError('Explicit approved scope required')
 if read(OUT/'preflight.json')['status']!='PASS':raise ValueError('Preflight must pass before network')
 with (OUT/'audit.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);transport=Transport(resume)
  prior=read(OUT/'preflight.json')['previous_rate_limit']
  if prior:
   not_before=dt.datetime.fromisoformat(retry_not_before(prior.get('headers',{}).get('Retry-After'),dt.datetime.fromisoformat(prior['checked_utc'])))
   if dt.datetime.now(dt.timezone.utc)<not_before:
    transport.state['circuits']['flickr.com']=dict(open=True,reason='Historical Retry-After window remains active',not_before=not_before.isoformat(),separate_future_approval_required=True);transport.save()
  results=[];stopped_for_resource=False
  for row in rows:
   target=OUT/'records'/f"{row['coco_image_id']:012d}.json"
   if target.exists():results.append(read(target));continue
   if transport.circuit_open() or stopped_for_resource:
    result=base_record(row);result.update(licensing_status=STATUSES[3] if not stopped_for_resource else STATUSES[1],reason='Not attempted: provider suspension' if not stopped_for_resource else 'Not attempted: resource safety stop',future_separate_approval_required=True)
   elif str(row['coco_image_id']) in transport.state['candidates']:
    # Never repeat an in-flight or consumed attempt after a crash.
    result=base_record(row);result.update(attempted=True,evidence_attempts=1,licensing_status=STATUSES[4],reason='Interrupted prior attempt retained; no second attempt allowed')
   else:
    try:result=one(transport,row)
    except ResourceLimit as e:
     stopped_for_resource=True;result=base_record(row);consumed=str(row['coco_image_id']) in transport.state['candidates'];result.update(attempted=consumed,evidence_attempts=int(consumed),licensing_status=STATUSES[1],reason=str(e))
   commit(target,result);results.append(result)
   if len(results)%20==0:print('RECORDED',len(results),'/400','attempted',sum(r['attempted'] for r in results),'verified',sum(r['licensing_status']==STATUSES[0] for r in results),'bytes',transport.state['actual_body_bytes'],flush=True)
  if not (OUT/'rights-audit.json').exists():commit(OUT/'rights-audit.json',dict(records=results,manifest_sha256=PIN,actual_body_bytes=transport.state['actual_body_bytes'],conservative_charge=transport.state['conservative_charge'],circuits=transport.state['circuits'],resource_safety_stop=stopped_for_resource,photograph_bodies=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.resume)
