"""Execute the exact approved Phase 3 plan; fail closed on unavailable/corrupt sources."""
import concurrent.futures as futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import urllib.error
import http.client
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase3-v1'))
import acquire as base
from common import *
from PIL import Image,ImageOps


def downloaded_integrity(d,obj,record,remote):
 path=d.dest/obj['destination']
 if path.stat().st_size!=remote['bytes'] or sha(path)!=record['sha256']:raise ValueError('Image size/SHA mismatch')
 with Image.open(path) as im:
  im.load()
  if im.width*im.height>4_194_304:raise ValueError('Unexpected small-image dimensions')
 etag=(remote.get('etag') or '').strip('"')
 if re.fullmatch(r'[0-9a-fA-F]{32}',etag):
  if hashlib.md5(path.read_bytes()).hexdigest().lower()!=etag.lower():raise ValueError('S3 ETag/MD5 mismatch')
  record['etag_md5_verified']=True
 record['full_decode_verified']=True;d.commit()


def main():
 verify_plan();plan=read(OUT/'acquisition-plan.json');approved=OUT/'acquisition-approval-v1.json';base.approval(plan,approved,OUT)
 dest=ROOT/plan['destination']
 if dest.exists() and not (dest/'transfer-ledger.json').exists():raise FileExistsError('Unbound destination exists; stop before modifying')
 dest.mkdir(exist_ok=True)
 with (dest/'acquisition.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d=base.Downloader(plan,dest);d.commit()
  def stop(*_):d.stop=True
  signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
  try:
   d.recover_commits();d.documentation()
   print(json.dumps({'stage':'official_licensing_verified','documents':d.state['documentation'],'observed_body_bytes':d.state['observed_body_bytes']}),flush=True)
   remote={}
   with futures.ThreadPoolExecutor(max_workers=8) as executor:
    pending={executor.submit(d.head,obj):obj for obj in plan['objects']}
    for completed in futures.as_completed(pending):
     if d.stop:
      for f in pending:f.cancel()
      raise InterruptedError('Preflight stopped')
     obj=pending[completed]
     try:remote[obj['image_id']]=completed.result()
     except Exception:
      for f in pending:f.cancel()
      raise
     if len(remote)%100==0:print(json.dumps({'stage':'HEAD','verified':len(remote),'total':len(plan['objects'])}),flush=True)
   preflight={'objects':remote,'actual_declared_total_image_bytes':sum(v['bytes'] for v in remote.values()),'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'no_redirects':True}
   prefile=dest/'remote-preflight-v1.json'
   if prefile.exists():
    old=read(prefile)['objects']
    if old!=remote:raise ValueError('Source object headers changed on resume')
   else:save(prefile,preflight)
   print(json.dumps({'stage':'all_sources_verified','image_bytes':preflight['actual_declared_total_image_bytes']}),flush=True)
   for n,obj in enumerate(plan['objects'],1):
    if d.stop:raise InterruptedError('Acquisition stopped; resume with same plan and limits')
    while True:
     try:
      record=d.get(obj,remote[obj['image_id']]);downloaded_integrity(d,obj,record,remote[obj['image_id']]);break
     except urllib.error.HTTPError as exc:
      exc.close();raise  # Unavailable source: never retry/replace it silently.
     except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.HTTPException):
      if len(d.state['objects'][obj['image_id']]['attempts'])>=plan['limits']['attempts_per_object']:raise
      print(json.dumps({'stage':'transport_retry','image_id':obj['image_id']}),flush=True)
     # Integrity/decoding/header/limit failures deliberately propagate without retry.
    if n%50==0:print(json.dumps({'stage':'GET','complete':n,'total':len(plan['objects']),'observed_body_bytes':d.state['observed_body_bytes']}),flush=True)
   save(dest/'acquisition-summary.json',{'status':'complete','products_requested':len(plan['products']),'images_downloaded':len(d.state['objects']),
    'observed_body_bytes':d.state['observed_body_bytes'],'conservative_charged_bytes':d.state['charged_bytes'],'allocated_storage_bytes':d.disk(),
    'plan_sha256':sha(OUT/'acquisition-plan.json'),'approval_sha256':sha(approved),'driver_sha256':sha(__file__),'all_image_hashes_full_decode_and_available_etag_md5_verified':True,
    'image_acceptance':'pending automatic quality/duplicate/weak-label screening; no training','finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
   print('Approved acquisition completed',flush=True)
  except Exception as exc:
   if isinstance(exc,urllib.error.HTTPError):exc.close()
   stopfile=dest/('stop-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
   save(stopfile,{'status':'stopped','reason':str(exc),'exception':type(exc).__name__,'observed_body_bytes':d.state['observed_body_bytes'],
    'conservative_charged_bytes':d.state['charged_bytes'],'downloaded_images':sum(r.get('status')=='downloaded' for r in d.state['objects'].values()),'resume_is_not_permission_to_ignore_stop_condition':True})
   raise

if __name__=='__main__':main()
