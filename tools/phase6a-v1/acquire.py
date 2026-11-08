"""Future exact-manifest image acquisition. Requires separate explicit approval."""
import argparse
from core import *
from network import Budget,approval,head

def run(approval_path,resume=False):
 path=MANIFEST;approval(path,approval_path);manifest=read(path)
 if DEST.exists() and not resume:raise FileExistsError('New destination required; explicit --resume for tracked partial run')
 if not DEST.exists() and resume:raise FileNotFoundError('No acquisition to resume')
 isolation(manifest['images'],set(read(OUT/'protected-identities.json')))
 for row in manifest['images']:
  if not row['license_verified'] or sha(ROOT/row['license_record'])!=row['license_record_sha256']:raise ValueError('Unverified or changed rights record')
  record=read(ROOT/row['license_record'])
  if record['status']!='verified_individual_source_license' or license_url(record['license_url'])!=row['license_url']:raise ValueError('Unapproved rights')
  # Evidence must be recent at actual acquisition. Stale records require new
  # versioned rights audit/manifest and approval rather than silent alteration.
  checked=dt.datetime.fromisoformat(record['checked_utc'])
  if (dt.datetime.now(dt.timezone.utc)-checked).total_seconds()>7*86400:raise ValueError('Rights evidence older than seven days; reverify before acquisition')
 if not DEST.exists():DEST.mkdir(parents=True)
 binding=DEST/'manifest-binding.json'
 value=dict(manifest_sha256=sha(path),approval_sha256=sha(approval_path))
 if binding.exists():
  if read(binding)!=value:raise ValueError('Acquisition binding changed')
 else:write_new(binding,value)
 # Include all saved evidence and preparation in the same total new budget.
 b=Budget(DEST,cap=TRANSFER-manifest['rights_conservative_charge'],disk_cap=STORAGE-allocated(OUT)-allocated(HERE));records=[]
 try:
  for row in manifest['images']:
   filename=DEST/(row['image_id']+'.json')
   if filename.exists():
    r=read(filename)
    if sha(DEST/r['file'])!=r['sha256']:raise ValueError('Downloaded photo changed')
    records.append(r);continue
   remote=head(image_url(row));pinned=row['remote_object']
   if any(remote[k]!=pinned[k] for k in ('bytes','etag','url')):raise ValueError('Remote object changed since approval')
   relative=recovery_filename(b.state['requests'],remote['url'],row['image_id'])
   # A partial body remains immutable and charged. Resume restarts only the
   # same approved object, at most twice, never appends mixed versions.
   rec=b.request(remote['url'],relative,remote['bytes'],'image',expected=remote['bytes'],etag=remote['etag'])
   from PIL import Image
   with Image.open(DEST/relative) as im:im.verify()
   with Image.open(DEST/relative) as im:im.load();size=im.size
   if min(size)<=0:raise ValueError('Invalid decoded dimensions')
   r=dict(image_id=row['image_id'],file=relative,sha256=rec['sha256'],bytes=rec['actual_bytes'],width=size[0],height=size[1],source=row,eligible_for_training=False)
   write_new(filename,r);records.append(r)
  write_new(DEST/'acquisition-complete.json',dict(images=len(records),records=records,actual_photo_bytes=b.state['actual_bytes'],conservative_photo_charge=b.state['charged'],actual_total_body_bytes=b.state['actual_bytes']+manifest['rights_audit_body_bytes'],training_authorized=False))
 finally:b.close()

def recovery_filename(requests,url,image_id):
 """Recover a committed body even if its decoded-image record was interrupted."""
 attempts=[r for r in requests if r['url']==url]
 complete=[r for r in attempts if r.get('complete')]
 if complete:return complete[-1]['file']
 return f'photos/{image_id}-a{len(attempts)+1}.jpg'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.approval,a.resume)
