"""Acquire only individually licence-verified photos from the frozen approved list."""
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from network import *
from licensing import individual_rights

META=ROOT/'data/open-images/metadata/phase5-v1'

def head(row):
    url=f"https://open-images-dataset.s3.amazonaws.com/{row['source_partition']}/{row['image_id']}.jpg"
    if url!=row['image_url'] or not re.fullmatch('[0-9a-f]{16}',row['image_id']):raise ValueError('Unapproved image object')
    with urllib.request.build_opener(NoRedirect()).open(urllib.request.Request(url,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=45) as r:
        if r.status!=200:raise ValueError('Image source unavailable')
        h=dict(url=url,status=r.status,bytes=int(r.headers.get('Content-Length','-1')),etag=r.headers.get('ETag'),content_type=r.headers.get('Content-Type'),body_bytes=0)
        if not 0<h['bytes']<=PHOTO_MAX or h['content_type']!='image/jpeg' or not re.fullmatch('"[0-9a-f]{32}"',h['etag'] or ''):raise ValueError('Unexpected source size/type/checksum')
        return row['image_id'],h

def verify_jpeg(path):
    from PIL import Image
    with Image.open(path) as im:
        if im.format!='JPEG':raise ValueError('Expected JPEG')
        if max(im.size)>4096 or im.width*im.height>16777216:raise ValueError('Image dimension safety limit')
        im.verify()
    with Image.open(path) as im:
        im.load();return dict(width=im.width,height=im.height,exif_orientation=im.getexif().get(274,1))

def run():
    m=load_manifest();b=Budget(resume=True)
    try:
        licence=json.loads((b.root/'licensing-audit.json').read_text());rights={r['image_id']:r for r in licence['records']}
        if set(rights)!={r['image_id'] for r in m['images']}:raise ValueError('Licensing/membership mismatch')
        rows=[r for r in m['images'] if rights[r['image_id']]['status']=='verified_individual_current_source']
        # Reparse each saved individual licence proof before any photograph GET.
        for row in rows:
            rec=next(r for r in b.state['requests'] if r.get('file')==rights[row['image_id']]['response_file'] and r['complete'])
            if sha(b.root/rec['file'])!=rec['sha256']:raise ValueError('Licence evidence changed')
            if individual_rights(b.decoded(rec),row)['license_url']!=row['license_url']:raise ValueError('Licence evidence changed')
        old=json.loads((META/'pilot-proposal-v4/image-object-head-preflight.json').read_text())
        old_byid={r['image_id']:r for r in old['objects']}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:heads=dict(pool.map(head,rows))
        for ident,h in heads.items():
            prior=old_byid[ident]
            if h['bytes']!=prior['bytes'] or h['etag']!=prior['etag'] or h['url']!=prior['url']:raise ValueError('Approved photo object version changed')
        if b.state['charged_bytes']+sum(r['bytes'] for r in heads.values())>CAP:raise RuntimeError('Insufficient cumulative body budget')
        path=b.root/'image-source-preflight.json'
        if not path.exists():atomic(path,dict(manifest_sha256=PIN,heads=heads,utc=dt.datetime.now(dt.timezone.utc).isoformat(),eligible_objects=len(rows),licensing_quarantined_objects=150-len(rows)))
        records=[]
        for i,row in enumerate(rows):
            h=heads[row['image_id']];relative='photos/'+row['image_id']+'.jpg'
            # Stop on corruption/changed source. Only ordinary connection/timeout errors
            # can consume the second attempt, retaining the first partial response.
            try:rec=b.fetch(h['url'],relative,h['bytes'],kind='image',expected=h['bytes'],etag=h['etag'])
            except (TimeoutError,ConnectionError) as exc:
                rec=b.fetch(h['url'],'photos/'+row['image_id']+'-attempt2.jpg',h['bytes'],kind='image',expected=h['bytes'],etag=h['etag'])
            p=b.root/rec['file']
            if hashlib.md5(p.read_bytes()).hexdigest()!=h['etag'].strip('"'):raise ValueError('JPEG source MD5 integrity failure')
            dimensions=verify_jpeg(p)
            result=dict(image_id=row['image_id'],proposed_split=row['proposed_split'],file=rec['file'],sha256=rec['sha256'],bytes=rec['actual_body_bytes'],source_url=h['url'],source_etag=h['etag'],**dimensions,licence_evidence='licensing/'+row['image_id']+'.json')
            target=b.root/'integrity'/(row['image_id']+'.json');target.parent.mkdir(exist_ok=True)
            if target.exists():
                if json.loads(target.read_text())!=result:raise ValueError('Completed image integrity stage changed')
            else:atomic(target,result)
            records.append(result)
            if i%10==0:print('PHOTO',i+1,'/',len(rows),'total response bytes',b.state['actual_body_bytes'],flush=True)
        atomic(b.root/'acquisition-summary.json',dict(approved_images=150,downloaded_images=len(records),licensing_quarantine=150-len(records),
             photo_body_bytes=sum(r['actual_body_bytes'] for r in b.state['requests'] if r['kind']=='image'),
             all_response_body_bytes=b.state['actual_body_bytes'],charged_bytes=b.state['charged_bytes'],
             proposed_membership_preserved=True,substitutions=0,records=records))
        print('COMPLETE',len(records),'photographs',b.state['actual_body_bytes'],'total response bytes',flush=True)
    finally:b.close()

if __name__=='__main__':run()
