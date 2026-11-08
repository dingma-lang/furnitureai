"""HEAD-only object-size quote for a proposed pilot. NEVER requests image bodies."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
import prepare_pilot as p
from acquire import NoRedirect


def head(row):
    expected=f"https://open-images-dataset.s3.amazonaws.com/{row['source_partition']}/{row['image_id']}.jpg"
    if row['image_url']!=expected or row['source_partition'] not in ('validation','test') or not re.fullmatch('[0-9a-f]{16}',row['image_id']):raise ValueError('Unapproved photo object path')
    opener=urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(urllib.request.Request(expected,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=30) as response:
            length=int(response.headers.get('Content-Length','-1'))
            return dict(image_id=row['image_id'],url=expected,status=response.status,bytes=length,
                        content_type=response.headers.get('Content-Type'),etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),method='HEAD',body_bytes=0)
    except (urllib.error.URLError,TimeoutError,ValueError) as e:
        return dict(image_id=row['image_id'],url=expected,status=getattr(e,'code',None),error=str(e),method='HEAD',body_bytes=0)


def main():
    folder=p.META/'pilot-proposal-v4';manifest=folder/'proposed-image-acquisition.json';rows=p.d.read(manifest)['images']
    if not len(rows)==150:raise ValueError('Expected bounded 150-object proposal')
    out=folder/'image-object-head-preflight.json'
    if out.exists():raise FileExistsError('Do not replace recorded preflight')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(head,rows))
    failed=[r for r in results if r.get('status')!=200 or r.get('bytes',0)<=0 or r.get('content_type') not in ('image/jpeg','image/jpg')]
    report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=p.d.sha(manifest),objects=results,
                object_count=len(results),failed_count=len(failed),expected_image_bytes=sum(r.get('bytes',0) for r in results if r.get('status')==200),
                maximum_object_bytes=max(r.get('bytes',0) for r in results),image_bodies_downloaded=0,
                individual_source_license_reverification='pending; no Flickr pages or photograph bodies fetched')
    p.d.write_new(out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='objects'},indent=2))
    if failed:raise RuntimeError('Image source availability unresolved; see preserved HEAD report')

if __name__=='__main__':main()
