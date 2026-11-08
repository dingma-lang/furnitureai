"""Certificate-verified source HEADs and bounded official licence/checksum documents."""
import ssl
import urllib.request
from core import *
from acquire import NoRedirect,headers_valid

DOCS={
 'official-lvis-site.js':'https://www.lvisdataset.org/static/js/main.18748841.chunk.js',
 'tensorflow-checksums.tsv':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/lvis/checksums.tsv',
}

def run():
 p=proposal();space(p['expected_body_bytes']+64*2**20)
 if (OUT/'preflight.json').exists():raise FileExistsError('Preflight already recorded; no repeated source GET')
 context=ssl.create_default_context();assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED
 opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context))
 docs=[];total=0;published={}
 for name,url in DOCS.items():
  record=dict(url=url,checked_utc=now(),tls_verified=True)
  try:
   with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=40) as response:
    data=response.read(1000001);total+=len(data)
    if len(data)>1000000 or total>2000000:raise RuntimeError('Documentation bound exceeded')
    data.decode('utf-8')
    with (OUT/name).open('xb') as f:f.write(data)
    record.update(bytes=len(data),sha256=sha(OUT/name),file=name,status=response.status)
   if name=='official-lvis-site.js':
    if record['sha256']!=sha(ROOT/'reports/phase7-data-strategy-v1/sources/lvis-main.js'):raise ValueError('Official licensing/source document changed')
    if b'https://creativecommons.org/licenses/by/4.0/' not in data or any(r['url'].encode() not in data for r in p['objects']):raise ValueError('Missing expected source/licence text')
   else:
    for line in data.decode().splitlines():
     parts=line.split()
     for obj in p['objects']:
      if parts and parts[0]==obj['url']:
       if len(parts)<3 or int(parts[1])!=obj['bytes'] or len(parts[2])!=64:raise ValueError('Independent source size/checksum inconsistency')
       published[obj['expected_local_filename']]=parts[2]
  except urllib.error.HTTPError as e:
   if name=='official-lvis-site.js':raise
   # Optional checksum documentation unavailable is explicitly recorded, not a mirror fallback.
   record.update(status=e.code,error=str(e),body_bytes_read=0)
  docs.append(record)
 heads=[]
 for r in p['objects']:
  with opener.open(urllib.request.Request(r['url'],method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=40) as h:
   headers_valid(h,r);heads.append(dict(url=r['url'],status=h.status,headers=dict(h.headers),checked_utc=now(),tls_verified=True))
 commit(OUT/'preflight.json',dict(proposal_sha256=PIN,heads=heads,documents=docs,document_body_bytes=total,published_sha256=published,
  extracted_metadata_bytes=0,processing='ZIP stream only; no full extraction; 32MiB allocated-space reserve',annotation_license='https://creativecommons.org/licenses/by/4.0/',
  photo_licensing='No photograph rights granted by LVIS annotation licence',checksum_caveat='Multipart ETag is not whole-file MD5; independent checksums used only where exact source/size matches.'))
 print('PREFLIGHT PASS',total,'documentation body bytes; independent hashes',published,flush=True)

if __name__=='__main__':run()
