"""Verify a matching individual Flickr ImageObject, never a dataset-wide assertion."""
import json
from html.parser import HTMLParser
from urllib.parse import urlparse
from network import *

class LD(HTMLParser):
    def __init__(self):super().__init__();self.active=False;self.parts=[];self.documents=[]
    def handle_starttag(self,tag,attrs):
        if tag=='script' and dict(attrs).get('type')=='application/ld+json':self.active=True;self.parts=[]
    def handle_data(self,data):
        if self.active:self.parts.append(data)
    def handle_endtag(self,tag):
        if tag=='script' and self.active:
            self.documents.append(json.loads(''.join(self.parts)));self.active=False

def photo_id(url):
    import re
    u=urlparse(url)
    if u.scheme!='https' or u.hostname not in ('www.flickr.com','flickr.com'):return None
    m=re.fullmatch(r'/photos/[^/]+/(\d+)/?',u.path)
    return m[1] if m else None

def individual_rights(text,row):
    parser=LD();parser.feed(text);objects=[]
    def visit(x):
        if isinstance(x,dict):
            if x.get('@type')=='ImageObject':objects.append(x)
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    for doc in parser.documents:visit(doc)
    ident=row['source_identity'].removeprefix('flickr-photo:')
    matching=[x for x in objects if photo_id(x.get('acquireLicensePage',''))==ident]
    if len(matching)!=1:raise ValueError('Missing/ambiguous matching individual photo licence evidence')
    evidence=matching[0];lic=evidence.get('license','');author=evidence.get('author',{})
    u=urlparse(lic)
    if u.scheme!='https' or u.hostname!='creativecommons.org' or u.path.rstrip('/')!='/licenses/by/2.0':raise ValueError('Individual source licence differs from approved CC BY 2.0')
    if not author.get('name') or not author.get('url'):raise ValueError('Missing individual attribution')
    owner=urlparse(author['url']);source=urlparse(evidence['acquireLicensePage'])
    if owner.hostname!=source.hostname or owner.path.rstrip('/')!=source.path.rstrip('/').rsplit('/',1)[0]:raise ValueError('Individual author account mismatch')
    return dict(status='verified_individual_current_source',photo_id=ident,license_url=lic,author_name=author['name'],author_url=author['url'],source_url=evidence['acquireLicensePage'],
                metadata_author_name=row['author'],metadata_author_profile_url=row['author_profile_url'],
                author_display_name_changed=author['name']!=row['author'],method='Matching ImageObject JSON-LD on original individual Flickr page; no embedded resources fetched',
                attribution_requirements='Retain author, individual source URL, CC BY 2.0 link and indication of CVDF resize / any orientation or crop changes')

def audit(resume=False):
    m=load_manifest();b=Budget(resume=resume);folder=b.root/'licensing';folder.mkdir(exist_ok=True)
    try:
        for i,row in enumerate(m['images']):
            out=folder/(row['image_id']+'.json')
            if out.exists():continue
            try:
                rec=b.fetch(row['source_landing_url'],'licensing/'+row['image_id']+'.response',1048576,reserve_after=42932701)
                if rec['status']!=200:raise ValueError('Individual source page HTTP '+str(rec['status']))
                evidence=individual_rights(b.decoded(rec),row)
                evidence.update(image_id=row['image_id'],proposed_split=row['proposed_split'],response_file=rec['file'],response_sha256=rec['sha256'],checked_utc=rec['utc'])
            except (ValueError,UnicodeError,json.JSONDecodeError) as exc:
                evidence=dict(image_id=row['image_id'],proposed_split=row['proposed_split'],status='quarantined_licensing',reason=str(exc),photograph_downloaded=False)
            atomic(out,evidence)
            if i%10==0:print('LICENCE',i+1,'/150',evidence['status'],'body bytes',b.state['actual_body_bytes'],flush=True)
        rows=[json.loads((folder/(row['image_id']+'.json')).read_text()) for row in m['images']]
        summary=dict(total=150,verified=sum(r['status']=='verified_individual_current_source' for r in rows),quarantined=sum(r['status']=='quarantined_licensing' for r in rows),body_bytes=b.state['actual_body_bytes'],images_downloaded=0,records=rows)
        atomic(b.root/'licensing-audit.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)
    finally:b.close()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args();audit(a.resume)
