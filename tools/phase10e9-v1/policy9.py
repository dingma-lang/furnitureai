"""Bounded hosting/robots/licence-document checks before candidate page access."""
import urllib.parse,urllib.robotparser
from common9 import *
from transport9 import Transport,POLICIES
def document(t,url):
    for _ in range(3):
        r=t.request(None,url,'GET',policy=True)
        if r['status'] in (301,302,303,307,308):
            new=urllib.parse.urljoin(url,next((v for k,v in r['headers'].items() if k.lower()=='location'),''))
            if new not in POLICIES:return dict(available=False,reason='Unapproved policy redirect;not followed',request=r)
            url=new;continue
        return dict(available=r['status']==200,request=r,url=url)
    return dict(available=False,reason='Policy redirect bound exhausted')
def run():
    if read(REPORT/'CPU-tests.json')['status']!='PASS':raise PermissionError('CPU tests must pass before network')
    t=Transport();result=dict(source_terms_assessment_pending=True,candidate_source_requests_permitted=False,photo_bodies=0)
    try:
        robots=document(t,'https://www.flickr.com/robots.txt');result['robots']=robots
        if not robots['available']:
            result['reason']='Provider robots/access check unavailable;candidate requests not attempted'
        else:
            parser=urllib.robotparser.RobotFileParser();parser.parse(t.decoded(robots['request']).splitlines())
            rows=manifest()['records'];allowed_ids=[r['image_id'] for r in rows if parser.can_fetch(UA,r['original_source_page_resolver'])]
            result['robots_allowed_candidate_IDs']=allowed_ids
            if len(allowed_ids)!=64:
                t.stop('flickr.com','Robots policy denies approved audit client;provider permission or permitted access is required')
                result['reason']='Robots policy does not permit all frozen resolver paths;do not substitute or bypass'
            else:
                result['hosting_terms']=document(t,'https://www.flickr.com/help/terms/')
                if result['hosting_terms']['available']:
                    result['CC_BY2_document']=document(t,'https://creativecommons.org/licenses/by/2.0/legalcode.en')
                    result['reason']='Available policy documents require actual scope inspection before candidate requests'
                else:result['reason']='Current hosting terms unavailable;no photo rights inferred'
    except Exception as e:result['reason']=type(e).__name__+': '+str(e)
    finally:
        result['network']=t.finish();emit(OUT/'policy-access-preflight.json',result)
    print('Policy access preflight',result.get('reason'),'requests',len(t.requests),'circuits',list(t.circuits),flush=True)
if __name__=='__main__':run()
