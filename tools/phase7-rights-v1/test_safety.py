import copy
import datetime as dt
import email.utils
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from audit_core import *
from source_evidence import classify,allowed_page
from transport import Transport,allowed_request,begin_candidate,provider,retry_not_before

class Response:
 status=429
 headers={'Retry-After':'600','Content-Type':'text/html'}
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read(self,*a):raise AssertionError('Error body must not be read')

class Tests(unittest.TestCase):
 def row(self):return manifest()['images'][0]
 def fixture(self,row=None,license='https://creativecommons.org/licenses/by/2.0/',author=True,photo=None):
  r=row or self.row();photo=photo or r['source_photo_id'].split(':')[1]
  obj={'@type':'ImageObject','acquireLicensePage':f'https://www.flickr.com/photos/owner/{photo}/','license':license,'name':'fixture photo'}
  if author:obj['author']={'name':'Fixture Owner','url':'https://www.flickr.com/photos/owner/'}
  return '<script type="application/ld+json">'+json.dumps(obj)+'</script>'
 def test_exact_400_and_frozen_allocations(self):
  m=manifest();self.assertEqual(len(m['images']),400);self.assertEqual(sum(r['proposed_partition']=='train' for r in m['images']),320)
 def test_photograph_GET_prohibited(self):
  r=self.row();self.assertTrue(allowed_request(r,r['official_image_object_url'],'HEAD'));self.assertFalse(allowed_request(r,r['official_image_object_url'],'GET'))
 def test_unapproved_source_and_wrong_photo_denied(self):
  r=self.row();self.assertFalse(allowed_request(r,'https://example.org/document','GET'));self.assertFalse(allowed_request(r,'https://www.flickr.com/photos/owner/1/','GET'));self.assertFalse(allowed_request(r,r['source_page_resolver'].replace('https:','http:'),'HEAD'))
 def test_resolver_redirect_bounded_same_photo(self):
  r=self.row();pid=r['source_photo_id'].split(':')[1];self.assertTrue(allowed_page(f'https://www.flickr.com/photo.gne?rb=1&id={pid}',pid));self.assertFalse(allowed_page(f'https://www.flickr.com/photo.gne?evil=1&id={pid}',pid))
 def test_one_candidate_attempt(self):
  state={'candidates':{}};begin_candidate(state,1)
  with self.assertRaises(ValueError):begin_candidate(state,1)
 def test_provider_aliases_same_circuit(self):self.assertEqual(provider('https://flickr.com/a'),provider('https://www.flickr.com/b'))
 def test_RetryAfter_seconds_and_date(self):
  t=dt.datetime(2026,11,6,tzinfo=dt.timezone.utc);self.assertEqual(dt.datetime.fromisoformat(retry_not_before('600',t))-t,dt.timedelta(seconds=600));self.assertEqual(dt.datetime.fromisoformat(retry_not_before(email.utils.format_datetime(t+dt.timedelta(hours=1)),t))-t,dt.timedelta(hours=1))
 def test_429_durably_blocks_not_yet_attempted_candidates(self):
  r=self.row();t=Transport.__new__(Transport);t.state=dict(candidates={},requests=[],circuits={},conservative_charge=0,actual_body_bytes=0,last_source_utc=None);t.save=lambda:None
  t.opener=SimpleNamespace(open=lambda *a,**k:Response());t.begin(r)
  with patch('transport.space'):
   response=t.request(r,r['source_page_resolver'],'HEAD')
  self.assertEqual(response['status'],429);self.assertTrue(t.circuit_open());self.assertEqual(t.state['actual_body_bytes'],0)
  restored=json.loads(json.dumps(t.state));other=copy.deepcopy(r);other['coco_image_id']+=1;t.state=restored
  with self.assertRaises(HostSuspended):t.begin(other)
  self.assertNotIn(str(other['coco_image_id']),t.state['candidates'])
 def test_photo_specific_license_and_creator(self):
  r=self.row();x=classify(self.fixture(),r,r['source_page_resolver']);self.assertEqual(x['licensing_status'],STATUSES[0]);self.assertEqual(x['creator'],'Fixture Owner')
 def test_other_photo_cannot_grant_rights(self):
  r=self.row();x=classify(self.fixture(photo='1'),r,r['source_page_resolver']);self.assertNotEqual(x['licensing_status'],STATUSES[0])
 def test_missing_creator_and_licence_never_verified(self):
  r=self.row()
  for text in (self.fixture(author=False),self.fixture(license=''),'<html>CC BY2.0</html>'):
   self.assertNotEqual(classify(text,r,r['source_page_resolver'])['licensing_status'],STATUSES[0])
 def test_noncommercial_license_restricted(self):
  r=self.row();x=classify(self.fixture(license='https://creativecommons.org/licenses/by-nc/2.0/'),r,r['source_page_resolver']);self.assertEqual(x['licensing_status'],STATUSES[2])
 def test_metadata_current_licence_conflict_quarantined(self):
  r=self.row();x=classify(self.fixture(license='https://creativecommons.org/licenses/by/4.0/'),r,r['source_page_resolver']);self.assertEqual(x['licensing_status'],STATUSES[4])

if __name__=='__main__':unittest.main()
