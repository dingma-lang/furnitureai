import tempfile
import unittest
from unittest.mock import patch
from common import *
from evidence import classify
from network import Transport,NoRedirect,ResourceLimit

class Tests(unittest.TestCase):
 def setUp(self):self.row=manifest()['images'][0];self.photo=self.row['source_identity'].removeprefix('flickr-photo:');self.url='https://www.flickr.com/photos/owner/'+self.photo
 def doc(self,lic='https://creativecommons.org/licenses/by/2.0/',photo=None,owner='owner'):
  obj={'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/'+owner+'/'+(photo or self.photo),'license':lic,'author':{'name':'Public creator','url':'https://www.flickr.com/photos/owner'}}
  return '<meta property="og:title" content="A supplied title"><script type="application/ld+json">'+json.dumps(obj)+'</script>'
 def test_frozen_manifest(self):self.assertEqual(len(manifest()['images']),400)
 def test_individual_id_creator_version_and_title(self):
  r=classify(self.doc(),self.row,self.url);self.assertEqual(r['licensing_status'],STATUSES[0]);self.assertEqual(r['title'],'A supplied title')
 def test_wrong_photo_does_not_verify(self):self.assertEqual(classify(self.doc(photo='999'),self.row,self.url)['licensing_status'],STATUSES[1])
 def test_creator_mismatch_unverified(self):self.assertEqual(classify(self.doc(owner='other'),self.row,self.url)['licensing_status'],STATUSES[1])
 def test_restricted_and_changed_licences(self):
  self.assertEqual(classify(self.doc(lic='https://creativecommons.org/licenses/by-nc/2.0/'),self.row,self.url)['licensing_status'],STATUSES[2])
  self.assertEqual(classify(self.doc(lic='https://creativecommons.org/licenses/by/4.0/'),self.row,self.url)['licensing_status'],STATUSES[4])
 def test_license_allowlist(self):
  self.assertEqual(license_kind('https://creativecommons.org/publicdomain/zero/1.0/')[0],'allowed')
  self.assertEqual(license_kind('https://creativecommons.org.evil/licenses/by/2.0/')[0],'unknown')
 def test_no_dataset_wide_proof(self):self.assertEqual(classify('COCO says CC BY',self.row,self.url)['licensing_status'],STATUSES[1])
 def test_no_photo_get_ever(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Transport(root=Path(tmp)/'r')
   try:
    with self.assertRaises(ValueError):t.request(self.row,self.row['proposed_secure_image_url'],'GET')
    self.assertEqual(len(t.state['requests']),0)
   finally:t.close()
 def test_no_arbitrary_redirect_or_source(self):
  self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://evil'))
  self.assertFalse(allowed_page('http://www.flickr.com/photos/owner/'+self.photo,self.photo));self.assertFalse(allowed_page('https://www.flickr.com/photos/owner/999',self.photo))
 def test_observed_same_photo_resolver_redirect_only(self):
  self.assertTrue(allowed_page('https://www.flickr.com/photo.gne?rb=1&id='+self.photo,self.photo))
  self.assertFalse(allowed_page('https://www.flickr.com/photo.gne?rb=2&id='+self.photo,self.photo))
  self.assertFalse(allowed_page('https://www.flickr.com/photo.gne?rb=1&id=999',self.photo))
 def test_cap_before_request(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Transport(root=Path(tmp)/'r',cap=10)
   try:
    with self.assertRaises(ResourceLimit):t.request(self.row,self.url,'GET')
    self.assertEqual(t.state['actual_body_bytes'],0)
   finally:t.close()
 def test_final_proposal_retains_slots_and_current_identity_groups(self):
  if not (OUT/'proposed-image-acquisition-v1.json').exists():self.skipTest('Audit not yet finalized')
  plan=read(OUT/'proposed-image-acquisition-v1.json');original={r['image_id']:r for r in manifest()['images']}
  self.assertEqual(plan['image_count'],72);self.assertEqual(plan['summary']['train']['images'],60);self.assertEqual(plan['summary']['development']['images'],12)
  self.assertTrue(all(r['proposed_split']==original[r['image_id']]['proposed_split'] and r['identity_group']==r['known_source_group'] and not r['rights_verification_required_before_photo_approval'] and r['image_sha256'] is None for r in plan['images']))
  self.assertFalse(plan['photograph_acquisition_approved'])
 def test_current_and_historical_creator_overlaps_quarantined(self):
  if not (OUT/'identity-audit.json').exists():self.skipTest('Audit not finalized')
  from integrity import Union
  u=Union();u.join('photo:1','creator:a');u.join('photo:2','creator:a');self.assertEqual(u.find('photo:1'),u.find('photo:2'))
  records=read(OUT/'identity-audit.json')['records']
  self.assertEqual(sum('historical_known_source_or_creator_group' in r['identity_flags'] for r in records),3)
  self.assertEqual(sum('current_creator_or_known_source_group_crosses_original_partitions' in r['identity_flags'] for r in records),17)
  self.assertTrue(all(not r['eligible_for_image_acquisition_proposal'] for r in records if r['identity_flags']))
 def test_observed_rate_limit_retry_respects_cooldown_and_stop(self):
  if not (OUT/'rights-audit-v3.json').exists():self.skipTest('Audit not completed')
  qs=read(OUT/'network-ledger.json')['requests'];retry=[r for r in qs if r['attempt']>1]
  self.assertEqual(len(retry),1);last=retry[0];self.assertEqual(last['status'],429)
  first=next(r for r in qs if r['url']==last['url'] and r['method']==last['method'] and r['attempt']==1)
  self.assertGreaterEqual((dt.datetime.fromisoformat(last['checked_utc'])-dt.datetime.fromisoformat(first['checked_utc'])).total_seconds(),120)
  self.assertFalse(any(r['checked_utc']>last['checked_utc'] and not r['photo_object'] for r in qs))
 def test_all_actual_document_findings_replay(self):
  if not (OUT/'rights-audit-v3.json').exists():self.skipTest('Audit not completed')
  import gzip,io
  rows={r['image_id']:r for r in manifest()['images']};qs=read(OUT/'network-ledger.json')['requests'];n=0
  for r in read(OUT/'rights-audit-v3.json')['records']:
   gets=[e for e in r['evidence_requests'] if e['method']=='GET' and e.get('file')]
   if not gets:continue
   e=gets[-1];q=next(q for q in qs if q.get('file')==e['file']);data=(OUT/e['file']).read_bytes();self.assertEqual(sha(OUT/e['file']),e['sha256'])
   if q['headers'].get('Content-Encoding')=='gzip':
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(DECODED+1)
   self.assertLessEqual(len(data),DECODED);self.assertEqual(classify(data.decode('utf-8'),rows[r['image_id']],e['url'])['licensing_status'],r['licensing_status']);n+=1
  self.assertEqual(n,95)

if __name__=='__main__':unittest.main()
