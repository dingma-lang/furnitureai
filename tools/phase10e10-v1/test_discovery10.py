import unittest
from net10 import allowed,MAX_TLS,ALLOWANCE
from common10 import *
class SafetyTests(unittest.TestCase):
    def test_photo_hosts_rejected(self):
        for u in ('https://upload.wikimedia.org/a.jpg','https://live.staticflickr.com/1/x.jpg','https://commons.wikimedia.org/a.jpg'):
            self.assertFalse(allowed(u))
    def test_no_flickr_api_without_permission(self):self.assertFalse(allowed('https://api.flickr.com/services/rest/?method=flickr.photos.getInfo'))
    def test_no_credentials(self):self.assertFalse(allowed('https://www.mediawiki.org/wiki/API:Main_page?api_key=secret'))
    def test_tls_only(self):self.assertFalse(allowed('http://commons.wikimedia.org/w/api.php'))
    def test_no_cross_host_or_userinfo(self):
        self.assertFalse(allowed('https://commons.wikimedia.org.evil.invalid/w/api.php'));self.assertFalse(allowed('https://secret@commons.wikimedia.org/w/api.php'))
    def test_commons_robot_denied_path_blocked(self):self.assertFalse(allowed('https://commons.wikimedia.org/w/api.php?action=query&format=json'))
    def test_budget_reservation(self):self.assertLess(MAX_TLS+ALLOWANCE,CAP)
    def test_E9_preserved_status(self):
        s=read(ROOT/'reports/phase10e9-v1/final-summary.json');self.assertEqual((s['rights_supported'],s['unresolved'],s['unsuitable']),(0,64,0))

class RightsTests(unittest.TestCase):
    def fixture(self):
        return dict(id='synthetic-only',source=dict(id='synthetic-work',title='Synthetic bedroom photograph; not a real review',genres=[dict(label='Photographs')],subjects=[dict(label='Bedrooms')],contributors=[]),locations=[dict(url='https://iiif.wellcomecollection.org/image/SYNTHETIC/info.json',credit='Synthetic institution only',license=dict(url='https://creativecommons.org/licenses/by/4.0/'),locationType=dict(id='iiif-image'),accessConditions=[dict(status=dict(id='open'))])])
    def adapt(self,item):
        from wellcome10 import adapt
        return adapt(item,dict(url='https://synthetic.invalid/metadata',sha256='synthetic',file='synthetic-not-source-evidence',checked_utc='SYNTHETIC'), 'fixture')
    def test_file_specific_licence_and_credit(self):
        r=self.adapt(self.fixture());self.assertEqual(r['rights_status'],'A_rights_supported');self.assertFalse(r['creator_identity_verified']);self.assertIsNone(r['photographer'])
    def test_no_invented_negative_labels(self):
        r=self.adapt(self.fixture());self.assertEqual(r['verified_negative_regions'],0);self.assertFalse(r['Chair_absence_verified']);self.assertIsNone(r['final_partition'])
    def test_missing_license_not_cleared(self):
        item=self.fixture();item['locations'][0].pop('license');self.assertEqual(self.adapt(item)['rights_status'],'B_unresolved')
    def test_NC_incompatible(self):
        item=self.fixture();item['locations'][0]['license']['url']='https://creativecommons.org/licenses/by-nc/4.0/';self.assertEqual(self.adapt(item)['rights_status'],'C_unsuitable')
    def test_sharealike_not_automatically_prohibited(self):
        item=self.fixture();item['locations'][0]['license']['url']='https://creativecommons.org/licenses/by-sa/4.0/';self.assertEqual(self.adapt(item)['rights_status'],'B_unresolved')
    def test_access_advisory_protected(self):
        item=self.fixture();item['locations'][0]['accessConditions'][0]['terms']='Additional permission required';self.assertEqual(self.adapt(item)['rights_status'],'B_unresolved')
    def test_conflicting_licence_unresolved(self):
        import copy
        item=self.fixture();other=copy.deepcopy(item['locations'][0]);other['license']['url']='https://creativecommons.org/licenses/by-nc/4.0/';item['locations'].append(other)
        self.assertEqual(self.adapt(item)['rights_status'],'B_unresolved')
    def test_unknown_creators_not_distinct_groups(self):
        item=self.fixture();a=self.adapt(item);item['id']='different';b=self.adapt(item);self.assertEqual(a['source_group_id'],b['source_group_id'])
    def test_metadata_hints_not_chair_absence_or_stool_mapping(self):
        from decisions10 import category_hints
        self.assertIn('Chair',category_hints('Bar stools beside a desk'));self.assertNotIn('Chair',category_hints('Ottoman, footstool, step stool'))
    def test_licence_domain_lookalike(self):
        from decisions10 import cc_kind
        self.assertEqual(cc_kind('https://creativecommons.org.evil.invalid/licenses/by/4.0/'),'unknown')
    def test_duplicate_and_maximum_guard(self):
        from decisions10 import assert_unique
        with self.assertRaises(ValueError):assert_unique([dict(candidate_id='same')]*2)
        with self.assertRaises(ValueError):assert_unique([dict(candidate_id=str(i)) for i in range(201)])

class NetworkTests(unittest.TestCase):
    def setUp(self):
        import io,uuid
        from unittest.mock import patch
        import common10,net10
        self.net=net10;self.folder=OUT/'cpu-fixtures'/uuid.uuid4().hex;self.folder.mkdir(parents=True)
        class FakeSSL:
            def version(self):return 'SYNTHETIC-NO-NETWORK'
            def getpeercert(self,binary_form=False):return b'synthetic-test-only'
        class FakeTLS:
            response=b'HTTP/1.1 429 Too Many Requests\r\nRetry-After: 900\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n'
            def __init__(s,host,cb):s.cb=cb;s.ssl=FakeSSL();cb('received',len(s.response))
            def sendall(s,data):s.cb('sent',len(data))
            def makefile(s,*_):return io.BytesIO(s.response)
            def close(s):pass
        self.fake=FakeTLS;self.patchers=[patch('common10.OUT',self.folder),patch('net10.OUT',self.folder),patch('net10.TLSStream',FakeTLS),patch('net10.time.sleep',lambda *_:None),patch('socket.create_connection',side_effect=AssertionError('No real network in tests'))]
        for p in self.patchers:p.start()
        self.c=net10.Client()
    def tearDown(self):
        for p in reversed(self.patchers):p.stop()
    def test_429_durable_and_no_retry(self):
        r,t=self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=synthetic')
        self.assertEqual(r['status'],429);self.assertIsNone(t);self.assertEqual(r['Retry_After'],'900')
        c=self.net.Client()
        with self.assertRaises(PermissionError):c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=different')
    def test_cap_before_request(self):
        self.c.used=CAP
        with self.assertRaises(RuntimeError):self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works')
        self.assertFalse(self.c.rows)
    def test_photo_forbidden_before_socket(self):
        with self.assertRaises(PermissionError):self.c.fetch('https://iiif.wellcomecollection.org/image/SYNTHETIC/full/1600,/0/default.jpg')
        self.assertFalse(self.c.rows)
    def test_body_size_limit_before_read(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2000000\r\n\r\n'
        with self.assertRaises(RuntimeError):self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=oversize')
        self.assertEqual(self.c.rows[0]['body_bytes'],0)
    def test_unexpected_image_mime_refused(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 10\r\n\r\n'
        with self.assertRaises(ValueError):self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=badmime')
        self.assertEqual(self.c.rows[0]['body_bytes'],0)
    def test_interrupted_intent_cannot_auto_retry(self):
        import common10
        common10.emit(self.folder/'intents/001.json',dict(number=1,url='synthetic',reserved=MAX_TLS+ALLOWANCE))
        with self.assertRaises(PermissionError):self.net.Client()
    def test_no_redirect_follow(self):
        self.fake.response=b'HTTP/1.1 302 Found\r\nLocation: https://example.invalid/forbidden.jpg\r\nContent-Length: 0\r\n\r\n'
        r,t=self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=redirect')
        self.assertEqual(r['status'],302);self.assertIsNone(t);self.assertEqual(len(self.c.rows),1)
    def test_TLS_measurement_and_allowance_separate(self):
        r,t=self.c.fetch('https://api.wellcomecollection.org/catalogue/v2/works?query=meter')
        self.assertEqual(r['budget_charge'],r['TLS_sent']+r['TLS_received']+ALLOWANCE)
    def test_closed_audit_cannot_resume_network(self):
        import common10
        common10.emit(self.folder/'discovery-closed.json',dict(status='SYNTHETIC'))
        with self.assertRaises(PermissionError):self.net.Client()

class CompletionTests(unittest.TestCase):
    def test_final_rights_counts_and_scope(self):
        s=read(REPORT/'discovery-summary.json');self.assertEqual(s['new_unique_image_metadata_records'],78)
        self.assertEqual(s['total_new_metadata_status_counts'],dict(A=76,B=22,C=0));self.assertLessEqual(s['conservative_new_record_count'],200)
    def test_acquisition_purpose_separation(self):
        t=read(REPORT/'potential-training-acquisition.json');e=read(REPORT/'potential-independent-evaluation-acquisition.json')
        self.assertEqual(len(t['records']),7);self.assertFalse(e['records'])
        self.assertTrue(all(r['rights_status']=='A_rights_supported' and r['final_partition'] is None and not r['photo_download_approved'] for r in t['records']))
    def test_only_supported_in_supported_manifest(self):
        rows=read(REPORT/'rights-supported-image-inventory.json')['records'];self.assertEqual(len(rows),76)
        self.assertTrue(all(r['rights_status']=='A_rights_supported' for r in rows))
    def test_all_unresolved_retained(self):
        rows=read(REPORT/'unresolved-rights-candidates.json')['records'];self.assertEqual(len(rows),22);self.assertTrue(all(r['rights_status']=='B_unresolved' for r in rows))
    def test_seven_sources_info_and_evidence_hashes(self):
        for r in read(REPORT/'potential-training-acquisition.json')['records']:
            self.assertEqual(sha(ROOT/r['evidence']['response_path']),r['evidence']['sha256'])
            self.assertGreater(r['width'],0);self.assertGreater(r['height'],0)
            self.assertTrue(r['exact_download_URL'].endswith('/full/!1600,1600/0/default.jpg'))
    def test_no_empty_consent_approval(self):
        from consent10 import validate_grant
        r=validate_grant({});self.assertEqual(r['status'],'unresolved');self.assertFalse(r['automatically_approved'])
    def test_grant_still_requires_evidence_verification(self):
        from consent10 import validate_grant,REQUIRED_SCOPES
        x=dict(photograph_ids=['SYNTHETIC'],creator_or_rightsholder='Synthetic fixture',source_group_id='SYNTHETIC',grant_timestamp='SYNTHETIC',affirmative_signature_or_evidence='SYNTHETIC-NOT-A-REAL-GRANT',rights_holder_authority_confirmed=True,scopes={k:True for k in REQUIRED_SCOPES},other_people_or_third_party_rights_resolved=True,evaluation_or_training_purpose='training')
        r=validate_grant(x);self.assertEqual(r['status'],'grant_ready_for_independent_evidence_verification');self.assertFalse(r['photo_download_authorized'])
    def test_source_network_within_cap_and_closed(self):
        x=read(OUT/'discovery-closed.json');self.assertLess(x['network']['conservative_charge'],CAP);self.assertEqual(x['photo_bodies'],0)
    def test_authoritative_manifest_hash(self):
        p=read(REPORT/'next-photo-approval-request.json');self.assertEqual(sha(ROOT/p['manifest']),p['manifest_SHA256'])
    def test_no_download_URL_can_be_fetched_by_current_client(self):
        for r in read(REPORT/'potential-training-acquisition.json')['records']:self.assertFalse(allowed(r['exact_download_URL']))

if __name__=='__main__':unittest.main()
