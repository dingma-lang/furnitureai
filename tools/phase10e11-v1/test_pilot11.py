import io,unittest,uuid
from unittest.mock import patch
from PIL import Image
from common11 import *
from acquire11 import allowed,Client
from screening11 import inspect_bytes,distance
class ScopeTests(unittest.TestCase):
    def test_exact_seven_supported(self):self.assertEqual(tuple(r['source_image_id'] for r in rows()),IDS)
    def test_alternate_image_or_size_rejected(self):
        u=rows()[0]['exact_download_URL'];self.assertTrue(allowed(u,'image'));self.assertFalse(allowed(u.replace('1600','800'),'image'));self.assertFalse(allowed(u.replace('/L0028138/','/UNKNOWN/'),'image'))
    def test_no_unapproved_metadata(self):self.assertFalse(allowed('https://api.wellcomecollection.org/catalogue/v2/images/unknown','metadata'))
    def test_changed_licence_blocks_photo(self):
        old=rows()[0];x=dict(id=old['source_image_id'],source=dict(id=old['source_work_id']),locations=[dict(url=old['image_metadata_URL'],credit='Wellcome Collection',license=dict(url='https://creativecommons.org/licenses/by-nc/4.0/'))])
        with self.assertRaises(ValueError):rights_validate(x,old)
    def test_no_whole_image_negative(self):self.assertTrue(all(r['verified_negative_regions']==0 for r in rows()))
    def test_hash_distance(self):self.assertEqual(distance('0000000000000000','ffffffffffffffff'),64)
    def test_jpeg_integrity_and_size(self):
        b=io.BytesIO();Image.new('RGB',(100,50),'gray').save(b,format='JPEG');r=inspect_bytes(b.getvalue(),dict(width=200,height=100));self.assertEqual(r['display_dimensions'],[100,50]);self.assertIn('low_contrast_or_blank_heuristic',r['flags'])
    def test_corrupt_image_fails(self):
        with self.assertRaises(Exception):inspect_bytes(b'not-a-JPEG',dict(width=100,height=100))
    def test_body_limit(self):
        with self.assertRaises(ValueError):inspect_bytes(b'x'*(IMAGE_CAP+1),dict(width=100,height=100))
class NetworkTests(unittest.TestCase):
    def setUp(self):
        import acquire11
        self.folder=REPORT/'.cpu-fixtures'/uuid.uuid4().hex;self.folder.mkdir(parents=True)
        class FakeSSL:
            def version(self):return 'SYNTHETIC-NO-NETWORK'
            def getpeercert(self,binary_form=False):return b'synthetic'
        class FakeTLS:
            response=b'HTTP/1.1 429 Too Many Requests\r\nRetry-After: 600\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n'
            def __init__(s,host,cb):s.cb=cb;s.ssl=FakeSSL();cb('received',len(s.response))
            def sendall(s,b):s.cb('sent',len(b))
            def makefile(s,*_):return io.BytesIO(s.response)
            def close(s):pass
        self.fake=FakeTLS
        self.patches=[patch('common11.OUT',self.folder),patch('acquire11.OUT',self.folder),patch('acquire11.TLSStream',FakeTLS),patch('acquire11.time.sleep',lambda *_:None),patch('socket.create_connection',side_effect=AssertionError('No actual network in CPU tests'))]
        for p in self.patches:p.start()
        self.c=Client()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
    def test_429_stop_and_durable(self):
        with self.assertRaises(RuntimeError):self.c.fetch(metadata_url(IDS[0]),'metadata')
        other=Client()
        with self.assertRaises(PermissionError):other.fetch(metadata_url(IDS[1]),'metadata')
        self.assertEqual(len(other.receipts),1)
    def test_transfer_stop_before_connection(self):
        self.c.used=CAP
        with self.assertRaises(RuntimeError):self.c.fetch(metadata_url(IDS[0]),'metadata')
        self.assertFalse(self.c.receipts)
    def test_oversize_header_no_body(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 512001\r\n\r\n'
        with self.assertRaises(RuntimeError):self.c.fetch(rows()[0]['exact_download_URL'],'image')
        self.assertEqual(self.c.receipts[0]['body_bytes'],0)
    def test_redirect_not_followed_and_accounted(self):
        self.fake.response=b'HTTP/1.1 302 Found\r\nLocation: https://example.invalid/other.jpg\r\nContent-Length: 0\r\n\r\n'
        with self.assertRaises(RuntimeError):self.c.fetch(rows()[0]['exact_download_URL'],'image')
        self.assertGreater(self.c.used,0);self.assertEqual(len(self.c.receipts),1)
    def test_no_repeating_successful_transfer(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}'
        self.c.fetch(metadata_url(IDS[0]),'metadata')
        with self.assertRaises(PermissionError):Client().fetch(metadata_url(IDS[0]),'metadata')
    def test_interrupted_transaction_preserved(self):
        emit(self.folder/'intents/001.json',dict(number=1))
        with self.assertRaises(PermissionError):Client()
    def test_closed_pilot_no_requests(self):
        emit(self.folder/'acquisition-closed.json',dict(synthetic=True))
        with self.assertRaises(PermissionError):Client()
    def test_image_request_MIME_rejected(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 0\r\n\r\n'
        with self.assertRaises(ValueError):self.c.fetch(rows()[0]['exact_download_URL'],'image')

class IntakeTests(unittest.TestCase):
    def fixture(self,purpose='training'):
        from intake11 import SCOPES
        return dict(photo_ids=['SYNTHETIC-NOT-A-PHOTOGRAPH'],creator_id='SYNTHETIC-CREATOR',source_group_id='SYNTHETIC-SOURCE',room_group_ids=['SYNTHETIC-ROOM'],permission_evidence='SYNTHETIC-TEST-NOT-A-GRANT',permission_date='SYNTHETIC',explicit_confirmation=True,privacy_and_third_party_rights_resolved=True,scopes={k:True for k in SCOPES},purpose=purpose)
    def test_no_empty_consent(self):
        from intake11 import validate
        self.assertEqual(validate({})['status'],'unresolved')
    def test_complete_metadata_not_approval(self):
        from intake11 import validate
        v=validate(self.fixture());self.assertEqual(v['status'],'metadata_complete_pending_evidence_and_identity_verification');self.assertFalse(v['human_reviewed']);self.assertFalse(v['annotations_active'])
    def test_room_or_creator_cannot_cross_purposes(self):
        from intake11 import validate
        a=self.fixture();b=self.fixture('independent_evaluation');b['photo_ids']=['DIFFERENT-SYNTHETIC'];self.assertEqual(validate(b,[a])['status'],'unresolved')
    def test_duplicate_photo(self):
        from intake11 import validate
        a=self.fixture();self.assertIn('Duplicate photograph ID',validate(a,[a])['errors'])
    def test_missing_category_not_negative(self):
        from intake11 import annotation_complete
        self.assertFalse(annotation_complete({'class_completeness':{'Chair':'complete_positive_and_negative_review'}}))
    def test_seven_categories_and_independent_evidence_required(self):
        from intake11 import annotation_complete,CLASSES
        self.assertTrue(annotation_complete(dict(class_completeness={c:'complete_positive_and_negative_review' for c in CLASSES},independent_reviewer_evidence='SYNTHETIC-TEST-EVIDENCE')))
class CompletionTests(unittest.TestCase):
    def test_actual_sources_and_integrity(self):
        x=read(OUT/'acquisition-closed.json');self.assertIsNone(x['stop']);self.assertEqual({r['image_id'] for r in x['records']},set(IDS))
        for r in x['records']:
            self.assertEqual(sha(ROOT/r['file']),r['sha256']);self.assertLessEqual(r['byte_size'],IMAGE_CAP)
            self.assertFalse(r['training_active']);self.assertIsNone(r['final_partition'])
    def test_fresh_evidence_hashes(self):
        for id in IDS:
            r=read(OUT/'rights'/(id+'.json'))
            self.assertEqual(sha(OUT/'evidence'/(id+'-catalogue.json')),r['catalogue_evidence']['SHA256'])
            self.assertEqual(sha(OUT/'evidence'/(id+'-IIIF.json')),r['IIIF_evidence']['SHA256'])
            self.assertEqual(r['rights']['credit'],['Wellcome Collection']);self.assertIsNone(r['rights']['creator'])
    def test_coverage_not_active_labels(self):
        v=read(OUT/'visual-review.json');self.assertEqual(v['summary']['suitability_counts'],dict(A_modern=0,B_historical_or_atypical=7,C_unsuitable=0))
        self.assertTrue(all(not r['annotation_active'] and not r['human_reviewed'] and r['verified_negative_regions']==0 for r in v['records']))
    def test_network_limits_and_attempts(self):
        n=read(OUT/'acquisition-closed.json')['network'];self.assertLess(n['conservative_charge'],CAP);self.assertEqual(n['actual_max_attempts'],1);self.assertEqual(n['retries'],0)
        receipts=[read(p) for p in (OUT/'requests').glob('*.json')];self.assertEqual(sum(r['kind']=='image' for r in receipts),7)
    def test_read_only_gallery_scope(self):
        from serve11 import ALLOWED
        self.assertNotIn('/../authorization.json',ALLOWED);self.assertEqual(len([k for k in ALLOWED if k.endswith('.jpg')]),7)
        text=(OUT/'gallery.html').read_text();self.assertEqual(text.count('<img '),7);self.assertNotIn('data:image/',text)
    def test_no_old_images_or_frozen_in_gallery(self):
        text=(OUT/'gallery.html').read_text();self.assertNotIn('data/open-images/',text)
        d=read(OUT/'duplicate-screening.json');self.assertEqual(d['reference_count'],78);self.assertEqual(d['frozen_or_reserved_images_read'],0)
    def test_intake_starts_unconfirmed_and_no_upload(self):
        t=(OUT/'creator-intake.html').read_text();self.assertNotIn('type="file"',t);self.assertNotIn(' checked',t);self.assertIn('submitted_unverified',t);self.assertIn('human_reviewed:false',t)

if __name__=='__main__':unittest.main()
