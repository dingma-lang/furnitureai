"""No-network CPU tests for permission provenance,caps and provider stop behavior."""
import copy,gzip,ssl,unittest
from unittest.mock import patch
from common9 import *
from evidence9 import classify,allowed_page,license_kind
from transport9 import Transport,ProviderStopped,TransferLimit,allowed,TLSStream

class EvidenceTests(unittest.TestCase):
    def setUp(self):self.row=manifest()['records'][0];self.photo=self.row['source_photo_id'].split(':')[1];self.url='https://www.flickr.com/photos/synthetic-fixture/'+self.photo+'/'
    def page(self,lic='https://creativecommons.org/licenses/by/2.0/',photo=None,creator='Synthetic fixture only'):
        return '<script type="application/ld+json">'+json.dumps({'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/synthetic-fixture/'+(photo or self.photo)+'/',
             'name':'Synthetic metadata fixture;not an actual review','license':lic,'author':{'name':creator,'url':'https://www.flickr.com/photos/synthetic-fixture/'}})+'</script>'
    def test_same_photo_creator_specific_licence_required(self):
        r=classify(self.page(),self.row,self.url);self.assertEqual(r['rights_status'],STATUS_A)
        self.assertEqual(r['creator'],'Synthetic fixture only')
    def test_dataset_licence_is_not_photo_rights(self):self.assertEqual(classify('<p>COCOannotations CC BY4.0</p>',self.row,self.url)['rights_status'],STATUS_B)
    def test_wrong_photo_does_not_clear_candidate(self):self.assertEqual(classify(self.page(photo='999'),self.row,self.url)['rights_status'],STATUS_B)
    def test_missing_creator_unresolved(self):self.assertEqual(classify(self.page(creator=''),self.row,self.url)['rights_status'],STATUS_B)
    def test_restricted_or_changed_licence(self):
        self.assertEqual(classify(self.page(lic='https://creativecommons.org/licenses/by-nc/2.0/'),self.row,self.url)['rights_status'],STATUS_C)
        self.assertEqual(classify(self.page(lic='https://creativecommons.org/licenses/by/4.0/'),self.row,self.url)['rights_status'],STATUS_B)
    def test_noai_signal_is_not_silently_cleared(self):self.assertEqual(classify('<meta name="robots" content="noai">'+self.page(),self.row,self.url)['rights_status'],STATUS_B)
    def test_licence_lookalike_domain_rejected(self):self.assertEqual(license_kind('https://creativecommons.org.evil.invalid/licenses/by/2.0/')[0],'unknown')
    def test_photo_GET_and_alternate_ID_rejected(self):
        self.assertFalse(allowed(self.row,self.row['official_image_url'],'GET'));self.assertTrue(allowed(self.row,self.row['official_image_url'],'HEAD'))
        self.assertFalse(allowed(self.row,'https://www.flickr.com/photos/synthetic/999/','GET'))
        self.assertFalse(allowed(self.row,self.url.replace('https:','http:'),'GET'))
    def test_no_arbitrary_policy_host(self):self.assertFalse(allowed(None,'https://example.invalid/arbitrary','GET',policy=True))
    def test_exact64_native_annotation_and_no_partition_assignment(self):
        r=read(OUT/'local-candidate-audit.json')['records'];self.assertEqual(len(r),64)
        self.assertEqual(sum(len(x['native_annotation_ids']) for x in r),268)
        self.assertTrue(all(not x['Chair_absence_inferred'] and not x['final_dataset_partition_assigned'] for x in r))

class FakeSSL:
    def version(self):return 'SYNTHETIC-FIXTURE-NO-NETWORK'
    def getpeercert(self,binary_form=False):return b'Synthetic-test-certificate-never-used-for-real-TLS'
class FakeTLS:
    response=b'HTTP/1.1 429 Too Many Requests\r\nRetry-After: 600\r\nContent-Length: 0\r\n\r\n'
    def __init__(self,host,callback):self.callback=callback;self.ssl=FakeSSL();callback('received',len(self.response))
    def sendall(self,data):self.callback('sent',len(data))
    def makefile(self,*_):return io.BytesIO(self.response)
    def close(self):pass

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.folder=OUT/'cpu-fixtures'/uuid.uuid4().hex;self.folder.mkdir(parents=True)
        self.patches=[patch('common9.OUT',self.folder),patch('transport9.OUT',self.folder),patch('transport9.TLSStream',FakeTLS),patch('transport9.time.sleep',lambda *_:None)]
        for p in self.patches:p.start()
        self.t=Transport();self.row=manifest()['records'][0]
    def tearDown(self):
        self.t.journal.close()
        for p in reversed(self.patches):p.stop()
    def test_429_stops_provider_and_preserves_retry_after(self):
        self.t.begin_candidate(self.row);r=self.t.request(self.row,self.row['original_source_page_resolver'])
        self.assertEqual(r['status'],429);self.assertEqual(self.t.circuits['flickr.com']['Retry_After'],'600')
        with self.assertRaises(ProviderStopped):self.t.request(self.row,self.row['original_source_page_resolver'],'GET')
        self.assertEqual(len(self.t.requests),1)
    def test_single_candidate_attempt_and_no_retries(self):
        self.t.begin_candidate(self.row)
        with self.assertRaises(ValueError):self.t.begin_candidate(self.row)
    def test_cap_checked_before_connection(self):
        self.t.begin_candidate(self.row);self.t.used=CAP
        with self.assertRaises(TransferLimit):self.t.request(self.row,self.row['original_source_page_resolver'])
        self.assertEqual(self.t.requests,[])
    def test_tls_traffic_and_overhead_are_separate(self):
        self.t.begin_candidate(self.row);r=self.t.request(self.row,self.row['original_source_page_resolver'])
        self.assertGreater(r['sent_TLS_bytes'],0);self.assertGreater(r['received_TLS_bytes'],0);self.assertEqual(r['actual_body_bytes'],0)
        self.assertEqual(r['conservative_charge'],r['sent_TLS_bytes']+r['received_TLS_bytes']+65536)
    def test_forbidden_photo_body_fails_before_socket(self):
        self.t.begin_candidate(self.row)
        with self.assertRaises(PermissionError):self.t.request(self.row,self.row['official_image_url'],'GET')
        self.assertEqual(len(self.t.requests),0)
    def test_saved_circuit_survives_same_audit_resume(self):
        self.t.begin_candidate(self.row);self.t.request(self.row,self.row['original_source_page_resolver']);self.t.finish()
        r=Transport(resume=True)
        try:
            self.assertIn('flickr.com',r.circuits)
            with self.assertRaises(ProviderStopped):r.begin_candidate(manifest()['records'][1])
        finally:r.journal.close()
    def test_default_ssl_context_verifies_chain_and_hostname(self):
        c=ssl.create_default_context();self.assertTrue(c.check_hostname);self.assertEqual(c.verify_mode,ssl.CERT_REQUIRED)
    def test_gzip_decode_bounded_and_checksum_verified(self):
        p=self.folder/'synthetic-evidence.gz';p.write_bytes(gzip.compress(b'x'*(DECODED+100)))
        rec=dict(file=p.name,sha256=sha(p),headers={'Content-Encoding':'gzip'})
        with self.assertRaises(ValueError):self.t.decoded(rec)

if __name__=='__main__':
    # Any accidental socket call would fail the CPU test suite immediately.
    with patch('socket.create_connection',side_effect=AssertionError('Unit tests must not access the network')):unittest.main()
