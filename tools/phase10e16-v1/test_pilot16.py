"""Bounded acquisition tests: synthetic transport/images, no source requests."""
import copy
import hashlib
import io
import tempfile
import http.client
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image,UnidentifiedImageError
from common16 import IDS,ROOT,rows,read
from acquire16 import Client,allowed,metadata_url,rights_validate,CAP
from screen16 import inspect_image


class ScopeTests(unittest.TestCase):
    def test_exact_ids_and_sources(self):
        self.assertEqual(tuple(r['source_id'] for r in rows()),IDS)
        url=rows()[0]['source_object_url']
        self.assertTrue(allowed(url,'image','GET'))
        self.assertTrue(allowed(url,'image','HEAD'))
        self.assertFalse(allowed(url.replace('Workspace','Other'),'image','GET'))
        self.assertFalse(allowed(url.replace('https:','http:'),'image','GET'))

    def test_rights_changes_stop(self):
        doc=read(ROOT/'reports/phase10e15-v1/commons-metadata-access.json')['document']
        self.assertEqual(len(rights_validate(doc,rows())),3)
        for field,value in [('size',1),('sha1','bad'),('url','https://example.invalid/photo')]:
            d=copy.deepcopy(doc);d['query']['pages']['150832261']['imageinfo'][0][field]=value
            with self.assertRaises(ValueError):rights_validate(d,rows())
        d=copy.deepcopy(doc);d['query']['pages']['150832261']['imageinfo'][0]['extmetadata']['LicenseShortName']['value']='CC BY-NC 4.0'
        with self.assertRaisesRegex(ValueError,'licence'):rights_validate(d,rows())

    def test_image_checks_source_hash_and_dimensions(self):
        b=io.BytesIO();Image.new('RGB',(120,80),'gray').save(b,format='JPEG');body=b.getvalue()
        r=dict(expected_bytes=len(body),source_sha1=hashlib.sha1(body).hexdigest(),width=120,height=80)
        self.assertEqual(inspect_image(body,r)['display_dimensions'],[120,80])
        with self.assertRaisesRegex(ValueError,'SHA1'):inspect_image(body,dict(r,source_sha1='f'*40))
        with self.assertRaisesRegex(ValueError,'dimensions'):inspect_image(body,dict(r,width=100))
        with self.assertRaises(ValueError):inspect_image(body+b'bad',r)

    def test_corrupt_jpeg_not_committed(self):
        b=b'invalid image';r=dict(expected_bytes=len(b),source_sha1=hashlib.sha1(b).hexdigest(),width=120,height=80)
        with self.assertRaises(UnidentifiedImageError):inspect_image(b,r)

    def test_storage_monitor_tolerates_only_vanished_browser_temporaries(self):
        import common16
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);p=base/'transient';p.write_bytes(b'synthetic temp')
            original=Path.lstat
            def vanished(path,*a,**k):
                if path==p:raise FileNotFoundError('Synthetic enumeration/stat race')
                return original(path,*a,**k)
            with patch('common16.OUT',base),patch('common16.REPORT',base/'r'),patch('common16.HERE',base/'h'),patch.object(Path,'lstat',vanished):
                self.assertGreaterEqual(common16.allocated(),0)
            def denied(path,*a,**k):
                if path==p:raise PermissionError('Synthetic inaccessible file')
                return original(path,*a,**k)
            with patch('common16.OUT',base),patch('common16.REPORT',base/'r'),patch('common16.HERE',base/'h'),patch.object(Path,'lstat',denied):
                with self.assertRaises(PermissionError):common16.allocated()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='E16-synthetic-transport-');self.root=Path(self.temp.name)
        class FakeSSL:
            def version(self):return 'synthetic-only'
            def getpeercert(self,binary_form=False):return b'synthetic-cert'
        class FakeTLS:
            response=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}'
            def __init__(s,host,callback):s.ssl=FakeSSL();s.callback=callback;callback('received',len(s.response))
            def sendall(s,b):s.callback('sent',len(b))
            def makefile(s,*args):return io.BytesIO(s.response)
            def close(s):pass
        self.fake=FakeTLS
        self.patches=[patch('common16.OUT',self.root),patch('acquire16.OUT',self.root),
            patch('common16.REPORT',self.root/'report'),patch('acquire16.TLSStream',FakeTLS),
            patch('acquire16.time.sleep',lambda *_:None),patch('socket.create_connection',side_effect=AssertionError('No source requests in tests'))]
        for p in self.patches:p.start()
        self.client=Client()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def test_success_persisted_and_not_repeated(self):
        r,b=self.client.fetch(metadata_url(),'metadata');self.assertEqual(b,b'{}')
        self.assertTrue(r['TLS_verified']);self.assertGreater(self.client.used,len(b))
        with self.assertRaises(PermissionError):Client().fetch(metadata_url(),'metadata')

    def test_redirects_accounted_never_followed(self):
        self.fake.response=b'HTTP/1.1 302 Found\r\nLocation: https://example.invalid/image.jpg\r\nContent-Length: 0\r\n\r\n'
        with self.assertRaisesRegex(RuntimeError,'302'):self.client.fetch(metadata_url(),'metadata')
        self.assertGreater(Client().used,0)
        self.assertEqual(len(self.client.receipts),1)

    def test_provider_429_stops_without_retry(self):
        self.fake.response=b'HTTP/1.1 429 Too Many Requests\r\nRetry-After: 600\r\nContent-Length: 0\r\n\r\n'
        with self.assertRaises(RuntimeError):self.client.fetch(metadata_url(),'metadata')
        with self.assertRaises(PermissionError):Client().fetch(metadata_url(),'metadata')

    def test_byte_budget_stops_before_socket(self):
        self.client.used=CAP
        with self.assertRaisesRegex(RuntimeError,'reservation'):self.client.fetch(metadata_url(),'metadata')
        self.assertEqual(self.client.receipts,[])

    def test_changed_head_size_refuses_image_body(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 1\r\n\r\n'
        with self.assertRaisesRegex(ValueError,'length changed'):self.client.fetch(rows()[0]['source_object_url'],'image','HEAD')
        self.assertEqual(self.client.receipts[0]['body_bytes'],0)

    def test_unexpected_MIME_refuses_body(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 2\r\n\r\n{}'
        with self.assertRaisesRegex(ValueError,'MIME'):self.client.fetch(metadata_url(),'metadata')

    def test_closed_or_interrupted_pilot_refuses_network(self):
        (self.root/'acquisition-closed.json').write_text('{}')
        with self.assertRaises(PermissionError):Client()
        (self.root/'acquisition-closed.json').unlink()
        (self.root/'intents').mkdir();(self.root/'intents/001.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'Interrupted'):Client()

    def test_truncated_metadata_body_fails(self):
        self.fake.response=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 9\r\n\r\n{}'
        with self.assertRaises(ValueError):self.client.fetch(metadata_url(),'metadata')


class CompletedPilotTests(unittest.TestCase):
    def test_actual_intake_imports_are_content_verified_but_remain_inactive(self):
        from common16 import OUT
        closed=read(OUT/'acquisition-closed.json')
        self.assertEqual(closed['downloaded_count'],3);self.assertIsNone(closed['stop'])
        for r in closed['records']:
            ident=r['source_id'].split(':')[1]
            intake=read(OUT/'intake'/(ident+'.json'));result=read(OUT/'intake'/(ident+'-validation.json'))
            self.assertEqual(intake['photo']['image_sha256'],r['sha256'])
            self.assertTrue(result['image_content_verified'])
            self.assertFalse(result['annotations_active']);self.assertFalse(result['eligible_for_training'])
            self.assertEqual(intake['annotations'],[]);self.assertFalse(intake['human_reviewed'])
            self.assertTrue(all(c['status']=='unknown' for c in intake['class_completeness'].values()))

    def test_gallery_contains_only_local_exact_images_and_honest_labels(self):
        from common16 import OUT
        page=(OUT/'gallery.html').read_text()
        self.assertEqual(page.count('<img '),3)
        self.assertNotIn('<img src="http',page)
        self.assertIn('not ground truth',page)
        for ident in IDS:self.assertIn('/images/'+ident.split(':')[1]+'.jpg',page)

    def test_gallery_and_intake_route_privacy(self):
        from serve16 import Handler,routes
        from http.server import ThreadingHTTPServer
        allowed=routes()
        self.assertEqual(sum(k.startswith('/images/') for k in allowed),3)
        with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            try:
                for method,path,expected in [('GET','/gallery.html',200),('GET','/intake.html',200),('GET','/../.git/config',404),('GET','/smoke.html',404),('GET','/smoke-data.json',404),('POST','/',405)]:
                    c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request(method,path);r=c.getresponse();r.read()
                    self.assertEqual(r.status,expected);c.close()
            finally:server.shutdown();t.join(2)


if __name__=='__main__':unittest.main()
