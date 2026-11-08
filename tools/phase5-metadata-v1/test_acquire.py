import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import acquire as a

class Response:
    def __init__(self,body,offset=0,fail=False):
        self.body=body;self.offset=offset;self.pos=0;self.fail=fail;self.status=206 if offset else 200
        self.headers={'Content-Length':str(len(body)-offset),'ETag':'"fixed"'}
        if offset:self.headers['Content-Range']=f'bytes {offset}-{len(body)-1}/{len(body)}'
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):
        if self.fail and self.pos:raise TimeoutError('interrupted test transfer')
        chunk=self.body[self.offset+self.pos:self.offset+self.pos+(2 if self.fail else n)];self.pos+=len(chunk);return chunk

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dest=Path(self.temp.name)/'metadata'
        self.body=b'abcdef'
        self.source=dict(local_name='fixture.csv',url='https://storage.googleapis.com/openimages/fixture.csv',bytes=6,etag='"fixed"',source_md5_hex=hashlib.md5(self.body).hexdigest())
    def run_fixture(self,opener,cap=100,resume=False):
        with patch.object(a,'DEST',self.dest),patch.object(a,'CAP',cap),patch.object(a,'approved',return_value=[]),patch.object(a,'preflight',return_value=[self.source]),patch.object(a.urllib.request,'build_opener',return_value=opener),contextlib.redirect_stdout(io.StringIO()):a.run(resume)
    def opener(self,responses):
        class Opener:
            def __init__(self):self.calls=[]
            def open(self,request,timeout):
                self.calls.append(request)
                return responses.pop(0)
        return Opener()
    def test_preflight_repeated_hash_headers_and_source_changes(self):
        from email.message import Message
        import base64
        response=Response(b'')
        response.headers=Message()
        md5=hashlib.md5(self.body).digest()
        response.headers['Content-Length']='6'
        response.headers['ETag']='"'+md5.hex()+'"'
        response.headers['Last-Modified']='unchanged'
        response.headers['x-goog-hash']='crc32c=AAAAAA=='
        response.headers['x-goog-hash']='md5='+base64.b64encode(md5).decode()
        spec=dict(url=self.source['url'],bytes=6,etag=response.headers['ETag'],last_modified='unchanged')
        op=self.opener([response]);result=a.preflight(op,[spec])
        self.assertEqual(result[0]['source_md5_hex'],md5.hex())
        self.assertEqual(op.calls[0].get_method(),'HEAD')
        with self.assertRaisesRegex(ValueError,'changed'):a.preflight(self.opener([response]),[dict(spec,bytes=7)])

    def test_success_hash_and_completed_resume(self):
        op=self.opener([Response(self.body)]);self.run_fixture(op)
        self.assertEqual((self.dest/'fixture.csv').read_bytes(),self.body)
        state=json.loads((self.dest/'transfer-ledger.json').read_text());self.assertEqual(state['actual_body_bytes'],6)
        self.run_fixture(op,resume=True);self.assertEqual(len(op.calls),1)
        with self.assertRaises(FileExistsError):self.run_fixture(op)
    def test_range_resume_counts_unknown_conservatively(self):
        op=self.opener([Response(self.body,fail=True),Response(self.body,offset=2)])
        self.run_fixture(op)
        self.assertEqual(op.calls[1].get_header('Range'),'bytes=2-')
        state=json.loads((self.dest/'transfer-ledger.json').read_text())
        self.assertEqual(state['actual_body_bytes'],6);self.assertEqual(state['charged_bytes'],10)
        self.assertEqual(state['conservative_unknown_bytes'],4)
    def test_cap_stops_before_get(self):
        op=self.opener([])
        with self.assertRaisesRegex(RuntimeError,'transfer budget'):self.run_fixture(op,cap=5)
        self.assertFalse(op.calls)
    def test_integrity_failure_preserves_partial(self):
        op=self.opener([Response(b'badbad')])
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):self.run_fixture(op)
        self.assertFalse((self.dest/'fixture.csv').exists());self.assertTrue((self.dest/'fixture.csv.part').exists())
    def test_changed_file_not_overwritten(self):
        op=self.opener([Response(self.body)]);self.run_fixture(op)
        (self.dest/'fixture.csv').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):self.run_fixture(op,resume=True)
        self.assertEqual((self.dest/'fixture.csv').read_bytes(),b'changed')

if __name__=='__main__':unittest.main()
