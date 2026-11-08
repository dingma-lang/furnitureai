import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from common import *
from acquire import validate_headers,NoRedirect,opener
from extract import inspect_archive

class Tests(unittest.TestCase):
 def test_approved_contract(self):
  self.assertEqual(approved()['expected_transfer_bytes'],SIZE)
 def test_source_headers_pinned(self):
  h={'Content-Length':str(SIZE),'ETag':ETAG,'Content-Type':'application/zip'}
  validate_headers(200,h,0)
  for key,value in [('ETag','changed'),('Content-Length','1'),('Content-Type','image/jpeg'),('Content-Encoding','gzip')]:
   with self.assertRaises(ValueError):validate_headers(200,dict(h,**{key:value}),0)
 def test_safe_range_recovery_headers(self):
  h={'Content-Length':str(SIZE-12),'ETag':ETAG,'Content-Type':'application/zip','Content-Range':f'bytes 12-{SIZE-1}/{SIZE}'}
  validate_headers(206,h,12)
  with self.assertRaises(ValueError):validate_headers(200,h,12)
  with self.assertRaises(ValueError):validate_headers(206,dict(h,**{'Content-Range':'bytes 0-1/2'}),12)
 def test_no_redirect_or_tls_bypass(self):
  import ssl,urllib.request
  self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://other'))
  handler=next(h for h in opener().handlers if isinstance(h,urllib.request.HTTPSHandler))
  self.assertEqual(handler._context.verify_mode,ssl.CERT_REQUIRED);self.assertTrue(handler._context.check_hostname)
 def archive(self,path,extra=None):
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for name in MEMBERS:z.writestr(name,'{}')
   if extra:z.writestr(extra,'x')
 def test_required_only_archive_inventory(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'test.zip';self.archive(p,'annotations/captions_train2017.json');r=inspect_archive(p)
   self.assertEqual(r['selected_extracted_bytes'],4);self.assertEqual(sum(x['selected'] for x in r['members']),2)
 def test_path_traversal_and_absolute_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   for i,name in enumerate(['../outside','/root/file','annotations/../../escape','C:/escape','a\\b']):
    p=Path(tmp)/(str(i)+'.zip');self.archive(p,name)
    with self.assertRaises(ValueError):inspect_archive(p)
 def test_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.zip';self.archive(p)
   with zipfile.ZipFile(p,'a') as z:
    info=zipfile.ZipInfo('link');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,'target')
   with self.assertRaises(ValueError):inspect_archive(p)
 def test_missing_required_metadata_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.zip'
   with zipfile.ZipFile(p,'w') as z:z.writestr(MEMBERS[0],'{}')
   with self.assertRaises(ValueError):inspect_archive(p)
 def test_expansion_bomb_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.zip';self.archive(p)
   with zipfile.ZipFile(p,'a',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('bomb',b'0'*1000000)
   with self.assertRaises(ValueError):inspect_archive(p)
 def test_atomic_marker_cannot_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'marker.json';commit(p,{'complete':True})
   with self.assertRaises(FileExistsError):commit(p,{'complete':False})
   self.assertEqual(read(p),{'complete':True});self.assertFalse(list(Path(tmp).glob('*.pending-*')))
 def test_verified_completion_recovery_without_network(self):
  import acquire
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'archive.zip';target.write_bytes(b'fixture');digest=sha(target)
   state=dict(actual_body_bytes=7,conservative_charged_bytes=7,archive_attempts=1)
   with patch.object(acquire,'OUT',Path(tmp)),patch.object(acquire,'SIZE',7),patch.object(acquire,'SHA',digest):acquire.complete(target,state,recovered=True)
   self.assertTrue(read(Path(tmp)/'acquisition-complete.json')['recovered_completion'])

if __name__=='__main__':unittest.main()
