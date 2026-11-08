import tempfile
import unittest
from types import SimpleNamespace
from pilot_core import *
from acquire import validate_headers,publish,inspect_download,NoRedirect

class Tests(unittest.TestCase):
 def test_exact_approved_ids_and_allocations(self):
  m=manifest();self.assertEqual(len(m['images']),27);self.assertEqual(sum(r['remote_object']['bytes'] for r in m['images']),3913471)
 def test_pinned_size_version_and_jpeg(self):
  row=manifest()['images'][0];obj=row['remote_object'];h={'Content-Length':str(obj['bytes']),'ETag':obj['etag'],'Content-Type':'image/jpeg'}
  validate_headers(SimpleNamespace(status=200,headers=h),row)
  for key,value in [('Content-Length','2'),('ETag','changed'),('Content-Type','text/html'),('Content-Encoding','gzip')]:
   with self.assertRaises(ValueError):validate_headers(SimpleNamespace(status=200,headers=dict(h,**{key:value})),row)
 def test_range_resume_requires_exact_tail(self):
  row=manifest()['images'][0];obj=row['remote_object'];h={'Content-Length':str(obj['bytes']-7),'ETag':obj['etag'],'Content-Type':'image/jpeg','Content-Range':f"bytes 7-{obj['bytes']-1}/{obj['bytes']}"}
  validate_headers(SimpleNamespace(status=206,headers=h),row,7)
  with self.assertRaises(ValueError):validate_headers(SimpleNamespace(status=200,headers=h),row,7)
 def test_exclusive_publish_preserves_old_file(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'part').write_bytes(b'new');(p/'old').write_bytes(b'old')
   with self.assertRaises(FileExistsError):publish(p/'part',p/'old')
   self.assertEqual((p/'old').read_bytes(),b'old');publish(p/'part',p/'photo');self.assertEqual((p/'photo').read_bytes(),b'new')
 def test_completed_body_decodes_without_network(self):
  from PIL import Image
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'photo.jpg.part';Image.new('RGB',(50,40),'white').save(p,format='JPEG');row=dict(manifest()['images'][0],width=50,height=40);row['remote_object']=dict(row['remote_object'],bytes=p.stat().st_size,etag='"'+hashlib.md5(p.read_bytes()).hexdigest()+'"')
   self.assertEqual(inspect_download(row,p,'photo.jpg',1,sha(p))['sha256'],sha(p))
   row['width']=51
   with self.assertRaises(ValueError):inspect_download(row,p,'photo.jpg',1,sha(p))
 def test_redirects_disabled(self):self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://other'))

if __name__=='__main__':unittest.main()
