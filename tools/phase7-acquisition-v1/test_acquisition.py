import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from acquire import headers_valid,reserve,publish
from core import CAP,proposal,sha

class AcquisitionTests(unittest.TestCase):
 def row(self):return proposal()['objects'][0]
 def headers(self,offset=0):
  r=self.row();d={'Content-Length':str(r['bytes']-offset),'ETag':r['etag'],'Content-Type':'application/zip'}
  if offset:d['Content-Range']=f"bytes {offset}-{r['bytes']-1}/{r['bytes']}"
  return SimpleNamespace(status=206 if offset else 200,headers=d)
 def test_exact_scope(self):self.assertEqual(sum(r['bytes'] for r in proposal()['objects']),414291789)
 def test_source_header_pin(self):headers_valid(self.headers(),self.row())
 def test_changed_header_rejected(self):
  for key,value in [('ETag','other'),('Content-Length','1'),('Content-Type','text/html'),('Content-Encoding','gzip')]:
   h=self.headers();h.headers[key]=value
   with self.assertRaises(ValueError):headers_valid(h,self.row())
 def test_resume_range(self):headers_valid(self.headers(100),self.row(),100)
 def test_no_ignored_range(self):
  h=self.headers(100);h.status=200
  with self.assertRaises(ValueError):headers_valid(h,self.row(),100)
 def test_wrong_range_rejected(self):
  h=self.headers(100);h.headers['Content-Range']='bytes 0-1/2'
  with self.assertRaises(ValueError):headers_valid(h,self.row(),100)
 def test_attempt_budget(self):
  state=dict(conservative_body_charge=0,requests=[]);reserve(state,'one',4);reserve(state,'one',3)
  with self.assertRaises(RuntimeError):reserve(state,'one',1)
 def test_transfer_reservation_before_request(self):
  state=dict(conservative_body_charge=CAP-10,requests=[]);reserve(state,'one',10)
  self.assertEqual(state['conservative_body_charge'],CAP)
  with self.assertRaises(RuntimeError):reserve(state,'two',1)
 def test_interrupted_attempt_keeps_full_charge(self):
  state=dict(conservative_body_charge=0,requests=[]);r=reserve(state,'one',100);r['actual_body_bytes']=20;reserve(state,'one',80)
  self.assertEqual(state['conservative_body_charge'],180)
 def test_exclusive_publish(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);(d/'part').write_bytes(b'new');(d/'old').write_bytes(b'old')
   with self.assertRaises(FileExistsError):publish(d/'part',d/'old')
   self.assertEqual((d/'old').read_bytes(),b'old');publish(d/'part',d/'new');self.assertFalse((d/'part').exists())
 def test_prefix_hash(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'part';p.write_bytes(b'prefix');before=sha(p);p.write_bytes(b'changed');self.assertNotEqual(before,sha(p))

if __name__=='__main__':unittest.main()
