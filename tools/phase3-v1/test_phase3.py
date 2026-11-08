import copy
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import common as c
import prepare as p
import acquire as a
import runner as r

class Response:
 def __init__(self,data,url,fail=False):
  self.data=data;self.pos=0;self.url=url;self.status=200;self.headers={'Content-Length':str(len(data)),'Content-Type':'image/png','ETag':'"fixture"'};self.fail=fail
 def geturl(self):return self.url
 def read(self,n):
  if self.fail and self.pos:raise OSError('interrupted stream')
  size=min(n,10 if self.fail else n);out=self.data[self.pos:self.pos+size];self.pos+=len(out);return out
 def __enter__(self):return self
 def __exit__(self,*_):pass

class FakeOpener:
 def __init__(self,body,fail=False):self.body=body;self.fail=fail;self.requests=[]
 def open(self,req,timeout):
  self.requests.append(req);response=Response(self.body,req.full_url,self.fail and req.method=='GET');return response

class Phase3Tests(unittest.TestCase):
 def fixture(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);dest=pathlib.Path(tmp.name)
  stream=io.BytesIO();Image.new('RGB',(128,128),'red').save(stream,format='PNG');body=stream.getvalue()
  obj={'image_id':'fixture','product_id':'p','metadata_path':'ab/abcdef.png','url':p.SOURCE+'ab/abcdef.png','destination':'images/ab/abcdef.png'}
  plan={'objects':[obj],'limits':p.LIMITS}
  return dest,body,obj,plan
 def test_reconciliation_originals_not_silently_relabelled(self):
  old=[{'image_id':'a','abo_product_id':'p','sha256':'hash','furniture_class':'chair'}]
  meta={'rows':[],'rejected':[{'image_id':'a','reason':'redundant_known_capture','kept':'b'}]}
  result=p.reconcile(old,meta,old);self.assertEqual(result['arithmetic'],'1 - 1 + 0 = 0')
  meta['rows']=[{**old[0],'furniture_class':'bed'}]
  with self.assertRaises(ValueError):p.reconcile(old,meta,old)
 def test_family_and_duplicate_selection_deterministic(self):
  pool=[{'product_id':str(i),'category':'bed','family_group':str(i//2),'subtype':str(i%3),'image_ids':[f'{i}-a',f'{i}-b']} for i in range(20)]
  quotas={x:5 for x in c.CLASSES};one=p.choose_products(pool,5,quotas);two=p.choose_products(list(reversed(pool)),5,quotas)
  self.assertEqual([x['product_id'] for x in one],[x['product_id'] for x in two]);self.assertEqual(len({x['family_group'] for x in one}),len(one))
 def test_generic_shared_image_blocks_second_selection_not_identity_merge(self):
  pool=[{'product_id':str(i),'category':'bed','family_group':str(i),'subtype':'platform','image_ids':['shared']} for i in range(2)]
  self.assertEqual(len(p.choose_products(pool,2,{x:2 for x in c.CLASSES})),1)
 def test_training_dev_family_and_hash_leakage(self):
  base={'abo_product_id':'fit','leakage_group_id':'g','weak_label':'bed','sha256':'hash','proposed_split':'train'}
  for dev in [{**base,'abo_product_id':'dev'},{**base,'abo_product_id':'dev','leakage_group_id':'h'}]:
   with self.assertRaises(ValueError):r.check_isolation([base],[dev],[])
 def test_ambiguous_product_labels_blocked(self):
  base={'abo_product_id':'p','weak_label':'chair','proposed_split':'train'}
  with self.assertRaises(ValueError):r.check_isolation([base,{**base,'weak_label':'table'}],[],[])
 def test_official_partition_blocked(self):
  with self.assertRaises(ValueError):r.check_isolation([{'abo_product_id':'p','weak_label':'bed','proposed_split':'test'}],[],[])
 def test_safe_interruption_is_charged_and_resume_preserves_partial(self):
  dest,body,obj,plan=self.fixture();bad=FakeOpener(body,True);d=a.Downloader(plan,dest,bad);remote=d.head(obj)
  with self.assertRaises(OSError):d.get(obj,remote)
  self.assertEqual(d.state['charged_bytes'],p.LIMITS['object_bytes']);self.assertEqual(d.state['observed_body_bytes'],10)
  partial=list(dest.rglob('*.part'))[0];before=partial.read_bytes()
  d=a.Downloader(plan,dest,FakeOpener(body));result=d.get(obj,remote)
  self.assertEqual(result['status'],'downloaded');self.assertEqual(d.state['charged_bytes'],2*p.LIMITS['object_bytes']);self.assertEqual(partial.read_bytes(),before)
  self.assertEqual((dest/obj['destination']).read_bytes(),body)
 def test_download_checksum_and_completed_image_immutable(self):
  dest,body,obj,plan=self.fixture();opener=FakeOpener(body);d=a.Downloader(plan,dest,opener);d.get(obj,d.head(obj));count=len(opener.requests)
  d.get(obj,d.head(obj));self.assertEqual(len(opener.requests),count+1)
  (dest/obj['destination']).write_bytes(b'changed')
  with self.assertRaises(ValueError):d.get(obj,{'bytes':len(body),'etag':'"fixture"'})
 def test_download_paths_redirects_and_limits(self):
  dest,body,obj,plan=self.fixture();d=a.Downloader(plan,dest,FakeOpener(body))
  for change in [{'url':'https://evil.example/image'},{'metadata_path':'../../etc/passwd'},{'destination':'../other'}]:
   with self.assertRaises(ValueError):d.head({**obj,**change})
  d.state['charged_bytes']=p.LIMITS['download_bytes']
  with self.assertRaises(RuntimeError):d.get(obj,{'bytes':len(body),'etag':None})
  with self.assertRaises(ValueError):a.NoRedirect().redirect_request(None,None,None,None,None,None)
 def test_download_attempt_limit_and_wrong_binding(self):
  dest,body,obj,plan=self.fixture();d=a.Downloader(plan,dest,FakeOpener(body,True));remote=d.head(obj)
  for _ in range(2):
   with self.assertRaises(OSError):d.get(obj,remote)
  with self.assertRaises(RuntimeError):d.get(obj,remote)
  with self.assertRaises(ValueError):a.Downloader({**plan,'different':True},dest,FakeOpener(body))
 def test_missing_approval_never_enables_network(self):
  with self.assertRaises(PermissionError):a.approval({},None,pathlib.Path('/tmp'))
 def test_pending_commit_recovers_without_overwrite(self):
  dest,body,obj,plan=self.fixture();d=a.Downloader(plan,dest,FakeOpener(body));remote=d.head(obj)
  real_link=a.os.link
  with patch.object(a.os,'link',side_effect=OSError('simulated crash before commit')):
   with self.assertRaises(OSError):d.get(obj,remote)
  d=a.Downloader(plan,dest,FakeOpener(body));d.recover_commits()
  self.assertEqual(d.state['objects']['fixture']['status'],'downloaded');self.assertEqual((dest/obj['destination']).read_bytes(),body)
 def test_bootstrap_reproducible_with_small_class_support(self):
  rows=[{'abo_product_id':str(i),'leakage_group_id':str(i),'weak_label':c.CLASSES[i]} for i in range(7)]
  logits=np.eye(7);one=r.uncertainty(rows,list(range(7)),logits,repeats=30);two=r.uncertainty(rows,list(range(7)),logits,repeats=30)
  self.assertEqual(one,two);self.assertIsNone(one['family_bootstrap_macro_f1_95_percentile_interval']);self.assertGreater(one['bootstrap_missing_category'],0)
 def test_atomic_checkpoint_failure_preserves_original(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=pathlib.Path(tmp)/'checkpoint.pt';c.atomic(path,b'original')
   with patch.object(c.w.os,'replace',side_effect=OSError('interrupt')):
    with self.assertRaises(OSError):c.atomic(path,b'new')
   self.assertEqual(path.read_bytes(),b'original')

if __name__=='__main__':unittest.main(verbosity=2)
