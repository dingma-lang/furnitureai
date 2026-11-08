import base64,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from photo_simulator.server import Store,decode_image

def payload(format='PNG'):
 image=Image.new('RGB',(160,120),'beige');buf=io.BytesIO();image.save(buf,format=format)
 return {'image':'data:image/'+{'PNG':'png','JPEG':'jpeg','WEBP':'webp'}[format]+';base64,'+base64.b64encode(buf.getvalue()).decode(),'name':'Isolated test fixture'}

class ServiceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.store=Store(self.tmp.name,'cpu')
 def tearDown(self):self.store.pool.shutdown(wait=True);self.tmp.cleanup()
 def test_upload_formats_and_original_preserved(self):
  for fmt in ['PNG','JPEG','WEBP']:
   value=payload(fmt);p=self.store.create(value);self.assertEqual((p['image_width'],p['image_height']),(160,120));self.assertFalse(p['physical_fit_verified']);self.assertEqual((self.store.folder(p['id'])/'original-upload.bin').read_bytes(),base64.b64decode(value['image'].split(',')[1]))
 def test_decode_rejects_unsupported_and_broken_input(self):
  for value in ['data:image/svg+xml;base64,AAAA','data:image/png;base64,!!!!','data:image/png;base64,AAAA']:
   with self.assertRaises(Exception):decode_image(value)
 def test_path_traversal_rejected(self):
  for id in ['../','a/../b','/tmp','x'*32]:
   with self.assertRaises(ValueError):self.store.folder(id)
 def test_settings_persist_reference_without_fit_claim(self):
  p=self.store.create(payload());settings={'box':[20,20,100,90],'placement':{'x':80,'y':70,'width':70,'height':60},'reference':{'value':2,'units':'m','note':'user supplied, unverified'}}
  self.store.settings(p['id'],settings);reopen=self.store.read(p['id']);self.assertEqual(reopen['settings'],settings);self.assertFalse(reopen['physical_fit_verified'])
 def test_asset_alpha_and_prior_versions_preserved(self):
  p=self.store.create(payload());p=self.store.set_asset(p['id'],payload());first=p['asset']['file'];p=self.store.set_asset(p['id'],payload());self.assertNotEqual(first,p['asset']['file']);self.assertTrue((self.store.folder(p['id'])/first).exists())
 def test_active_job_prevents_competing_writes(self):
  p=self.store.create(payload());self.store.active.add(p['id'])
  with self.assertRaises(ValueError):self.store.settings(p['id'],{})
 def test_render_without_detection_or_depth_keeps_original(self):
  p=self.store.create(payload());p=self.store.set_asset(p['id'],payload());id=p['id'];original=(self.store.folder(id)/'original.png').read_bytes();j='synthetic';self.store.jobs[j]={}
  self.store.execute(j,id,'render',{'box':[20,20,110,95],'placement':{'x':80,'y':60,'width':75,'height':60},'options':{'depth_assist':True}})
  self.assertEqual(self.store.jobs[j]['status'],'complete');p=self.store.read(id);self.assertIn('unavailable',p['last_diagnostics']['depth_status']);self.assertEqual((self.store.folder(id)/'original.png').read_bytes(),original);self.assertTrue(p['latest_result_url'])
 def test_empty_mask_failure_preserves_previous_project(self):
  p=self.store.create(payload());id=p['id'];j='failure';self.store.jobs[j]={};before=self.store.read(id)
  self.store.execute(j,id,'mask',{'box':[20,20,100,90],'mask_edits':[{'mode':'subtract','points':[[0,0],[159,0],[159,119],[0,119]]}]})
  self.assertEqual(self.store.jobs[j]['status'],'failed');self.assertEqual(self.store.read(id),before)
 def test_worker_failure_reported_and_manual_work_still_available(self):
  p=self.store.create(payload());id=p['id'];j='failed-worker';self.store.jobs[j]={}
  with patch('photo_simulator.server.subprocess.Popen',side_effect=RuntimeError('simulated missing worker')):self.store.execute(j,id,'analyze',{})
  self.assertEqual(self.store.jobs[j]['status'],'failed');self.assertTrue((self.store.folder(id)/'original.png').exists());self.store.settings(id,{'box':[20,20,100,90]})

if __name__=='__main__':unittest.main()
