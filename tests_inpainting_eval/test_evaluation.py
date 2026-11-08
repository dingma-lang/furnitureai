import tempfile,unittest,json
from pathlib import Path
import numpy as np
from PIL import Image
from inpainting_eval.evaluate import evaluate,seam_metric
from inpainting_eval.backends import fit_pad,restore,BackendUnavailable

class EvaluationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.image=Image.new('RGB',(80,60),(100,120,130));self.mask=Image.new('L',(80,60));self.mask.paste(255,(10,10,60,50));self.protected=Image.new('L',(80,60));self.protected.paste(255,(20,20,40,30))
 def tearDown(self):self.tmp.cleanup()
 def run_case(self,config=None,backend=None):return evaluate(self.image,self.mask,self.protected,config or {},self.root/'run',backend)
 def test_aggressive_candidate_guard_and_raw_retention(self):
  def malicious(image,mask,p,c):
   raw=Image.new('RGB',image.size,'red');return raw,raw,{}
  r=self.run_case({'backend':'test-fixture'},malicious)
  self.assertEqual(r['status'],'complete');self.assertEqual(r['outside_mask_pixel_differences'],0);self.assertEqual(r['protected_pixel_differences'],0);self.assertGreater(r['raw_protected_pixel_differences'],0);self.assertTrue(r['original_unchanged']);self.assertTrue((self.root/'run/raw-native.png').exists())
 def test_crop_mapping_no_changes_outside_crop(self):
  r=self.run_case({'crop':[30,30,70,60]},lambda i,m,p,c:(Image.new('RGB',i.size,'blue'),Image.new('RGB',i.size,'blue'),{}))
  im=np.array(Image.open(self.root/'run/background.png'));self.assertEqual(r['permitted_pixels'],600);self.assertTrue((im[35,35]==[0,0,255]).all());self.assertTrue((im[15,15]==[100,120,130]).all())
 def test_unequal_mask_rejected(self):
  self.mask=Image.new('L',(79,60));self.assertEqual(self.run_case()['status'],'failed')
 def test_nonfinite_mask_rejected(self):
  self.mask=np.full((60,80),float('nan'));self.assertEqual(self.run_case()['status'],'failed')
 def test_invalid_crop_rejected(self):self.assertEqual(self.run_case({'crop':[-1,0,50,50]})['status'],'failed')
 def test_empty_authorized_region_rejected(self):
  self.protected=Image.new('L',(80,60),255);self.assertEqual(self.run_case()['status'],'failed')
 def test_bad_backend_resolution_fails_with_raw_preserved(self):
  r=self.run_case({},lambda i,m,p,c:(Image.new('RGB',(8,8)),Image.new('RGB',(8,8)),{}));self.assertEqual(r['status'],'failed');self.assertTrue((self.root/'run/raw-native.png').exists())
 def test_no_silent_neural_fallback(self):
  r=self.run_case({'backend':'diffusion','model_path':str(self.root/'absent')});self.assertEqual(r['status'],'unavailable');self.assertFalse((self.root/'run/background.png').exists());self.assertFalse(r['neural_inference_executed'])
 def test_lama_missing_graceful(self):self.assertEqual(self.run_case({'backend':'lama'})['status'],'unavailable')
 def test_authorization_required_for_existing_path(self):
  r=self.run_case({'backend':'diffusion','model_path':str(self.root)});self.assertEqual(r['status'],'unavailable');self.assertIn('authorization',r['error'])
 def test_fit_pad_preserves_aspect_and_mask_labels(self):
  image=Image.new('RGB',(1600,1067));mask=Image.new('L',image.size);mask.paste(255,(100,100,500,600));i,m,meta=fit_pad(image,mask,512)
  self.assertEqual(i.size,m.size);self.assertEqual(i.width%8,0);self.assertEqual(i.height%8,0);self.assertTrue(set(np.unique(m)).issubset({0,255}));self.assertEqual(meta['resized'],(512,341))
 def test_classical_original_and_guard(self):
  r=self.run_case({'backend':'classical','classical':{'exclude_foreground_donors':True}});self.assertEqual(r['status'],'complete');self.assertEqual(r['outside_mask_pixel_differences'],0);self.assertTrue(r['original_unchanged'])
 def test_no_overwrite(self):
  self.run_case()
  with self.assertRaises(FileExistsError):self.run_case()
 def test_seam_not_a_score(self):
  d=seam_metric(np.full((10,10,3),50,np.uint8),np.pad(np.ones((4,4),bool),3));self.assertEqual(d['mean_rgb_jump_0_255'],0);self.assertIn('not a perceptual',d['interpretation'])
