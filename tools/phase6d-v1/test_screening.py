import unittest
from common import *
from screen import signature,duplicate,iou,annotation_diagnostics

class Tests(unittest.TestCase):
 def test_geometry_and_ambiguity_measurements(self):
  a=dict(id=1,category_id=63,bbox=[0,0,50,50],iscrowd=0);b=dict(id=2,category_id=65,bbox=[0,0,50,50],iscrowd=0)
  d=annotation_diagnostics([a,b],100,100)
  self.assertEqual(len(d['category_conflicts']),1);self.assertEqual(iou([0,0,50,50],[0,0,50,50]),1);self.assertEqual(len(d['border_contact_ids']),2)
 def test_exact_and_perceptual_checks_are_distinct(self):
  self.assertEqual(duplicate({'sha256':'a'},{'sha256':'a'}),'exact')
  self.assertEqual(duplicate({'dhash64':'0'},{'dhash64':'1'}),'suspected_perceptual')
  self.assertIsNone(duplicate({'sha256':'a'},{'sha256':'b'}))
 def test_blank_signature(self):
  from PIL import Image
  s=signature(Image.new('RGB',(50,50),'white'));self.assertEqual(s['white_fraction'],1);self.assertEqual(s['gray_std'],0)
 def test_actual_memberships_annotations_and_final_gate(self):
  if not (OUT/'screening-v2.json').exists():self.skipTest('Screening not yet completed')
  s=read(OUT/'screening-v2.json');original={r['image_id']:r for r in manifest()['images']};kept=read(OUT/'eligible-candidates-v1.json')['images']
  self.assertEqual(len(s['images']),72);self.assertEqual(len(kept),55);self.assertEqual(sum(r['proposed_split']=='train' for r in kept),45)
  for r in s['images']:
   self.assertEqual(r['original_annotations'],original[r['image_id']]['original_furniture_annotations']);self.assertEqual(r['proposed_split'],original[r['image_id']]['proposed_split']);self.assertFalse(r['human_verified'])
  self.assertTrue(all(r['status']=='provisionally_accepted' and not r['near_coincident_exclusive_furniture_categories'] for r in kept))
 def test_all_acquired_hashes_and_group_isolation(self):
  if not (OUT/'acquisition-complete.json').exists():self.skipTest('Acquisition not completed')
  for r in read(OUT/'acquisition-complete.json')['records']:self.assertEqual(sha(OUT/r['file']),r['sha256'])
  rows=read(OUT/'eligible-candidates-v1.json')['images'];train={r['identity_group'] for r in rows if r['proposed_split']=='train'};dev={r['identity_group'] for r in rows if r['proposed_split']=='development'};self.assertFalse(train&dev)
 def test_transfer_counts_and_no_model_execution(self):
  if not (OUT/'integrity-verification.json').exists():self.skipTest('Verification not completed')
  v=read(OUT/'integrity-verification.json');self.assertEqual(v['photo_body_bytes'],10853122);self.assertEqual(v['retries'],0);self.assertEqual(v['model_inference_or_training'],0)
 def test_combined_pool_assessment_is_not_training_authorization(self):
  if not (OUT/'phase6e-training-readiness.json').exists():self.skipTest('Assessment not completed')
  r=read(OUT/'phase6e-training-readiness.json');self.assertEqual(r['nominal_combined_train_images'],81);self.assertEqual(r['conservative_positive_scope_train_images'],74);self.assertEqual(r['fresh_development_images'],15);self.assertFalse(r['training_authorized'])
  self.assertEqual(r['conservative_train_counts']['desk']['boxes'],0);self.assertEqual(r['fresh_development_counts']['lamp']['boxes'],0)

if __name__=='__main__':unittest.main()
