import unittest
from unittest.mock import patch
from core import *
from data import exposed_rows,final_rows_metadata,image
from metrics import evaluate
from siglip import suitable,should_abstain

class HistoricalProtocol(unittest.TestCase):
 def test_original_saved_reference_metrics_reproduced_without_inference(self):
  rows=exposed_rows();folder=ROOT/'data/open-images/experiments/phase5b-v1/original-development-v1'
  predictions={r['image_id']:read(folder/(r['image_id']+'.json'))['predictions'] for r in rows};m=evaluate(rows,predictions)
  self.assertEqual((m['tp'],m['fp'],m['fn']),(28,101,13));self.assertAlmostEqual(m['ap50'],.7215544615149045);self.assertAlmostEqual(m['ap75'],.5191443602038877)
 def test_adapted_saved_reference_metrics_reproduced_without_inference(self):
  rows=exposed_rows();folder=ROOT/'data/open-images/experiments/phase5c-v1/comparison-v1/heads-lr3e-5'
  predictions={r['image_id']:read(folder/(r['image_id']+'.json'))['predictions'] for r in rows};m=evaluate(rows,predictions)
  self.assertEqual((m['tp'],m['fp'],m['fn']),(29,67,12));self.assertAlmostEqual(m['ap50'],.7956075915382238);self.assertAlmostEqual(m['ap75'],.5359586327385157)
 def test_fresh_pixels_require_locked_selection_before_open(self):
  r=final_rows_metadata()[0]
  with patch('data.checked',return_value=None),patch('PIL.Image.open',side_effect=AssertionError('Pixels must stay unopened')):
   with self.assertRaises(PermissionError):image(r)
 def test_forged_membership_and_frozen_access_rejected(self):
  r=dict(exposed_rows()[0],phase9_partition='frozen')
  with patch('PIL.Image.open',side_effect=AssertionError('Frozen pixels forbidden')):
   with self.assertRaises(ValueError):image(r)
 def test_fresh_membership_positive_counts_and_partial_FP_scope(self):
  rows=final_rows_metadata();self.assertEqual(len(rows),36);self.assertEqual(sum(len(r['annotations']) for r in rows),126)
  self.assertTrue(any(any(r['coverage'][c] and not r['fp_coverage'][c] for c in CLASSES) for r in rows))
 def test_SigLIP_unsuitable_crop_does_not_force_category(self):
  self.assertFalse(suitable(dict(box=[0,0,.9,.9]),640,480));self.assertFalse(should_abstain(dict(label='chair'),dict(status='unsuitable_geometry_unresolved')))
 def test_SigLIP_disagreement_margin_and_no_relabel(self):
  p=dict(label='chair');self.assertTrue(should_abstain(p,dict(status='category_scored',category='bed',relative_softmax_margin=.2)));self.assertEqual(p['label'],'chair')
  self.assertFalse(should_abstain(p,dict(status='category_scored',category='bed',relative_softmax_margin=.05)))

if __name__=='__main__':unittest.main()
