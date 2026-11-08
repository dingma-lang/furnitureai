import copy
import unittest
import numpy as np
from PIL import Image
from pilot_core import *
from screen import annotation_check
from signatures import signature,duplicate

class Tests(unittest.TestCase):
 def fixture(self):return dict(id=1,category_id=361,bbox=[10,10,40,40],area=1600,segmentation=[[10,10,50,10,50,50,10,50]])
 def test_valid_native_box_and_polygon(self):
  a=self.fixture();original=copy.deepcopy(a);d=annotation_check(a,100,100);self.assertTrue(d['valid']);self.assertEqual(d['bbox_polygon_max_delta_pixels'],0);self.assertEqual(a,original)
 def test_invalid_polygon_and_bounds(self):
  a=self.fixture();a['segmentation']=[[10,20,30]];self.assertFalse(annotation_check(a,100,100)['valid'])
  a=self.fixture();a['segmentation'][0][0]=-1;self.assertFalse(annotation_check(a,100,100)['valid'])
 def test_nonfinite_or_outside_box_rejected(self):
  for value in ([float('nan'),0,1,1],[90,90,20,20]):
   a=self.fixture();a['bbox']=value;self.assertFalse(annotation_check(a,100,100)['valid'])
 def test_polygon_box_registration_discrepancy_flag(self):
  a=self.fixture();a['bbox']=[5,5,50,50];self.assertIn('bbox_polygon_extent_discrepancy',annotation_check(a,100,100)['reasons'])
 def test_border_contact_is_not_automatically_invalid(self):
  a=self.fixture();a['bbox']=[0,0,50,50];a['segmentation']=[[0,0,50,0,50,50,0,50]];d=annotation_check(a,100,100);self.assertTrue(d['valid']);self.assertIn('border_contact_proxy',d['reasons'])
 def test_decoded_exact_and_near_duplicate_flags(self):
  a={'sha256':'a','decoded_pixel_sha256':'same'};b={'sha256':'b','decoded_pixel_sha256':'same'};self.assertEqual(duplicate(a,b),'exact')
  self.assertEqual(duplicate({'dhash64':'0'},{'dhash64':'1'}),'suspected_perceptual')
 def test_native_semantics_not_invented_by_geometry(self):
  d=annotation_check(self.fixture(),100,100);self.assertFalse(d['semantic_unit_verified']);self.assertFalse(d['visual_registration_verified'])

if __name__=='__main__':unittest.main()
