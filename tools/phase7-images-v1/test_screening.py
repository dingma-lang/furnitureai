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

@unittest.skipUnless((OUT/'combined-training-readiness.json').exists(),'Requires completed approved acquisition and screening')
class AuditTests(unittest.TestCase):
 def test66_memberships_and_original_geometry_preserved(self):
  source={r['coco_image_id']:r for r in manifest()['images']};rows=read(OUT/'screening-v2.json')['images'];self.assertEqual(len(rows),66)
  for r in rows:self.assertEqual(r['original_annotations'],source[r['coco_image_id']]['original_furniture_annotations']);self.assertEqual(r['proposed_split'],source[r['coco_image_id']]['proposed_split']);self.assertFalse(r['human_verified'])
 def test_quarantine_not_replaced_or_reassigned(self):
  data=read(OUT/'screening-v2.json');self.assertEqual(data['status_counts'],dict(provisional_source_candidate=48,quarantined=18));self.assertEqual(data['eligible_partition_counts'],dict(train=40,development=8))
 def test_usable_masks_count_core_not_parts(self):
  d=read(OUT/'eligible-candidates-v1.json');n=sum(a['id'] in r['valid_core_annotation_ids'] and a['category_id'] in CLASSES for r in d['images'] for a in r['original_annotations']);self.assertEqual(n,81)
 def test_combined_unique_images_and_known_groups(self):
  d=read(OUT/'combined-training-readiness.json');self.assertEqual((d['prospective_unique_training_photographs'],d['prospective_unique_fresh_development_photographs']),(114,23));self.assertFalse(d['exact_duplicate_photo_references']);self.assertFalse(d['prospective_source_group_exclusions'])
  bygroup={}
  for r in d['source_rows']:
   group=r['known_source_component'];bygroup.setdefault(group,set()).add(r['split'])
  self.assertTrue(all(len(parts)==1 for parts in bygroup.values()))
 def test_no_double_count_of33_existing_overlay_annotations(self):
  d=read(OUT/'combined-training-readiness.json');self.assertEqual(d['existing_COCO_LVIS_overlay_annotations_inactive'],33);self.assertEqual(d['total_training_instance_masks'],285);self.assertEqual(d['total_fresh_development_instance_masks'],76)
 def test_native_table_strata_and_weak_categories_explicit(self):
  d=read(OUT/'combined-training-readiness.json');self.assertIn('COCO:67:dining_table',d['per_category_training']['table']['native_source_strata']);self.assertEqual(d['per_category_fresh_development']['cabinet']['images'],0);self.assertFalse(d['ready_for_broad_seven_class_training'])
 def test_existing_ten_remain_metadata_only_and_inactive(self):
  d=read(OUT/'existing-ten-registration-reassessment.json');self.assertEqual(d['annotations'],33);self.assertEqual(d['annotations_activated_or_merged'],0);self.assertEqual(d['image_pixels_decoded'],0)
 def test_rights_retry_and_frozen_exclusions_preserved(self):
  d=read(OUT/'preflight.json');self.assertEqual(len(d['nine_identity_quarantines_excluded']),9);self.assertTrue(d['unattempted296_not_resumed']);self.assertTrue(d['known_frozen_related_154613_excluded']);self.assertEqual(d['live_licensing_requests'],0)

if __name__=='__main__':unittest.main()
