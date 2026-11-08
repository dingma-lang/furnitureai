import collections as C
import unittest
from core import *
from audit_lvis import geometry,connect
from cross_reference import iou,photo_id,exact_pairs,candidate_partition,preserved_status

class PureTests(unittest.TestCase):
 def test_metadata_photo_identity(self):
  self.assertEqual(photo_id('http://farm1.staticflickr.com/2/12345_abcd.jpg'),'flickr-photo:12345')
  self.assertEqual(photo_id('https://www.flickr.com/photos/owner/12345/'),'flickr-photo:12345')
 def test_iou_math(self):
  self.assertEqual(iou([0,0,2,2],[0,0,2,2]),1);self.assertEqual(iou([0,0,1,1],[2,2,1,1]),0);self.assertAlmostEqual(iou([0,0,2,2],[1,1,2,2]),1/7)
 def test_same_native_geometry_not_duplicate_objects(self):
  native=[dict(id=1,bbox=[0,0,2,2]),dict(id=2,bbox=[0,0,2,2])];old=[dict(id=3,bbox=[0,0,2,2],category_id=62)]
  pairs,left=exact_pairs(native,old,232);self.assertEqual(len(pairs),1);self.assertEqual(len(left),1)
 def test_table_taxonomy_never_silently_equated(self):
  native=[dict(id=1,bbox=[0,0,2,2])];old=[dict(id=2,bbox=[0,0,2,2],category_id=67)]
  self.assertEqual(exact_pairs(native,old,1050),([],native))
 def test_preserve_old_and_new_status_schemas(self):
  self.assertEqual(preserved_status({'eligibility_status':'quarantined_scene'},'fallback'),'quarantined_scene')
  self.assertEqual(preserved_status({'status':'quarantined'},'fallback'),'quarantined')
 def test_known_group_partition_and_reproducibility(self):
  rows=[dict(coco_image_id=i,known_source_group='source:'+str(i//2)) for i in range(20)]
  a=candidate_partition(rows);self.assertEqual(a,candidate_partition(list(reversed(rows))))
  self.assertTrue(all(a[i]==a[i+1] for i in range(0,20,2)))
 def test_missing_mask_remains_flagged(self):
  kind,flags=geometry(dict(bbox=[10,10,50,50]),100,100);self.assertEqual(kind,'missing');self.assertIn('missing_mask',flags)
 def test_nonfinite_source_box_stops(self):
  with self.assertRaises(ValueError):geometry(dict(bbox=[float('nan'),0,1,1]),100,100)
 def test_polygon_integrity_and_geometry_proxies(self):
  kind,flags=geometry(dict(bbox=[0,0,10,10],segmentation=[[0,0,10,0,10,10]]),100,100)
  self.assertEqual(kind,'polygons');self.assertIn('border_contact_proxy',flags)
  with self.assertRaises(ValueError):geometry(dict(bbox=[1,1,10,10],segmentation=[[1,2,3]]),100,100)

@unittest.skipUnless((OUT/'annotation-overlay-summary-v2.json').exists(),'Requires completed approved metadata audit')
class CompletedAuditTests(unittest.TestCase):
 def test_native_counts_and_segmentation_exact(self):
  d=read(OUT/'furniture-counts.json')['splits']
  self.assertEqual(d['train']['images'],100170);self.assertEqual(d['val']['images'],19809)
  for split in d.values():
   for r in split['furniture_categories']:
    self.assertEqual(r['instances'],r['boxes']);self.assertEqual(r['instances'],r['segmented_instances']);self.assertEqual(r['unknown_presence_images']+r['explicit_negative_images']+r['images'],split['images'])
 def test_exact_77_join_and_native_role_separation(self):
  d=read(OUT/'existing-photo-cross-reference-v2.json');self.assertEqual(len(d['records']),77);self.assertTrue(all(r['lvis_image_available'] for r in d['records']))
  s=read(OUT/'annotation-overlay-summary-v2.json');self.assertEqual(s['role_accounting']['seven_core_classes']['instances_boxes_and_masks'],127);self.assertEqual(s['role_accounting']['whole_objects_including_distinct_subtypes']['instances_boxes_and_masks'],178)
  self.assertEqual(s['role_accounting']['parts_or_outdoor_not_core_furniture']['instances_boxes_and_masks'],38)
 def test_existing_quality_quarantine_not_lifted(self):
  byid={r['coco_image_id']:r for r in read(OUT/'existing-photo-cross-reference-v2.json')['records']}
  self.assertEqual(byid[14883]['historical_status'],'quarantined_scene');self.assertEqual(byid[155937]['historical_status'],'quarantined');self.assertTrue(all(not r['annotation_reuse_authorized_for_training'] for r in byid.values()))
 def test_no_lvis_source_split_leakage(self):
  db=connect(OUT/'furniture-index.sqlite')
  self.assertEqual(db.execute("SELECT COUNT(*) FROM images a JOIN images b ON a.id=b.id WHERE a.split='train' AND b.split='val'").fetchone()[0],0);db.close()
 def test_frozen_source_groups_excluded(self):
  d=read(OUT/'frozen-and-source-group-isolation.json');self.assertEqual(len(d['frozen_source_group_hits']),5);self.assertTrue(d['proposed_ids_and_known_groups_isolated']);self.assertEqual(d['frozen_pixels_opened'],0)
  ids={r['lvis_image_id'] for r in d['frozen_source_group_hits']};candidate=read(OUT/'proposed-photo-rights-audit-v1.json')['images'];self.assertFalse(ids&{r['lvis_image_id'] for r in candidate})
 def test_proposed_400_are_not_rights_approved(self):
  d=read(OUT/'proposed-photo-rights-audit-v1.json');self.assertEqual(d['image_count'],400);self.assertEqual(d['proposed_allocation'],dict(train=320,development=80));self.assertTrue(all(not r['photo_acquisition_eligible'] for r in d['images']));self.assertTrue(all(not r['human_reviewed'] and r['lvis_split']=='train' and r['coco_source_split']=='train' for r in d['images']))
 def test_old_rate_limited_source_ids_never_reselected(self):
  old=read(ROOT/'data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json')['records'];new=read(OUT/'proposed-photo-rights-audit-v1.json')['images'];self.assertFalse({r['coco_image_id'] for r in old}&{r['coco_image_id'] for r in new})
 def test_published_hashes_and_limits(self):
  d=read(OUT/'acquisition-complete.json');self.assertEqual(len(d['archives']),2);self.assertEqual(sum(r['bytes'] for r in d['archives'].values()),414291789);self.assertLessEqual(d['conservative_body_charge'],CAP);self.assertLess(allocated(),DISK)
 def test_ancillary_drawers_do_not_become_cabinets(self):
  d=read(OUT/'additional-unmapped-furniture-categories.json')
  for split in d['splits'].values():
   r=next(r for r in split if r['native_id']==390);self.assertEqual(r['name'],'drawer');self.assertIsNone(r['seven_class_mapping']);self.assertEqual(r['instances'],r['masks'])
 def test_legacy_quarantine_annotation_opportunity_is_separate(self):
  d=read(OUT/'annotation-overlay-summary-v2.json');r=d['new_core_class_on_previously_accepted_COCO_photos']
  self.assertEqual((r['images'],r['instances']),(10,33));self.assertFalse(r['automatic_annotation_eligibility_granted'])
 def test_no_full_extraction_under_storage_cap(self):
  d=read(OUT/'archive-integrity.json');self.assertGreater(d['archive_plus_full_extraction_bytes'],DISK);self.assertFalse(d['full_extraction_performed']);self.assertFalse(list(OUT.glob('lvis_v1_*.json')))

if __name__=='__main__':unittest.main()
