import collections as C
import unittest
from audit_core import *

class RegistrationTests(unittest.TestCase):
 def test_ten_existing_photos_remain_inactive(self):
  r=read(OUT/'existing-ten-photo-registration-assessment.json');self.assertEqual((r['photos'],r['native_annotations']),(10,33));self.assertEqual(r['photo_hashes_verified'],10);self.assertEqual(r['annotations_merged_into_training'],0)
  self.assertTrue(all(not a['annotation_training_eligible'] and not a['semantic_unit_verified'] and not a['image_to_mask_visual_registration_verified'] for a in r['records']))
 def test_preflight_preserves_source_shapes_and_frozen_exclusions(self):
  p=read(OUT/'preflight.json');self.assertEqual(p['core_boxes_masks_verified'],905);self.assertEqual(p['all_native_annotations_verified'],1080);self.assertEqual(len(p['five_frozen_related_ids_excluded']),5);self.assertFalse(p['known_historical_source_overlaps'])

@unittest.skipUnless((OUT/'summary.json').exists(),'Audit finalization still in progress')
class CompletedTests(unittest.TestCase):
 def test_exact_records_and_original_partitions(self):
  m=manifest();original={r['coco_image_id']:r for r in m['images']};audit=read(OUT/'rights-audit.json');self.assertEqual(len(audit['records']),400)
  for r in audit['records']:self.assertEqual(r['provisional_partition'],original[r['coco_image_id']]['proposed_partition']);self.assertLessEqual(r['evidence_attempts'],1);self.assertFalse(r['human_reviewed'])
 def test_unattempted_checkpoint_is_not_retry_authorization(self):
  checkpoint=read(OUT/'unattempted-checkpoint-v1.json');self.assertFalse(checkpoint['retry_or_resume_authorized']);self.assertTrue(checkpoint['new_separate_approval_required'])
  attempted={r['coco_image_id'] for r in read(OUT/'rights-audit.json')['records'] if r['attempted']};self.assertFalse(attempted&{r['coco_image_id'] for r in checkpoint['records']})
 def test_only_verified_isolated_candidates_can_advance(self):
  proposal=read(OUT/'proposed-image-acquisition-v1.json');source={r['coco_image_id']:r for r in manifest()['images']}
  decisions={r['coco_image_id']:r for r in read(OUT/'identity-audit.json')['records']}
  for r in proposal['images']:
   self.assertTrue(decisions[r['coco_image_id']]['eligible_for_acquisition_proposal']);self.assertEqual(r['current_rights_status'],STATUSES[0]);self.assertEqual(r['original_lvis_furniture_annotations'],source[r['coco_image_id']]['original_lvis_furniture_annotations']);self.assertFalse(r['photo_acquisition_approved'])
 def test_known_groups_do_not_cross_proposed_partitions(self):
  groups=C.defaultdict(set)
  for r in read(OUT/'proposed-image-acquisition-v1.json')['images']:groups[r['identity_decision']['known_source_group']].add(r['proposed_partition'])
  self.assertTrue(all(len(p)==1 for p in groups.values()))
 def test_transfer_and_one_GET_bounds(self):
  d=read(OUT/'network-ledger.json');self.assertEqual(d['actual_body_bytes'],sum(r['actual_body_bytes'] for r in d['requests']));self.assertLessEqual(d['conservative_charge'],CAP)
  per=C.Counter(r['coco_image_id'] for r in d['requests'] if r['method']=='GET');self.assertTrue(all(v<=1 for v in per.values()));self.assertLess(allocated(),DISK)

if __name__=='__main__':unittest.main()
