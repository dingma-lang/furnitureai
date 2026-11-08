import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=read(MANIFEST);cls.a=read(DEST/'acquisition-complete.json');cls.s=read(DEST/'screening-v3.json');cls.v=read(DEST/'integrity-verification.json')
 def test_exact_approved_membership_and_slots(self):
  from collections import Counter
  self.assertEqual({r['image_id'] for r in self.a['records']},{r['image_id'] for r in self.m['images']})
  self.assertEqual(Counter(r['proposed_split'] for r in self.s['images']),dict(train=32,development=9))
 def test_all_photo_hashes_and_sizes(self):
  for r in self.a['records']:
   self.assertEqual(sha(DEST/r['file']),r['sha256']);self.assertEqual((DEST/r['file']).stat().st_size,r['bytes']);self.assertLessEqual(r['bytes'],IMAGE_MAX)
 def test_all_decoded_dimensions_match(self):
  from PIL import Image
  for r in self.a['records']:
   with Image.open(DEST/r['file']) as im:self.assertEqual(im.size,(r['width'],r['height']));im.verify()
 def test_original_annotations_never_rewritten(self):
  lookup={r['image_id']:r for r in self.m['images']}
  for r in self.s['images']:self.assertEqual(r['original_annotations'],lookup[r['image_id']]['annotations'])
  self.assertEqual(self.v['source_confirmed_native_boxes'],108)
 def test_finite_source_geometry_and_category_ids(self):
  for r in self.s['images']:
   for g in r['original_annotations']:box(g['box']);self.assertEqual(g['source_mid'],MIDS[g['label']])
 def test_license_evidence_copies_match_verified_records(self):
  for row in self.m['images']:
   f=DEST/'licensing-evidence'/(row['image_id']+'.json');self.assertEqual(sha(f),row['license_record_sha256']);rec=read(f)
   self.assertEqual(sha(DEST/'licensing-evidence'/(row['image_id']+'.response')),rec['response_sha256']);self.assertEqual(license_url(rec['license_url']),'https://creativecommons.org/licenses/by/2.0/')
 def test_transfer_attempts_and_ceilings(self):
  ledger=read(DEST/'network-ledger.json');self.assertEqual(ledger['actual_bytes'],10487710);self.assertEqual(len(ledger['requests']),41)
  self.assertTrue(all(r['attempt']==1 and r['complete'] for r in ledger['requests']));self.assertLess(self.v['conservative_combined_charge'],TRANSFER)
 def test_effective_screening_pointer_and_quarantine(self):
  pointer=read(DEST/'CURRENT-SCREENING.json');self.assertEqual(pointer['path'],'screening-v3.json');self.assertEqual(pointer['sha256'],sha(DEST/pointer['path']))
  eligible=read(DEST/'eligible-candidates-v1.json')['images'];self.assertEqual(len(eligible),14);self.assertTrue(all(r['status']=='provisional_automated_candidate' for r in eligible));self.assertEqual(sum(r['status']=='quarantined' for r in self.s['images']),27)
 def test_partition_identity_and_frozen_protection(self):
  self.assertTrue(isolation(self.s['images'],set(read(OUT/'protected-identities.json'))));self.assertEqual(self.v['frozen_pixels_decoded_or_scored'],0)
 def test_ambiguous_supervision_is_masked(self):
  for r in read(DEST/'eligible-candidates-v1.json')['images']:
   self.assertFalse(r['coverage']['desk']);self.assertFalse(r['coverage']['cabinet'])
   self.assertTrue(all(g['ignore'] for g in r['annotations'] if g['label'] in ('desk','cabinet')))
  self.assertEqual(sum(r['categories']['table']['boxes'] for r in [read(DEST/'train-eligible-counts.json')]),0)
 def test_duplicate_checks_and_annotation_coverage_are_not_accuracy(self):
  self.assertFalse(self.s['duplicates']);self.assertFalse(self.v['abo_exact_overlap']);self.assertTrue(all(not r['human_verified'] for r in self.s['images']))
 def test_no_training_or_model_inference(self):
  self.assertEqual(self.v['optimizer_updates'],0);self.assertEqual(self.v['model_inference'],0);self.assertFalse(read(DEST/'eligible-candidates-v1.json')['training_approved'])
 def test_additional_archives_are_unapproved_and_tls_checked(self):
  p=read(DEST/'expansion-proposal-v1/coco-metadata-acquisition-proposal.json')
  self.assertFalse(p['approval_granted']);self.assertEqual(p['expected_transfer_bytes'],252907541);self.assertEqual(p['objects'][0]['sha256'],'113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268')
  self.assertEqual(p['photographs_authorized'],0);self.assertFalse((ROOT/'data/coco/metadata/phase6b-v1').exists())

if __name__=='__main__':unittest.main()
