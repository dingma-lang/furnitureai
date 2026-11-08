import copy
import contextlib
import io
import unittest
from unittest.mock import patch
from PIL import Image
from pilot_core import *
from acquire import validate_state,inspect_download,run

class IntegrityTests(unittest.TestCase):
 def test_all27_hashes_dimensions_ETag_and_native_annotations(self):
  source={r['coco_image_id']:r for r in manifest()['images']};rows=read(OUT/'screening-v2.json')['images']
  self.assertEqual(len(rows),27)
  for r in rows:
   s=source[r['coco_image_id']];self.assertEqual(r['original_annotations'],s['original_furniture_annotations']);self.assertEqual(r['proposed_split'],s['proposed_split'])
   self.assertEqual(sha(OUT/r['file']),r['sha256']);self.assertEqual(hashlib.md5((OUT/r['file']).read_bytes()).hexdigest(),s['remote_object']['etag'].strip('"'))
   with Image.open(OUT/r['file']) as im:self.assertEqual(im.size,(s['width'],s['height']))
 def test_complete_resume_never_contacts_network_or_rewrites_ledger(self):
  with patch('acquire.urllib.request.build_opener') as opener,patch('acquire.atomic'),contextlib.redirect_stdout(io.StringIO()):
   opener.return_value.open.side_effect=AssertionError('A completed acquisition must not request sources')
   run(resume=True);opener.return_value.open.assert_not_called()
 def test_transfer_and_attempt_journal_corruption_rejected(self):
  state=read(OUT/'network-ledger.json');validate_state(state,manifest()['images'])
  bad=copy.deepcopy(state);bad['actual_body_bytes']+=1
  with self.assertRaises(ValueError):validate_state(bad,manifest()['images'])
  bad=copy.deepcopy(state);next(r for r in bad['requests'] if r['method']=='GET')['attempt']=3
  with self.assertRaises(ValueError):validate_state(bad,manifest()['images'])
 def test_unresolved_targets_inactive_and_not_background(self):
  rows=read(OUT/'screening-v2.json')['images']
  for r in rows:
   full={a['id'] for a in r['original_annotations']};keep=set(r['retained_annotation_ids']);inactive=set(r['unresolved_annotation_ids'])
   self.assertFalse(keep&inactive);self.assertEqual(keep|inactive,full);self.assertFalse(r['human_verified'])
   self.assertTrue(r['unresolved_regions_must_not_be_background'])
   for sc in r['derived_training_scope'].values():
    self.assertFalse(sc['background_or_unmatched_negative_allowed'])
    if sc['retained_ids'] and len(sc['retained_ids'])<len(sc['native_source_ids']):self.assertTrue(sc['derived_positive_nonexhaustive'])
 def test_quarantine_cannot_enter_combined_readiness(self):
  rows=read(OUT/'screening-v2.json')['images'];bad={r['image_id'] for r in rows if r['status']=='quarantined'}
  ready=read(OUT/'combined-training-readiness.json')['source_rows'];self.assertFalse(bad&{r['image_id'] for r in ready})
  selected={r['image_id']:set(r['retained_annotation_ids']) for r in rows if r['status']=='provisional_source_candidate'}
  for r in ready:
   if r['source']=='phase8c_LVIS':self.assertEqual({a['original_annotation']['id'] for a in r['resolved_targets']},selected[r['image_id']])
 def test_exact_identity_hash_group_split_isolation_and_no_doublecount(self):
  d=read(OUT/'combined-training-readiness.json');rows=d['source_rows']
  for key in ('source_photo_id','sha256','image_id'):
   self.assertEqual(len({r[key] for r in rows}),len(rows))
  for key in ('source_photo_id','sha256','known_source_component'):
   self.assertFalse({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='development'})
  self.assertTrue(d['existing33_LVIS_annotations_inactive']);self.assertFalse(d['training_approved'])
 def test_no_new_Cabinet_units_from_door_and_front_annotations(self):
  rows=read(OUT/'eligible-candidates-v1.json')['images']
  self.assertFalse(any(a['category_id']==181 and a['id'] in r['retained_annotation_ids'] for r in rows for a in r['original_annotations']))
 def test_source_scope_and_source_geometry_not_rewritten(self):
  source={r['coco_image_id']:r for r in manifest()['images']}
  for r in read(OUT/'screening-v2.json')['images']:
   self.assertEqual(r['source_image_scope'],source[r['coco_image_id']]['source_image_scope'])
   self.assertEqual(r['original_annotations'],source[r['coco_image_id']]['original_furniture_annotations'])

if __name__=='__main__':unittest.main()
