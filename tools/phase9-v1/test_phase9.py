import copy
import math
import unittest
from core import CLASSES
from fusion import map_box,internal_edge,consensus,nms,method,views
from metrics import evaluate,curve
from crossview import compare_group,combine_camera_boxes

def pred(box=(.1,.1,.4,.4),label='chair',score=.8,**kw):return dict(box=list(box),label=label,score=score,**kw)
def row(exhaustive=True):return dict(image_id='fixture',phase9_partition='exposed',coverage={c:c=='chair' for c in CLASSES},fp_coverage={c:c=='chair' and exhaustive for c in CLASSES},annotations=[dict(label='chair',box=[.1,.1,.4,.4],ignore=False)])

class Coordinates(unittest.TestCase):
 def test_full_resize_normalized_coordinates_invariant(self):self.assertEqual(map_box([.1,.2,.3,.4],[0,0,1,1]),[.1,.2,.3,.4])
 def test_flip_inversion_and_double_flip(self):
  b=[.1,.2,.3,.4];r=map_box(b,[0,0,1,1],True);self.assertAlmostEqual(r[0],.7);self.assertAlmostEqual(r[2],.9)
  for a,c in zip(map_box(r,[0,0,1,1],True),b):self.assertAlmostEqual(a,c)
 def test_tile_offset_not_full_frame_stretch(self):self.assertEqual(map_box([0,0,1,1],[.375,.375,1,1]),[.375,.375,1,1])
 def test_actual_fractional_crop_region(self):
  b=map_box([.2,.3,.8,.9],[101/501,0,401/501,1]);self.assertAlmostEqual(b[0],161/501)
 def test_internal_tile_edges_and_true_photo_border_distinct(self):
  self.assertTrue(internal_edge([0,.2,.7,.8],[.375,0,1,.625]));self.assertFalse(internal_edge([0,.2,.7,.8],[0,0,.625,.625]))
 def test_bad_coordinates_rejected(self):
  with self.assertRaises(ValueError):map_box([float('nan'),0,1,1],[0,0,1,1])
 def test_fixed_bound_views_genuine_tensor_sizes(self):
  v=views();self.assertEqual(len(v),8);self.assertEqual(sorted({x['size'] for x in v}),[768,960,1152]);self.assertEqual(sum(x['name'].startswith('tile') for x in v),4)

class Fusion(unittest.TestCase):
 def test_duplicate_same_view_cannot_manufacture_agreement(self):self.assertFalse(consensus({'a':[pred(),pred()]})[0])
 def test_cross_category_not_merged(self):self.assertFalse(consensus({'a':[pred()],'b':[pred(label='table')]})[0])
 def test_weighted_box_math_and_no_invented_confidence(self):
  out,_=consensus({'a':[pred(score=.8)],'b':[pred([.12,.1,.42,.4],score=.4)]},weighted=True)
  self.assertEqual(len(out),1);self.assertAlmostEqual(out[0]['box'][0],(.1*.8+.12*.4)/1.2);self.assertAlmostEqual(out[0]['score'],.6)
 def test_consensus_min_score_not_probability_boost(self):self.assertEqual(consensus({'a':[pred(score=.8)],'b':[pred(score=.4)]})[0][0]['score'],.4)
 def test_adjacent_distinct_objects_not_blindly_suppressed(self):
  a=pred([.1,.1,.4,.4]);b=pred([.42,.1,.72,.4]);out,_=consensus({'a':[a,b],'b':[copy.deepcopy(a),copy.deepcopy(b)]},weighted=True);self.assertEqual(len(out),2)
 def test_complete_link_does_not_chain_drift(self):
  a=pred([.1,.1,.4,.4]);b=pred([.17,.1,.47,.4]);c=pred([.24,.1,.54,.4]);out,_=consensus({'a':[a],'b':[b],'c':[c]},threshold=.55);self.assertEqual(len(out),1);self.assertEqual(out[0]['support_count'],2)
 def test_tile_fragments_not_added(self):
  cache={'full960':[]};cache.update({f'tile{i}':[] for i in range(4)});cache['tile0']=[pred(internal_tile_edge=True)];self.assertFalse(method('tiled_guarded',cache)[0])
 def test_source_predictions_not_mutated(self):
  d={'a':[pred()],'b':[pred()]};old=copy.deepcopy(d);consensus(d,weighted=True);self.assertEqual(d,old)

class Metrics(unittest.TestCase):
 def test_known_positive_and_duplicate_counted_once(self):
  p=[pred(),pred(score=.7)];m=evaluate([row()],{'fixture':p});self.assertEqual((m['tp'],m['fp'],m['fn'],m['duplicate_fp']),(1,1,0,1))
 def test_unknown_class_not_false_positive(self):
  m=evaluate([row()],{'fixture':[pred(),pred(label='lamp')]});self.assertEqual(m['fp'],0)
 def test_nonexhaustive_unmatched_not_false_positive(self):
  m=evaluate([row(False)],{'fixture':[pred(),pred([.6,.6,.8,.8])]});self.assertEqual(m['fp'],0);self.assertEqual(m['tp'],1)
 def test_exhaustive_native_unmatched_is_false_positive(self):
  m=evaluate([row()],{'fixture':[pred(),pred([.6,.6,.8,.8])]});self.assertEqual(m['fp'],1)
 def test_suppress_everything_cannot_fake_success(self):
  m=evaluate([row()],{'fixture':[]});self.assertIsNone(m['precision']);self.assertEqual(m['recall'],0);self.assertEqual(m['f1'],0)
 def test_localization_and_AP75_differ(self):
  m=evaluate([row()],{'fixture':[pred([.15,.1,.45,.4])]});self.assertEqual(m['ap50'],1);self.assertEqual(m['ap75'],0)
 def test_frozen_metric_partition_denied(self):
  r=row();r['phase9_partition']='frozen'
  with self.assertRaises(ValueError):evaluate([r],{'fixture':[]})
 def test_group_regions_ignored(self):
  r=row();r['annotations'].append(dict(label='chair',box=[.6,.6,.8,.8],ignore=True));m=evaluate([r],{'fixture':[pred(),pred([.6,.6,.8,.8])]});self.assertEqual(m['fp'],0)
 def test_PR_thresholds_group_equal_scores_into_attainable_points(self):
  events=[dict(score=.5,image_id='a',index=0,label='chair',outcome='tp'),dict(score=.5,image_id='a',index=1,label='chair',outcome='fp')]
  points=curve(events,1);self.assertEqual(len(points),1);self.assertEqual(points[0]['precision'],.5)

class CrossView(unittest.TestCase):
 def test_creator_or_product_ID_not_physical_room_proof(self):
  with self.assertRaises(ValueError):compare_group([],{'creator':'same'})
 def test_distinct_camera_coordinates_never_fused(self):
  with self.assertRaises(ValueError):combine_camera_boxes([],[])
 def test_verified_views_kept_separate_and_no_unseen_objects_invented(self):
  out=compare_group([dict(view_id='one',image_sha256='a',predictions=[pred()]),dict(view_id='two',image_sha256='b',predictions=[])],dict(same_physical_scene_verified=True,provenance='fixture-only'))
  self.assertFalse(out['boxes_combined_across_views']);self.assertFalse(out['object_identity_consistency_evaluated']);self.assertTrue(out['category_absence_is_not_a_missed_object'])

if __name__=='__main__':unittest.main()
