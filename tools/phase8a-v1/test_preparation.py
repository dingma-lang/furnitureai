import copy
import importlib.util
import sys
import unittest
from unittest.mock import patch
from prep_core import *
from ontology import mapping,native_scope,loss_mask,masked_binary_loss,pixel_xyxy_view
from expansion import select

class PolicyTests(unittest.TestCase):
 def test_native_table_and_desk_remain_distinct(self):
  self.assertEqual(mapping('COCO',67)['comparison_group'],'table');self.assertEqual(mapping('LVIS',361)['comparison_group'],'desk')
  self.assertFalse(mapping('COCO',67)['project_positive_allowed']);self.assertFalse(mapping('LVIS',1050)['project_positive_allowed'])
 def test_cabinetry_cupboard_drawer_not_silently_equivalent(self):
  self.assertFalse(mapping('OpenImages','/m/01s105')['project_positive_allowed']);self.assertIsNone(mapping('LVIS',346)['comparison_group']);self.assertIsNone(mapping('LVIS',390)['comparison_group'])
 def test_no_generic_lamp_is_proved_floor_lamp(self):
  self.assertFalse(mapping('LVIS',626)['floor_lamp_verified']);self.assertFalse(mapping('LVIS',1051)['floor_lamp_verified']);self.assertIsNone(mapping('LVIS',627)['comparison_group']);self.assertIsNone(mapping('LVIS',628)['comparison_group'])
 def test_missing_COCO_label_is_unknown(self):self.assertEqual(native_scope('COCO',62,False,{}),'unknown')
 def test_LVIS_nonexhaustive_scope(self):
  self.assertEqual(native_scope('LVIS',181,True,{'not_exhaustive_category_ids':[181]}),'positive_nonexhaustive')
  self.assertFalse(loss_mask('LVIS',181,'positive_nonexhaustive',matched=False)['negative_class'])
 def test_only_explicit_native_negatives_enable_negative(self):
  self.assertTrue(loss_mask('LVIS',181,'explicit_native_negative')['negative_class'])
  self.assertFalse(loss_mask('LVIS',181,'positive_source_exhaustive')['negative_class'])
 def test_native_negative_not_project_or_other_subtype_negative(self):
  self.assertEqual(native_scope('LVIS',1050,False,{'neg_category_ids':[367]}),'unknown')
  self.assertFalse(loss_mask('LVIS',181,'explicit_native_negative',head_space='project')['negative_class'])
 def test_exhaustive_native_negatives_require_all_explicit_gates(self):
  self.assertTrue(loss_mask('LVIS',232,'positive_source_exhaustive',allow_exhaustive_native_negatives=True,all_native_targets_retained=True)['negative_class'])
  self.assertFalse(loss_mask('LVIS',232,'positive_source_exhaustive',allow_exhaustive_native_negatives=True,all_native_targets_retained=False)['negative_class'])
  self.assertFalse(loss_mask('LVIS',232,'positive_nonexhaustive',allow_exhaustive_native_negatives=True,all_native_targets_retained=True)['negative_class'])
  self.assertFalse(loss_mask('LVIS',232,'positive_source_exhaustive',head_space='project',allow_exhaustive_native_negatives=True,all_native_targets_retained=True)['negative_class'])
 def test_scope_contradictions_rejected(self):
  with self.assertRaises(ValueError):native_scope('LVIS',181,True,{'neg_category_ids':[181]})
  with self.assertRaises(ValueError):native_scope('OpenImages','/m/01mzpv',True,{'native_image_level_labels':{'/m/01mzpv':0}})
 def test_positive_native_mask_and_box_without_background(self):
  plan=loss_mask('LVIS',232,'positive_source_exhaustive',matched=True,has_mask=True)
  self.assertTrue(plan['positive_class'] and plan['box'] and plan['mask']);self.assertFalse(plan['negative_class'] or plan['background_objectness'])
 def test_ambiguous_ignored_and_group_regions_have_no_loss(self):
  for flags in ({'ignored':True},{'group_of':True},{'unit_ambiguous':True}):
   plan=loss_mask('OpenImages','/m/01mzpv','positive_source_coverage',matched=True,**flags)
   self.assertFalse(any(plan[k] for k in ['positive_class','negative_class','box','mask','background_objectness']))
 def test_unknown_query_predictions_do_not_change_loss(self):
  a=masked_binary_loss([.7],[1],[True]);b=masked_binary_loss([.7,100,-100],[1,0,1],[True,False,False]);self.assertEqual(a,b);self.assertEqual(masked_binary_loss([100],[0],[False]),0)
 def test_native_geometry_is_never_changed_by_conversion(self):
  box=[.1,.2,.5,.8];old=box.copy();self.assertEqual(pixel_xyxy_view('OpenImages',box,100,200),[10,40,50,160]);self.assertEqual(box,old)
  self.assertEqual(pixel_xyxy_view('LVIS',[10,40,40,120],100,200),[10,40,50,160])
 def test_no_clipping_or_axis_guessing(self):
  with self.assertRaises(ValueError):pixel_xyxy_view('COCO',[-1,0,2,2],100,100)
 def test_stable_OI_ids_preserve_row_occurrences(self):
  a=dict(source_mid='/m/03ssj5',box=[0,0,1,1],flags={});self.assertNotEqual(stable_box_id('image',a,0),stable_box_id('image',a,1))
 def test_selector_reproducible_does_not_weaken_shortfall(self):
  pool=[dict(coco_image_id=i,known_source_group=str(i),eligible_anchor_ids=[181],core_cooccurrence_count=2) for i in range(3)]
  a,done=select(pool,{181:8,626:8});b,_=select(list(reversed(pool)),{181:8,626:8});self.assertEqual(a,b);self.assertEqual(len(a),3);self.assertEqual(done[626],0)

class ActualAuditTests(unittest.TestCase):
 def test_original_coverage_totals_and_independent_photo_counts(self):
  d=read(OUT/'current-coverage.json');t=d['split_totals']['train'];v=d['split_totals']['fresh_development'];self.assertEqual((t['images'],t['boxes'],t['masks']),(114,361,285));self.assertEqual((v['images'],v['boxes'],v['masks']),(23,99,76));self.assertEqual(len({r['source_photo_id'] for r in d['rows']}),137)
 def test_annotation_audit_identifiers_are_unique(self):
  d=read(OUT/'current-coverage.json');ids=[a['stable_audit_annotation_id'] for r in d['rows'] for a in r['resolved_targets']];self.assertEqual(len(ids),460);self.assertEqual(len(set(ids)),460)
 def test_weak_category_counts(self):
  d=read(OUT/'current-coverage.json')['split_totals']['fresh_development']['per_category'];self.assertEqual(d['cabinet']['images'],0);self.assertEqual(d['lamp']['images'],1);self.assertEqual(d['bed']['images'],2)
 def test_all33_inactive_and_not_fake_human_review(self):
  d=read(OUT/'existing33-registration-v2.json');self.assertEqual(len(d['findings']),33);self.assertTrue(all(not r['active'] and not r['independent_semantic_verification_complete'] for r in d['findings']));self.assertEqual(d['annotations_activated'],0)
 def test44_new_IDs_exclude_paused_and_all_prior_cohorts(self):
  rows=read(OUT/'targeted-rights-candidates-v1.json')['images'];prior={r['coco_image_id'] for r in read(META/'proposed-photo-rights-audit-v1.json')['images']};self.assertEqual(len(rows),44);self.assertFalse(prior&{r['coco_image_id'] for r in rows});self.assertTrue(all(r['photo_rights_status']=='individually_unverified' for r in rows))
 def test_no_synthetic_or_real_evaluation_reassignment(self):
  s=read(OUT/'approval-C3-conditional-render-pilot.json');self.assertEqual(s['synthetic_partition'],'train_only');self.assertFalse(s['execution_ready']);self.assertFalse(s['approved']);self.assertEqual(s['views_max'],100)
 def test_exact_asset_bundle_and_runtime_installation_not_approved(self):
  s=read(OUT/'approval-C1-exact-CC0-asset-bundle.json');self.assertEqual(s['expected_response_body_bytes'],11026401);self.assertEqual(len(s['unique_objects']),33);self.assertFalse(s['Blender_installation']);self.assertFalse(read(OUT/'approval-C2-runtime-metadata-preflight.json')['installations'])
 def test_synthetic_export_fixture_not_renderer(self):
  import numpy as np
  sys.path.insert(0,str(ROOT/'tools/phase7-v1'));spec=importlib.util.spec_from_file_location('phase8a_synthetic_test',ROOT/'tools/phase7-v1/synthetic.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  r=m.visible_annotations(np.array([[0,1,1],[0,1,0]],dtype=np.int16),{1:{'category':'chair','asset_id':'fixture'}})[0]
  self.assertEqual(r['bbox_xywh'],[1,0,2,2]);self.assertEqual(r['area'],3);self.assertEqual(sum(r['segmentation']['counts']),6)
 def test_offline_utilities_never_contact_network(self):
  with patch('socket.create_connection',side_effect=AssertionError('Network forbidden')),patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
   self.assertEqual(native_scope('COCO',62,False,{}),'unknown');self.assertFalse(mapping('LVIS',628)['project_positive_allowed'])

if __name__=='__main__':unittest.main()
