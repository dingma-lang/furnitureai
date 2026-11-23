"""CPU regression tests: source isolation, coordinates, evidence and intake gates."""
import copy,unittest
from unittest.mock import patch
import numpy as np
from commonE6 import *
from acceptanceE6 import validate,validate_manifest

class SourceSafetyTests(unittest.TestCase):
    def test_validation_and_unknown_ids_rejected(self):
        _,train,dev=context()
        for ident in [dev[0]['image_id'],'reserved-test-unknown']:
            with self.assertRaises(PermissionError):require_train(ident,train)

    def test_partition_and_source_group_are_both_enforced(self):
        _,train,_=context();r=dict(next(iter(train.values())))
        r['internal_partition']='internal_validation'
        with self.assertRaises(PermissionError):require_train(r['image_id'],{r['image_id']:r})

    def test_original_non_square_registration(self):
        # A640x480image padded only bottom to640 square. This square model box
        # is exactly the original-image bottom-right quadrant,without centering.
        a=registered([[.75,.5625,.5,.375]],640,480)
        np.testing.assert_allclose(a,[[.5,.5,1,1]],rtol=0,atol=1e-12)

    def test_padding_is_not_silently_clipped(self):
        a=registered([[.9,.9,.3,.3]],480,640)[0]
        self.assertGreater(a[2],1)
        self.assertGreater(a[3],1)

    def test_stool_and_furniture_parts_not_nonfurniture(self):
        for label in ('stool','bar stool','footstool','ottoman','chair'):
            self.assertEqual(semantic(dict(native_label=label,protect_furniture=False,part_or_semantics_unresolved=False)),'seating_or_ambiguous_seating')
        self.assertEqual(semantic(dict(native_label='drawer',protect_furniture=False,part_or_semantics_unresolved=True)),'furniture_fragment_or_ambiguous_part')

    def test_polygon_boundary_and_protected_overlap(self):
        sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
        from geometry6 import certify
        poly=[(0,0),(100,0),(100,100),(0,100)]
        self.assertTrue(certify([10,10,80,80],poly))
        self.assertFalse(certify([0,0,100,100],poly))
        self.assertTrue(overlap([.1,.1,.8,.8],[.7,.7,.9,.9]))
        self.assertFalse(overlap([.1,.1,.2,.2],[.3,.3,.4,.4]))

class CandidateIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=read(DATA/'training-candidate-manifest-v1.json')
        cls.rows=read(DATA/'candidate-evidence-v1.json')['records']

    def test_no_tiers_are_silently_activated(self):
        self.assertEqual(self.manifest['sets']['A_trusted_new_training_eligible_hard_negatives'],[])
        self.assertFalse(self.manifest['training_authorized'])
        self.assertTrue(all(not r['human_reviewed'] and not r['training_eligible'] and not r['training_activated'] for r in self.rows))

    def test_all_candidates_have_optimization_identity_and_lineage(self):
        _,train,dev=context()
        lineage={p['image_id']:p for p in read(DATA/'source-lineage-v1.json')['photos']}
        for r in self.rows:
            row=require_train(r['image_id'],train)
            self.assertEqual(r['source_sha256'],row['sha256'])
            if r['source_annotation_key']:
                self.assertIn(r['source_annotation_key'],{a['annotation_key'] for a in lineage[r['image_id']]['annotations']})

    def test_tiers_form_disjoint_complete_partition(self):
        grouped=list(self.manifest['sets'].values());allids=[v for group in grouped for v in group]
        self.assertEqual(len(allids),len(set(allids)))
        self.assertEqual(set(allids),{r['candidate_id'] for r in self.rows})
        self.assertEqual([len(g) for g in grouped],[0,183,9781])

    def test_low_original_confidence_and_bbox_context_are_not_negatives(self):
        candidates=[r for r in self.rows if r['evidence_tier']=='B' and any(o['original_score']<.15 for o in r['observations'])]
        self.assertTrue(candidates)
        self.assertTrue(all(not r['training_eligible'] and not r['source_geometry_support'] for r in candidates))

    def test_visual_assessments_are_not_human_confirmations(self):
        rows=read(DATA/'visual-observations.json')['records']
        self.assertEqual(len(rows),24)
        self.assertTrue(all(r['assistant_actual_visual_inspection'] and not r['human_confirmed'] and not r['human_reviewed'] and not r['negative_label_approved'] for r in rows))
        self.assertIn('apparent_real_seating',{r['finding'] for r in rows})

    def test_cache_alias_expansion_adds_no_numerical_snapshots(self):
        a=read(DATA/'query-index-v2.json');b=read(DATA/'query-index-v3.json')
        self.assertEqual({r['snapshot_id'] for r in a['records']},{r['snapshot_id'] for r in b['records']})
        self.assertEqual(b['raw_source_bound_files'],617)
        self.assertEqual(len(b['records']),427)

    def test_unindexed_bodies_are_all_internal_validation(self):
        _,_,dev=context();r=read(DATA/'query-index-v3.json')['unindexed_non_evaluation_NPZ']
        self.assertEqual({Path(v['path']).stem for v in r},{v['image_id'] for v in dev})
        self.assertEqual(len(r),19)

    def test_e5_did_not_reuse_c11_outputs(self):
        r=read(REPORT/'evaluation-verification.json')
        self.assertEqual(r['identical_prediction_arrays'],0)
        self.assertEqual(r['identical_prediction_files'],0)
        self.assertNotEqual(r['E5_sha256'],r['C11_sha256'])
        self.assertFalse(r['AP50_exactly_equal'])
        self.assertTrue(r['identical_fixed_point_counts'])

    def test_no_gpu_approved_in_prepared_request(self):
        r=read(REPORT/'inference-request-disabled.json');_,train,_=context()
        self.assertFalse(r['approved']);self.assertFalse(r['enabled'])
        self.assertEqual(set(r['exact_photo_ids']),set(train))
        self.assertEqual(r['maximum_detector_forwards'],59)
        self.assertEqual(r['optimizer_updates'],0)

    def test_conditional_rights_ids_are_new_and_not_paused(self):
        records=read(DATA/'conditional-rights-candidates-v1.json')['records']
        ids={r['image_id'] for r in records}
        self.assertEqual(len(ids),64)
        self.assertFalse(ids&set(read(DATA/'independent-exclusion-registry.json')['image_ids']))
        for rel in ('data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json','data/lvis/rights-audits/phase7-v1/license-and-attribution-ledger.json','data/lvis/rights-audits/phase8a-v1/license-attribution-ledger.json'):
            old=read(ROOT/rel)['records']
            old_ids={r.get('image_id') or 'coco2017:'+str(r.get('coco_image_id',r.get('lvis_image_id'))).zfill(12) for r in old}
            self.assertFalse(ids&old_ids)
        self.assertTrue(all(r['rights_status']=='individually_unverified' and r['creator'] is None and not r['photo_download_authorized'] for r in records))

class IndependentIntakeTests(unittest.TestCase):
    def setUp(self):
        self.registry={k:[] for k in ('image_ids','sha256','source_photo_ids','source_groups')}
        self.row=dict(image_id='fixture:new',source_photo_id='fixture:photo',source_group='fixture:creator',capture_session_group='fixture:capture',path='data/fixture.jpg',sha256='a'*64,
          rights_evidence_path='data/fixture-rights.json',rights_evidence_sha256='b'*64,license_url='https://creativecommons.org/licenses/by/4.0/',license_version='CC BY4.0',creator='Synthetic test fixture',source_url='https://example.invalid/fixture',rights_verified_at='2026-11-22T12:00:00Z',
          role='independent_acceptance',source_group_independence_verified=True,source_group_evidence='Unit test fixture only',rights_verified=True,
          category_complete={c:True for c in CLASSES},Chair_includes_seating_and_bar_stools=True,whole_object_taxonomy_confirmed=True,unresolved_objects=[],annotations=[],verified_visible_instance_counts={c:0 for c in CLASSES},
          review=dict(human_reviewed=True,blind_to_model_predictions=True,reviewer='Synthetic fixture only',date='2026-11-22',evidence='Test fixture;no actual review was performed'))

    def test_synthetic_fixture_is_explicitly_not_verified(self):
        self.assertTrue(validate(self.row,self.registry,verify_files=False)['synthetic_test_only'])

    def test_all_categories_including_absence_must_be_complete(self):
        for category in CLASSES:
            r=copy.deepcopy(self.row);r['category_complete'][category]=False
            with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_no_automatic_human_review(self):
        for key,value in [('human_reviewed',False),('blind_to_model_predictions',False),('reviewer',''),('evidence','')]:
            r=copy.deepcopy(self.row);r['review'][key]=value
            with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_historical_identity_or_group_rejected(self):
        for field,key in [('image_id','image_ids'),('sha256','sha256'),('source_photo_id','source_photo_ids'),('source_group','source_groups'),('capture_session_group','source_groups')]:
            reg=copy.deepcopy(self.registry);reg[key]=[self.row[field].upper()]
            with self.assertRaises(PermissionError):validate(self.row,reg,False)

    def test_unresolved_taxonomy_and_stool_policy_rejected(self):
        for field,value in [('unresolved_objects',['ambiguous lighting']),('Chair_includes_seating_and_bar_stools',False),('whole_object_taxonomy_confirmed',False)]:
            r=copy.deepcopy(self.row);r[field]=value
            with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_invalid_boxes_and_mismatched_counts(self):
        for box in ([0,0,2,1],[.5,0,.4,.3],[0,0,float('nan'),.5]):
            r=copy.deepcopy(self.row);r['annotations']=[dict(label='chair',box=box,ignore=False)]
            with self.assertRaises(ValueError):validate(r,self.registry,False)
        r=copy.deepcopy(self.row);r['verified_visible_instance_counts']['chair']=1
        with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_repeated_creator_groups_rejected(self):
        other=copy.deepcopy(self.row);other.update(image_id='fixture:second',sha256='c'*64,source_photo_id='fixture:second-photo',capture_session_group='fixture:second-capture')
        with self.assertRaises(ValueError):validate_manifest([self.row,other],self.registry,False)

    def test_rights_and_files_cannot_pass_on_booleans_alone(self):
        with self.assertRaises(FileNotFoundError):validate(self.row,self.registry,True)
        r=copy.deepcopy(self.row);r['rights_verified']=False
        with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_restricted_or_lookalike_licence_urls_rejected(self):
        for url in ('https://creativecommons.org/licenses/by-nc/4.0/','https://creativecommons.org.evil.invalid/licenses/by/4.0/','http://creativecommons.org/licenses/by/4.0/'):
            r=copy.deepcopy(self.row);r['license_url']=url
            with self.assertRaises(ValueError):validate(r,self.registry,False)

    def test_empty_real_intake_stays_zero(self):
        self.assertEqual(validate_manifest([],read(DATA/'independent-exclusion-registry.json'))['eligible_photographs'],0)

if __name__=='__main__':unittest.main()
