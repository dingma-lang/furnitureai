import copy
import unittest
from preflight import *

class ReviewTests(unittest.TestCase):
    def test_all_reviews_bound_to_sources_and_partitions(self):
        r=read(RAW3/'review-validation.json');self.assertEqual(r['reviewed_items'],65);self.assertEqual(r['unique_photo_hashes_checked'],17);self.assertEqual(r['prediction_boxes_checked'],56)
        self.assertTrue(all(x['source_photo_sha256_verified'] and not x['negative_supervision_accepted'] for x in r['records']))
    def test_human_scope_decisions_not_turned_into_absence(self):
        m=read(RAW3/'supervision-manifest-v1.json');self.assertEqual(len(m['rows']),59);self.assertEqual(m['accepted_negative_evidence'],[])
        self.assertEqual(sum(r['human_scope_decision']=='missing_or_ambiguous_chairs' for r in m['rows']),9)
        self.assertTrue(all(r['chair_negative_regions']==[] for r in m['rows']))
    def test_taxonomy_is_explicit_without_silent_native_remapping(self):
        t=read(RAW3/'taxonomy.json');self.assertTrue(t['policy']['user_confirmed']);self.assertIn('bar_stools',t['policy']['project_chair_includes']);self.assertIn('step_stools',t['policy']['project_chair_excludes'])
        names={c['id']:c['name'] for c in t['native_lvis_definitions']};self.assertEqual(names[232],'chair');self.assertEqual(names[1018],'stool');self.assertFalse(t['policy']['stool_positive_annotations_automatically_activated'])
    def test_no_evaluation_images_enter_new_manifest(self):
        m=read(RAW3/'supervision-manifest-v1.json');old=manifest();ids={r['image_id'] for r in old['rows'] if optimizer_member(r)}
        self.assertEqual({r['image_id'] for r in m['rows']},ids)
        self.assertTrue(all(r['internal_partition']=='optimization' for r in m['rows']))
    def test_empty_and_unconfirmed_region_template_cannot_activate(self):
        import torch
        template=read(RAW3/'region-evidence-template.json')['records'][0];row=next(r for r in manifest()['rows'] if r['image_id']==template['image_id'])
        self.assertRaises(PermissionError,validate_region_record,template,row)
    def test_stool_policy_confirmation_is_mandatory(self):
        from trusted_loss import evidence_valid
        r=next(r for r in manifest()['rows'] if optimizer_member(r))
        e=dict({k:r[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')},category='chair',native_category=next(a['native_id'] for a in r['annotations'] if a['label']=='chair') if any(a['label']=='chair' for a in r['annotations']) else '/m/01mzpv',
            kind='negative_region',regions=[[0.,0.,.01,.01]],potential_unannotated_chair_regions=[],native_annotation_digest=digest(r['annotations']),human_reviewed=True,confirmed=True,
            reviewer='CPU SYNTHETIC FIXTURE',review_date='2026-11-19',evidence='Synthetic absence region fixture only',trust_basis='human_category_scope_review',taxonomy_sha256=digest(TAXONOMY),review_covers_seating_stools=False)
        self.assertRaises(PermissionError,validate_region_record,e,r)
        # Use the exact native dataset mapping for this fixture.
        dataset=r['annotations'][0]['native_dataset'];e['native_category']={'OpenImages':'/m/01mzpv','LVIS':232,'COCO':62}[dataset]
        e['review_covers_seating_stools']=True;self.assertTrue(validate_region_record(e,r))
        e['internal_partition']='inner_development';self.assertRaises(ValueError,validate_region_record,e,r)
    def test_semantic_conflict_flags_preserve_original_decision(self):
        d=dict(kind='prediction_diagnostic',decision='background_non_furniture',evidence='partially a child seating chair');original=copy.deepcopy(d)
        self.assertTrue(semantic_flags(d));self.assertEqual(d,original)
        self.assertTrue(semantic_flags(dict(kind='prediction_diagnostic',decision='other_furniture',evidence='a plant')))
    def test_fixture_masks_exact_and_training_gate_closed(self):
        m=read(RAW3/'mask-difference-audit.json');self.assertTrue(m['all_target_and_mask_tensors_bitwise_equal']);self.assertFalse(m['meaningful_AB_difference']);self.assertEqual(m['matched_native_positives'],182);self.assertEqual(m['corrected_added_chair_negative_terms'],0)

if __name__=='__main__':unittest.main()
