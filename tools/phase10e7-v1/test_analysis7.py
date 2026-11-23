"""Post-cache CPU checks;no detector construction or additional inference."""
import unittest
import numpy as np
from common7 import *
from analyze7 import evidence_for_query

class CacheIntegrityTests(unittest.TestCase):
    def test_actual_forward_and_parameter_fingerprints(self):
        d=read(RUN/'complete.json');s=d['resources']
        self.assertEqual([s[k] for k in ('attempted_forwards','actual_forwards','completed_forwards','cached_photos')],[59]*4)
        self.assertEqual(s['optimizer_steps'],0);self.assertEqual(s['backward_calls'],0)
        self.assertLess(s['GPU_runtime_seconds'],180)
        self.assertLessEqual(s['peak_allocated_VRAM_bytes'],LIMITS['VRAM_bytes'])
        self.assertEqual(d['model_before'],d['model_after'])

    def test_exact_raw_membership_source_hashes_and_arrays(self):
        _,_,train=inputs();r=read(RUN/'raw-complete.json')['records']
        self.assertEqual([x['image_id'] for x in r],list(train))
        for x in r:
            self.assertEqual(sha(RUN/x['file']),x['sha256'])
            self.assertEqual(x['source_sha256'],train[x['image_id']]['sha256'])
            with np.load(RUN/x['file'],allow_pickle=False) as z:
                expected=raw_values(z['logits'],z['square_cxcywh'],x['width'],x['height'])
                self.assertEqual(set(z.files),set(expected))
                for k,v in expected.items():np.testing.assert_array_equal(z[k],v)

    def test_thresholds_monotonic_all_seven_categories(self):
        s=read(REPORT/'query-summary.json')['thresholds']
        for category in CLASSES:
            for metric in ('raw_class_terms','raw_argmax_queries','postNMS_argmax'):
                v=[s[str(t)][metric][category] for t in (.15,.5,.8)]
                self.assertGreaterEqual(v[0],v[1]);self.assertGreaterEqual(v[1],v[2])

    def test_no_candidate_activated_or_human_evidence_fabricated(self):
        m=read(RUN/'analysis/candidate-manifest.json')
        self.assertEqual(m['training_eligible_new'],[]);self.assertEqual(m['new_labels_activated'],0)
        self.assertFalse(set(m['B_automatic_unverified'])&set(m['C_uncertain_protected_or_rejected']))
        obs=[json.loads(s) for s in (RUN/'analysis/high-confidence-Chair.jsonl').read_text().splitlines()]
        self.assertTrue(all(not o['training_eligible'] and not o['training_activated'] and not o['human_reviewed'] and not o['negative_label_created'] for o in obs))
        self.assertEqual(set(m['B_automatic_unverified'])|set(m['C_uncertain_protected_or_rejected']),{o['candidate_id'] for o in obs})

    def test_deduplication_does_not_add_repeated_states_as_examples(self):
        d=read(REPORT/'query-summary.json')['deduplication']
        self.assertEqual(d['combined_distinct_snapshots'],d['E6_distinct_snapshots']+d['E7_distinct_snapshots']-d['E7_exact_snapshot_repeats_of_E6'])
        self.assertEqual(d['high_Chair_current_unique_patch_hotspots'],d['high_Chair_patch_hotspots_previously_seen']+d['high_Chair_new_patch_hotspots'])
        self.assertEqual(d['unique_photo_hashes'],59);self.assertEqual(d['unique_source_photo_ids'],59)

class EvidenceGateTests(unittest.TestCase):
    def fixture(self):
        row=dict(image_id='synthetic_fixture_only',width=640,height=480,annotations=[dict(label='chair',ignore=False,box=[.2,.2,.6,.8])])
        return row,dict(annotations=[]),[dict(key='positive:fixture',box=[.2,.2,.6,.8])]

    def test_real_Chair_overlap_never_becomes_negative(self):
        row,photo,protected=self.fixture()
        r=evidence_for_query(1,[.2,.2,.6,.8],.99,.01,row,photo,protected,[],True,{})
        self.assertEqual(r['verified_Chair_IoU'],1)
        self.assertFalse(r['training_eligible']);self.assertFalse(r['negative_label_created'])

    def test_fragment_and_unknown_are_kept_inactive(self):
        row,photo,protected=self.fixture()
        r=evidence_for_query(1,[.3,.3,.4,.4],.99,.01,row,photo,protected,[],True,{})
        self.assertTrue(r['potential_fragment']);self.assertFalse(r['training_eligible'])
        r=evidence_for_query(2,[.7,.1,.8,.2],.99,.01,row,photo,protected,[],True,{})
        self.assertEqual(r['source_context'],'uncertain_or_unannotated_real_object');self.assertFalse(r['training_eligible'])

    def test_region_overlap_is_not_full_containment(self):
        row,photo,protected=self.fixture();legacy=[dict(image_id=row['image_id'],candidate_id='fixture-region',box=[.65,.1,.8,.2])]
        r=evidence_for_query(1,[.7,.15,.9,.25],.99,.01,row,photo,protected,legacy,True,{})
        self.assertEqual(r['existing_negative_region_overlap'],['fixture-region'])
        self.assertEqual(r['existing_negative_region_full_containment'],[])
        self.assertFalse(r['trusted_existing_region_high_query_coverage'])

if __name__=='__main__':unittest.main()
