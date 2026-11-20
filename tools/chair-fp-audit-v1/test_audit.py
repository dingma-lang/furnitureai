import copy
import math
import unittest
from audit import *

class Tests(unittest.TestCase):
    def test_bin_boundaries_and_invalid_scores(self):
        self.assertEqual(score_bin(.15),'0.15–0.3');self.assertEqual(score_bin(.3),'0.3–0.5');self.assertEqual(score_bin(1),'0.9–1')
        self.assertEqual(area_bin(.01),'1–5%');self.assertEqual(area_bin(.05),'5–20%');self.assertEqual(area_bin(.2),'>=20%')
        self.assertRaises(ValueError,score_bin,.14)
    def test_geometry_is_not_semantic_proof(self):
        row=dict(width=100,height=100,annotations=[dict(label='chair',ignore=False,box=[.1,.1,.6,.9])])
        x=geometry(dict(box=[.2,.2,.25,.25]),row)
        self.assertEqual(x['primary_geometry_bucket'],'contained_chair_fragment');self.assertFalse(x['semantic_error_confirmed'])
        self.assertEqual(geometry(dict(box=[.7,.1,.8,.2]),row)['primary_geometry_bucket'],'no_chair_overlap_unresolved')
        self.assertEqual(geometry(dict(box=[0,0,1,1]),row)['primary_geometry_bucket'],'broad_box_covering_chair')
    def test_duplicate_precedes_geometry_but_does_not_confirm_error(self):
        r=dict(width=100,height=100,annotations=[dict(label='chair',ignore=False,box=[.1,.1,.8,.8])])
        x=geometry(dict(box=[.1,.1,.8,.8]),r,dict(duplicate=True));self.assertEqual(x['primary_geometry_bucket'],'protocol_duplicate');self.assertFalse(x['semantic_error_confirmed'])
    def test_actual_saved_counter_reconciliation(self):
        a=read(OUT/'analysis.json')
        self.assertEqual(sum(p['chair_FP'] for p in a['photographs']),1094);self.assertEqual(sum(a['chair_primary_geometry_counts'].values()),1094)
        self.assertEqual(sum(a['category_aggregates']['chair']['confidence_bins'].values()),1094)
        self.assertEqual(sum(a['category_aggregates']['chair']['area_bins'].values()),1094)
        self.assertEqual(sum(p['chair_FP']>0 for p in a['photographs']),4);self.assertEqual(a['chair_unscored_operating'],3252)
    def test_missing_historical_predictions_stay_unavailable(self):
        a=read(OUT/'analysis.json');self.assertEqual(len(a['available_per_image_models']),1);self.assertEqual(len(a['unavailable_per_image_models']),2)
        self.assertTrue((OUT/'index.html').exists());self.assertIn('No substitute predictions',(OUT/'index.html').read_text())
    def test_sample_no_fake_human_confirmation(self):
        x=read(OUT/'review-sample.json')['records'];self.assertEqual(len(x),56)
        self.assertEqual(sum(r['status']=='fp' for r in x),48);self.assertEqual(sum(r['status']=='unscored_incomplete' for r in x),8)
        self.assertEqual(len({(r['image_id'],r['index']) for r in x}),56)
        import csv
        with (OUT/'review-template.csv').open() as f:
            for r in csv.DictReader(f):
                for k in ('reviewer','review_date','human_reviewed','decision','evidence'):self.assertEqual(r[k],'')
    def test_source_supervision_coefficients_and_mask_policy(self):
        old=(ROOT/'tools/phase5c-v1/adaptation.py').read_text();new=(ROOT/'tools/phase10a-v1/expanded.py').read_text()
        self.assertIn("row['coverage'][c]",old);self.assertIn("negatives.get(c,False)",new)
        self.assertIn('.25*target+.75*(1-target)',old.replace(' ',''));self.assertIn('.25*target+.75*(1-target)',new.replace(' ',''))
        self.assertIn('cls+5*loss_box+2*loss_giou',old.replace(' ',''));self.assertIn('cls+5*loss_box+2*loss_giou',new.replace(' ',''))
        self.assertNotIn('recall',new)
    def test_positive_only_shared_shift_gradient_direction_fixture(self):
        # Analytic gamma2 focal derivatives at logit0. No model or optimizer.
        positive=.25*(-.25*math.log(2)-.125)
        negative=.75*(.25*math.log(2)+.125)
        self.assertLess(positive,0);self.assertGreater(positive+3*negative,0)
    def test_no_model_or_execution_entry(self):
        import ast
        for name in ('audit.py','visualize.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    self.assertNotIn('torch',ast.unparse(node));self.assertNotIn('transformers',ast.unparse(node))

if __name__=='__main__':unittest.main()
