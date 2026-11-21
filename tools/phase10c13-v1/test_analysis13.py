"""Synthetic CPU tests for uncertainty counts, including unknown predictions."""
import copy,unittest
from common13 import CLASSES,metric_module
from analyze13 import paired_uncertainty


class UncertaintyTests(unittest.TestCase):
    def fixture(self):
        rows=[];ps={}
        for i in (1,2):
            ident='SYNTHETIC-'+str(i);coverage={c:c=='chair' for c in CLASSES};fp={c:False for c in CLASSES}
            rows.append(dict(image_id=ident,identity_group='SYNTHETIC-GROUP-'+str(i),coverage=coverage,fp_coverage=fp,annotations=[dict(label='chair',ignore=False,box=[.1,.1,.3,.3])]))
            ps[ident]=[dict(label='chair',score=.9,box=[.1,.1,.3,.3]),dict(label='chair',score=.8,box=[.7,.7,.9,.9])]
        keys=('original','corrected_B','frozen_calibration','projection_regularized');predictions={k:copy.deepcopy(ps) for k in keys};metrics={k:dict(tp=2,fp=0,fn=0) for k in keys}
        return rows,predictions,metrics,metric_module()

    def test_unknown_unmatched_predictions_do_not_become_false_positives(self):
        args=self.fixture();r=paired_uncertainty(*args)
        for item in r['comparisons'].values():self.assertEqual(item['precision_delta_95_percentile_interval'],[0.,0.]);self.assertEqual(item['recall_delta_95_percentile_interval'],[0.,0.])

    def test_undefined_precision_is_not_fabricated_and_recall_still_reported(self):
        rows,pred,metrics,m=self.fixture();pred['projection_regularized']={r['image_id']:[] for r in rows};metrics['projection_regularized']=dict(tp=0,fp=0,fn=2);r=paired_uncertainty(rows,pred,metrics,m)
        self.assertIsNone(r['comparisons']['original']['precision_delta_95_percentile_interval']);self.assertEqual(r['comparisons']['original']['recall_delta_95_percentile_interval'],[-1.,-1.])

    def test_mismatched_metric_counts_rejected(self):
        rows,pred,metrics,m=self.fixture();metrics['original']['fp']=2
        with self.assertRaisesRegex(ValueError,'frozen metric'):paired_uncertainty(rows,pred,metrics,m)

    def test_bootstrap_is_reproducible(self):
        args=self.fixture();self.assertEqual(paired_uncertainty(*args),paired_uncertainty(*args))


if __name__=='__main__':unittest.main()
