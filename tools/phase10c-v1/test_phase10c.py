import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from c10 import *
import metric_cpu as cpu

class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'tools/phase9-v1'))
        import metrics,fusion
        cls.old=metrics;cls.fusion=fusion
        lock=read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')
        cls.rows=[dict(r,fp_coverage=r['coverage']) for r in lock['rows'] if r['split']=='development']
    def test_legacy_events_curves_and_ap_on_cached_exposed_predictions(self):
        # CPU cache replay only; no photograph/model loader and no frozen panel.
        for name in ('adapted-scale_nms','original-baseline'):
            p=ROOT/f'data/open-images/experiments/phase9-multiview-v1/evaluations/exposed/{name}.json'
            if not p.exists():
                if name=='original-baseline':p=ROOT/'data/open-images/experiments/phase9-multiview-v1/evaluations/exposed/original-baseline.json'
                if not p.exists():continue
            pred=read(p)['predictions']
            for c in CLASSES:
                for threshold,floor in ((.5,.001),(.75,.001),(.5,.15)):
                    a=self.old.events(self.rows,pred,c,threshold,floor);b=cpu.events(self.rows,pred,c,threshold,floor)
                    self.assertEqual(a,b);self.assertEqual(self.old.ap(a[0],a[1]),cpu.ap(b[0],b[1]))
                    for ties in (True,False):self.assertEqual(self.old.curve(a[0],a[1],ties),cpu.curve(b[0],b[1],ties))
    def test_native_scope_equivalence(self):
        sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
        from evaluation import metric_row
        for r in manifest()['rows']:
            a=metric_row(r);b=cpu.scopes(r)
            self.assertEqual(a['coverage'],b['coverage']);self.assertEqual(a['fp_coverage'],b['fp_coverage'])
    def test_full_internal_metric_aggregate_equivalence(self):
        sys.path.insert(0,str(ROOT/'tools/phase10b-v1'))
        from internal import evaluate
        rows=internal_rows();pred={}
        for r in rows:
            ps=[]
            for a in r['annotations']:
                if not a['ignore']:
                    ps.extend([dict(label=a['label'],score=.9,box=a['box']),dict(label=a['label'],score=.8,box=a['box'])])
            ps.append(dict(label='chair',score=.3,box=[.01,.01,.03,.03]));pred[r['image_id']]=ps
        old=evaluate(rows,pred);new=cpu.evaluate([cpu.scopes(r) for r in rows],pred)
        for key in ('precision','recall','ap50','ap75','f1'):self.assertAlmostEqual(old[key],new[key],places=14)
        for key in ('tp','fp','fn','categories','pr_curve','diagnostic_heuristics'):self.assertEqual(old[key],new[key])
    def test_nms_order_ties_and_strict_iou(self):
        ps=[dict(label='chair',score=.7,box=[.1,.1,.5,.5]),dict(label='chair',score=.7,box=[.1,.1,.5,.5]),dict(label='bed',score=.7,box=[.1,.1,.5,.5])]
        self.assertEqual(cpu.nms(ps),self.fusion.nms(ps));self.assertEqual(cpu.nms(ps,1),ps)
        self.assertRaises(ValueError,cpu.nms,[dict(ps[0],score=float('nan'))])
    def test_unknown_ignore_and_duplicate(self):
        r=copy.deepcopy(self.rows[0]);r['image_id']='fixture';r['coverage']={c:c=='chair' for c in CLASSES};r['fp_coverage']={c:False for c in CLASSES}
        r['annotations']=[dict(label='chair',ignore=False,box=[.1,.1,.5,.5])]
        ps={'fixture':[dict(label='chair',score=.8,box=[.1,.1,.5,.5]),dict(label='chair',score=.8,box=[.1,.1,.5,.5]),dict(label='bed',score=.6,box=[.7,.7,.9,.9])]}
        for exhaustive in (False,True):
            r['fp_coverage']['chair']=exhaustive
            self.assertEqual(self.old.events([r],ps,'chair'),cpu.events([r],ps,'chair'))
            self.assertEqual(self.old.events([r],ps,'bed'),cpu.events([r],ps,'bed'))
    def test_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'new.json';save(p,dict(x=1));self.assertRaises(FileExistsError,save,p,dict(x=2));self.assertEqual(read(p),dict(x=1))
    def test_no_external_membership(self):
        self.assertEqual(len(internal_rows()),19)
        self.assertRaises(PermissionError,source_path,dict(internal_rows()[0],historical_partition='frozen'))

class BoxTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers.models.owlv2.modeling_owlv2 import Owlv2BoxPredictionHead,Owlv2ClassPredictionHead
        from transformers import Owlv2Config,Owlv2VisionConfig,Owlv2TextConfig
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__();self.owlv2=torch.nn.Linear(768,768)
                config=Owlv2Config(text_config=Owlv2TextConfig().to_dict(),vision_config=Owlv2VisionConfig(hidden_size=768).to_dict())
                self.box_head=Owlv2BoxPredictionHead(config);self.class_head=Owlv2ClassPredictionHead(config);self.objectness_head=Owlv2BoxPredictionHead(config,out_dim=1)
        torch.manual_seed(20261119);self.model=Stub();self.row=copy.deepcopy(internal_rows()[0])
        self.row['annotations']=[dict(label='chair',ignore=False,box=[.1,.2,.6,.8])];self.row['explicit_native_negative']={}
    def test_real_box_head_backward_and_frozen_optimizer_scope(self):
        import torch
        import box_only as box
        opt=box.optimizer(self.model);before={n:p.detach().clone() for n,p in self.model.named_parameters()}
        x=torch.randn(1,12,768);features=self.model.owlv2(x);queries=torch.randn(1,7,512)
        logits=self.model.class_head(features,queries,None)[0][0];b=self.model.box_head(features)[0].sigmoid()
        loss,parts=box.loss(logits,b,self.row);loss.backward();box.scope(self.model,opt,True)
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),1184260)
        self.assertEqual(parts['classification'],0);self.assertFalse(parts['unknowns_as_background'])
        for n,p in self.model.named_parameters():self.assertTrue(torch.equal(before[n],p))
        self.assertTrue(torch.equal(logits,self.model.class_head(features,queries,None)[0][0]))
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if not n.startswith('box_head.')))
        bad=torch.optim.AdamW(self.model.parameters());self.assertRaises(ValueError,box.scope,self.model,bad)
        self.assertFalse(torch.cuda.is_initialized())
    def test_ambiguous_and_nonfinite_rejected(self):
        import torch
        import box_only as box
        self.row['annotations'][0]['label']='table'
        self.assertRaises(ValueError,box.loss,torch.zeros(3,7),torch.ones(3,4)*.5,self.row)
        self.assertRaises(ValueError,box.loss,torch.full((3,7),float('nan')),torch.ones(3,4)*.5,self.row)
    def test_no_classification_gradient_even_with_grad_enabled_logits(self):
        import torch
        import box_only as box
        logits=torch.zeros(3,7,requires_grad=True);b=torch.full((3,4),.4,requires_grad=True)
        value,_=box.loss(logits,b,self.row);value.backward();self.assertIsNone(logits.grad);self.assertIsNotNone(b.grad)
    def test_historical_positive_only_classification_gradient_mechanism(self):
        import torch
        from expanded import detection_loss
        logits=torch.zeros(4,7,requires_grad=True);boxes=torch.full((4,4),.4,requires_grad=True)
        value,parts=detection_loss(logits,boxes,self.row);value.backward()
        self.assertEqual(parts['valid_class_terms'],1)
        self.assertEqual(int((logits.grad<0).sum()),1);self.assertEqual(int((logits.grad>0).sum()),0)
        self.assertFalse(parts['unknowns_as_background'])

class RecoveryTests(unittest.TestCase):
    def test_authorization_gate_before_any_inference(self):
        import recover_validation as recovery
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unapproved.json';save(p,dict(approved=False));dest=Path(d)/'outputs'
            with patch.object(recovery,'DEST',dest):self.assertRaises(PermissionError,recovery.infer,p)
            self.assertFalse(dest.exists())
    def test_raw_cache_roundtrip_immutable_and_corruption(self):
        import recover_validation as recovery
        import numpy as np
        arrays=dict(logits=np.zeros((2,7),dtype='float32'),square_cxcywh=np.ones((2,4),dtype='float32')*.5,
            boxes=np.array([[.1,.1,.5,.5]]*2,dtype='float32'),scores=np.full(2,.5,dtype='float32'),labels=np.zeros(2,dtype='int64'))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.npz';recovery.write_arrays(p,arrays)
            for k,a in recovery.load_arrays(p).items():np.testing.assert_array_equal(a,arrays[k])
            self.assertRaises(FileExistsError,recovery.write_arrays,p,arrays)
            bad=dict(arrays,labels=np.zeros(2,dtype='float32'));self.assertRaises(ValueError,recovery.arrays_valid,bad)
            bad=dict(arrays,logits=np.full((2,7),np.nan,dtype='float32'));self.assertRaises(ValueError,recovery.arrays_valid,bad)
    def test_exact_recovery_scope_and_zero_updates(self):
        import recover_validation as recovery
        p=recovery.proposal();self.assertFalse(p['approved']);self.assertEqual(p['max_forwards'],19);self.assertEqual(p['optimizer_updates'],0)
        self.assertEqual(len(p['image_ids']),19);self.assertEqual(p['max_seconds'],600);self.assertEqual(p['checkpoint_sha256'],'3c8774892d6ba85ddb01f735fa9391790e4200259dd302176ab429a863d6eb2f')
        source=(HERE/'recover_validation.py').read_text();self.assertNotIn('optimizer.step(',source);self.assertNotIn('scaler.step(',source)
    def test_cpu_score_requires_complete_cache(self):
        import recover_validation as recovery
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            with patch.object(recovery,'DEST',Path(d)):
                self.assertRaises(FileNotFoundError,recovery.score)

class AnalysisTests(unittest.TestCase):
    def test_pr_operating_selection_and_ties(self):
        from analyze import best_at_recall,best_at_precision,quantiles
        xs=[dict(threshold=.9,tp=1,fp=0,recall=.5,precision=1),dict(threshold=.8,tp=2,fp=2,recall=1.,precision=.5)]
        self.assertEqual(best_at_recall(xs,.9),xs[1]);self.assertEqual(best_at_precision(xs,.8),xs[0]);self.assertIsNone(best_at_recall(xs,1.1))
        self.assertEqual(quantiles(xs)['scored_events'],4);self.assertEqual(quantiles(xs)['quantiles']['0.5'],.8)

if __name__=='__main__':unittest.main()
