"""CPU replay,provenance,isolation and read-only HTTP regression tests."""
import copy,http.client,json,threading,unittest,uuid
import numpy as np
import torch
from PIL import Image
from common14 import *
from replay14 import replay
from expanded import xyxy_to_cxcywh
from moments14 import recover_clipped
from server14 import make_server


class ReplayTests(unittest.TestCase):
    def data(self):
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False),dict(label='sofa',box=[.4,.4,.6,.6],ignore=False)],explicit_native_negative={})
        boxes=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.4,.4,.6,.6],[.1,.1,.2,.2]],dtype=torch.float32)).numpy();z=np.zeros((3,7),np.float32);z[2,:]=-10;e=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC-REGION')]
        return dict(logits=z,square_cxcywh=boxes),row,e
    def test_positive_source_negative_and_ignored_assignments(self):
        a,row,e=self.data();d,s,_=replay(a,row,e,[],True)
        self.assertEqual(s['positive_terms_by_category'],[1,1,0,0,0,0,0]);self.assertEqual(s['negative_terms_by_category'],[0,1,0,0,0,0,0]);self.assertEqual(int(d['assignment'][2]),-1)
    def test_control_does_not_apply_available_source_negatives(self):
        a,row,e=self.data();d,s,_=replay(a,row,e,[],False);self.assertEqual(int(d['added_available'].sum()),1);self.assertEqual(int(d['used'].sum()),2);self.assertEqual(s['gradient']['logit_added_negative_L2'],0)
    def test_high_score_unmatched_unknown_remains_ignored(self):
        a,row,e=self.data();a['logits'][2,1]=10;d,s,_=replay(a,row,[],[],True)
        self.assertFalse(d['used'][2].any());self.assertFalse(d['grad_classification'][2].any());self.assertFalse(d['grad_box'][2].any())
    def test_unrelated_four_category_gradients_zero(self):
        a,row,e=self.data();a['logits'][:,[2,4,5,6]]=8;d,s,_=replay(a,row,e,[],True)
        self.assertFalse(d['used'][:,[2,4,5,6]].any());self.assertFalse(d['grad_classification'][:,[2,4,5,6]].any())
    def test_protected_positive_never_source_negative(self):
        a,row,e=self.data();d,s,_=replay(a,row,e,[[0,0,.3,.3]],True);self.assertEqual(int(d['added_available'].sum()),0)
    def test_whole_query_containment_not_center_only(self):
        a,row,e=self.data();e[0]['box']=[.14,.14,.16,.16];d,s,_=replay(a,row,e,[],True);self.assertEqual(int(d['added_available'].sum()),0)
    def test_saved_mismatch_is_reported_not_silently_repaired(self):
        a,row,e=self.data();_,s,_=replay(a,row,e,[],True);bad=dict(s['stats'],classification_loss=123);_,ss,_=replay(a,row,e,[],True,bad);self.assertFalse(ss['saved_loss_and_mask_checks_passed']);self.assertIn('saved_classification_loss_differs',ss['reconstruction_warnings'])
    def test_easy_negative_focal_gradient_attenuated_relative_to_BCE(self):
        a,row,e=self.data();_,s,_=replay(a,row,e,[],True);g=s['gradient'];self.assertGreater(g['counterfactual_source_BCE_logit_gradient_L1'],10000*g['counterfactual_source_focal_logit_gradient_L1'])
    def test_uniform_positive_offset_direction_lowers_sparse_objective(self):
        a,row,e=self.data();_,s,_=replay(a,row,e,[],True);self.assertLess(s['gradient']['uniform_seven_logit_offset_derivative'],0)
    def test_nonfinite_saved_arrays_rejected(self):
        a,row,e=self.data();a['logits'][0,0]=np.nan
        with self.assertRaises(ValueError):replay(a,row,e,[],True)
    def test_Adam_moment_recovery_is_aggregate_not_parameter_update(self):
        old=torch.tensor([.1,.2]);g=torch.tensor([-.2,.5]);new=.9*old+.1*g;self.assertTrue(torch.allclose(recover_clipped(new,old,.9),g.double(),atol=1e-7))


class IsolationTests(unittest.TestCase):
    def test_exact_optimization_rows_required(self):
        row=dict(image_id='SYNTHETIC',sha256='x',identity_group='g',historical_partition='train',internal_partition='optimization');self.assertTrue(require_training_row(row,[row],[]));bad=dict(row,internal_partition='inner_development')
        with self.assertRaises(PermissionError):require_training_row(bad,[row],[])
    def test_source_group_overlap_rejected(self):
        row=dict(image_id='SYNTHETIC',sha256='x',identity_group='g',historical_partition='train',internal_partition='optimization');dev=dict(row,image_id='DEV',sha256='other')
        with self.assertRaises(PermissionError):require_training_row(row,[row],[dev])
    def test_all_real_dashboard_photos_optimization_only(self):
        data=read(RAW14/'dashboard-v3.json');_,train,dev=verify_inputs13();self.assertEqual({r['image_id'] for r in data['photos']},{r['image_id'] for r in train});self.assertFalse({r['image_id'] for r in data['photos']}&{r['image_id'] for r in dev})
    def test_actual_replay_all312_loss_and_mask_checks_pass(self):
        d=read(RAW14/'query-summaries.json');self.assertEqual(len(d['records']),312);self.assertEqual(d['reconstruction_failures'],[])
    def test_teacher_outputs_only24_training_photos_not_development(self):
        d=read(REPORT14/'teacher-feasibility.json');self.assertEqual(d['available_training_teacher_photos'],24);self.assertEqual(d['training_photos_missing_teacher_outputs'],35);self.assertFalse(d['development_features_or_errors_used_for_teacher_or_mining'])
    def test_no_new_additive_negative_labels_or_human_flags(self):
        d=read(REPORT14/'visual-findings.json');self.assertEqual(d['new_additive_training_eligible_hard_negatives'],0);self.assertEqual(d['human_reviews_created'],0)


class FixtureStore:
    def __init__(self,path):
        photo=dict(image_id='SYNTHETIC-TRAIN',photo_key='synthetictrain',role='optimization',path=str(path.relative_to(ROOT)),sha256=sha(path),width=200,height=150,identity_group='SYNTHETIC-GROUP',available_states=['original','c13_e5'],annotations=[dict(label='chair',ignore=False,box=[.1,.1,.4,.4])],native=[],protected=[],source_regions=[dict(candidate_id='SYNTHETIC-REGION',box=[.5,.5,.8,.8])]);uncached=dict(photo,image_id='SYNTHETIC-UNCACHED',photo_key='syntheticuncached',identity_group='SYNTHETIC-UNCACHED-GROUP',available_states=[])
        self.data=dict(photos=[photo,uncached],states=[dict(key='original',label='Original',epoch=0),dict(key='c13_e5',label='C13 synthetic',epoch=5)],classes=list(CLASSES),hard_negative=dict(new_source_geometry_candidate_regions=0),candidate_regions=[],inspection_samples=[],read_only=True);self.photos={p['photo_key']:p for p in self.data['photos']}
    def view(self,key,state):
        if key!='synthetictrain' or state not in ('original','c13_e5'):raise KeyError('No cached state')
        scores=[[.01,.9,.01,.01,.01,.01,.01],[.01,.8,.01,.01,.01,.01,.01],[.01,.7,.01,.01,.01,.01,.01]]
        summary=dict(saved_loss_and_mask_checks_passed=True,positive_terms_by_category=[0,1,0,0,0,0,0],negative_terms_by_category=[0,1,0,0,0,0,0],ignored_terms_by_category=[3,1,3,3,3,3,3],stats=dict(classification_loss=.1,added_Chair_negative_loss=.01),corrected_source_negatives_applied=True,gradient=dict(logit_positive_L1_by_category=[0,.1,0,0,0,0,0],logit_added_negative_L1_by_category=[0,.01,0,0,0,0,0]),saved_parameter_gradient_norms=dict(source_negative_to_data_norm_ratio=.01))
        return dict(boxes=[[.1,.1,.4,.4],[.5,.5,.8,.8],[.1,.5,.3,.8]],scores=scores,positive_bits=[2,0,0],valid_bits=[2,2,0],source_available=[0,1,0],assignment=[0,-1,-1],padding_anchor=[0,0,0],known_Chair_IoU05=[1,0,0],fragment_heuristic=[0,0,1],summary=summary)
    def detail(self,key,state,q):
        self.view(key,state)
        if not 0<=q<3:raise KeyError('Bad query')
        return dict(image_id='SYNTHETIC-TRAIN',query_index=q,box_original_pixels_unclipped=[20,15,80,60],classes=list(CLASSES),sigmoid_scores=[.1]*7,active_loss_mask=[0]*7,native_bbox_context=[],output_logit_gradients={},note='Synthetic CPU fixture;not a real review')


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=RAW14/('unit-fixture-'+uuid.uuid4().hex[:8]);cls.folder.mkdir();path=cls.folder/'synthetic.png';Image.new('RGB',(200,150),'#bdd4df').save(path);cls.store=FixtureStore(path);cls.server=make_server(0,fixture=cls.store);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path,method='GET',host=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10);c.request(method,path,headers={'Host':host or 'localhost:'+str(self.server.server_port)});r=c.getresponse();body=r.read();status=r.status;c.close();return status,body
    def test_registered_training_photo_and_query_data(self):self.assertEqual(self.request('/photo/synthetictrain')[0],200);self.assertEqual(self.request('/api/view/synthetictrain/c13_e5')[0],200)
    def test_uncached_photo_has_no_inference_fallback(self):self.assertEqual(self.request('/api/view/syntheticuncached/c13_e5')[0],404)
    def test_reserved_unknown_photo_cannot_be_served(self):self.assertEqual(self.request('/photo/frozen-evaluation')[0],404)
    def test_path_traversal_rejected(self):self.assertEqual(self.request('/photo/../../models/model.safetensors')[0],404);self.assertEqual(self.request('/photo/%2e%2e%2fmodels')[0],404)
    def test_remote_host_rejected(self):self.assertEqual(self.request('/api/data',host='attacker.test')[0],403)
    def test_no_training_inference_review_or_export_writes(self):
        for endpoint in ('/api/train','/api/infer','/api/review','/api/export'):
            self.assertEqual(self.request(endpoint,'POST')[0],405)
    def test_missing_or_invalid_query_state_rejected(self):self.assertEqual(self.request('/api/detail/synthetictrain/c13_e5/99999')[0],404);self.assertEqual(self.request('/api/view/synthetictrain/reserved')[0],404)
    def test_source_hash_failure_rejected(self):
        old=self.store.photos['synthetictrain']['sha256'];self.store.photos['synthetictrain']['sha256']='bad'
        try:self.assertEqual(self.request('/photo/synthetictrain')[0],409)
        finally:self.store.photos['synthetictrain']['sha256']=old


if __name__=='__main__':unittest.main()
