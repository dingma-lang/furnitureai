"""CPU tests of query identity, uncertainty, isolation, geometry and HTTP safety."""
import unittest,copy,uuid,threading,http.client
import numpy as np
from PIL import Image
from common9 import *
from analyze9 import reconstruct,statistical_mode_change,annotated_mechanisms
from mine9 import eligible_rectangle
from audit_objective9 import negative_focal_from_probability
from server9 import make_server

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(image_id='TRAIN-SYNTHETIC',sha256='x',identity_group='groupA',historical_partition='train',internal_partition='optimization',width=100,height=100)
        self.native=dict(metadata_candidate_eligible=True,mapped_auxiliary_category='people',source_native_ignored=False,box=[.1,.1,.9,.9],original_annotation=dict(segmentation=[[10,10,90,10,90,90,10,90]]))
    def test_coordinate_inverse_non_square(self):
        b=registered_box(np.array([.5,.25,.2,.2]),200,100);np.testing.assert_allclose(b,[.4,.3,.6,.7])
    def test_source_polygon_interior_candidate(self):
        x,reason=eligible_rectangle(self.row,[.2,.2,.6,.6],self.native,[],[],[self.row],[]);self.assertIsNotNone(x);self.assertEqual(x['box_pixels'],[20,20,60,60])
    def test_protected_chair_not_negative(self):
        x,_=eligible_rectangle(self.row,[.2,.2,.6,.6],self.native,[[.3,.3,.4,.4]],[],[self.row],[]);self.assertIsNone(x)
    def test_native_label_without_polygon_not_negative(self):
        n=copy.deepcopy(self.native);n['original_annotation']={};x,_=eligible_rectangle(self.row,[.2,.2,.6,.6],n,[],[],[self.row],[]);self.assertIsNone(x)
    def test_existing_region_not_new(self):
        x,_=eligible_rectangle(self.row,[.2,.2,.6,.6],self.native,[],[[.25,.25,.3,.3]],[self.row],[]);self.assertIsNone(x)
    def test_development_mining_rejected(self):
        row=dict(self.row,internal_partition='inner_development')
        with self.assertRaises(PermissionError):eligible_rectangle(row,[.2,.2,.6,.6],self.native,[],[],[self.row],[row])
    def test_group_overlap_rejected(self):
        dev=dict(self.row,image_id='DEV',sha256='y')
        with self.assertRaises(PermissionError):assert_mining_scope(self.row,[self.row],[dev])
    def test_out_of_frame_not_clipped_into_negative(self):
        x,_=eligible_rectangle(self.row,[-.01,.2,.6,.6],self.native,[],[],[self.row],[]);self.assertIsNone(x)
    def test_tiny_query_not_promoted(self):
        x,_=eligible_rectangle(self.row,[.2,.2,.3,.3],self.native,[],[],[self.row],[]);self.assertIsNone(x)
    def test_missing_coverage_preserves_uncertainty(self):
        p=dict(box=[.2,.2,.5,.5],duplicate=False);r=dict(annotations=[]);a=annotated_mechanisms(p,r,[],metric_module());self.assertEqual(a['taxonomy_primary'],'F');self.assertFalse(a['human_reviewed'])
    def test_native_object_box_not_absence_certificate(self):
        p=dict(box=[.2,.2,.5,.5],duplicate=False);r=dict(annotations=[]);native=[dict(box=p['box'],annotation_key='Plant:1',native_label='Plant',category='plants_flowers')]
        a=annotated_mechanisms(p,r,native,metric_module());self.assertEqual(a['taxonomy_primary'],'F');self.assertEqual(a['source_verified_findings'],[])

class ObjectiveTests(unittest.TestCase):
    def test_observations_bound_to_actually_viewed_samples(self):
        viewed=read(HERE/'viewed-evidence.json');samples=read(OUT/'review-sample.json')['records']+read(OUT/'training-review-sample.json')['records']
        self.assertEqual(viewed['sample_digests'],{p['sample_id']:digest(p) for p in samples});self.assertFalse(viewed['human_reviewed'])
    def test_common_shift_statistics(self):
        a=np.arange(21,dtype=np.float32).reshape(3,7);b=a+2;d=statistical_mode_change(a,b);self.assertEqual(d['common_mode_fraction_squared_delta'],1);self.assertEqual(d['fraction_queries_all_seven_logits_increase'],1);self.assertEqual(d['semantic_residual_RMS'],0)
    def test_focal_damps_easy_negatives(self):self.assertGreater(negative_focal_from_probability(.5)/negative_focal_from_probability(.00016),1e9)
    def test_GT_normalization_is_not_query_averaging(self):self.assertAlmostEqual(negative_focal_from_probability(.5,4)*4,negative_focal_from_probability(.5,1))
    def test_positive_only_shared_shift_gradient(self):
        import torch
        shift=torch.tensor(0.,requires_grad=True);z=torch.tensor([-3.,-2.,-3.,-3.,-3.,-3.,-3.])+shift;p=z[1].sigmoid();loss=.25*(1-p)**2*torch.nn.functional.softplus(-z[1]);g=torch.autograd.grad(loss,shift)[0]
        self.assertLess(float(g),0);self.assertTrue(torch.equal((z+1).argsort(),z.argsort()))
    def test_proposed_frozen_calibration_scope(self):
        import torch
        from proposal9 import configure_candidate
        m=torch.nn.Module();m.class_head=torch.nn.Module();m.box_head=torch.nn.Module();m.backbone=torch.nn.Linear(2,2);m.objectness_head=torch.nn.Linear(2,1)
        m.class_head.dense0=torch.nn.Linear(768,512);m.class_head.logit_shift=torch.nn.Linear(768,1);m.class_head.logit_scale=torch.nn.Linear(768,1)
        m.box_head.dense0=torch.nn.Linear(768,768);m.box_head.dense1=torch.nn.Linear(768,768);m.box_head.dense2=torch.nn.Linear(768,4)
        keys=configure_candidate(m);self.assertEqual(len(keys),8);self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),1577988);self.assertFalse(m.class_head.logit_shift.weight.requires_grad);self.assertFalse(m.backbone.weight.requires_grad)
    def test_query_identity_roundtrip(self):
        a=dict(boxes=np.array([[.1,.1,.5,.5]],dtype=np.float32),scores=np.array([.8],dtype=np.float32),labels=np.array([1],dtype=np.int64));old=[dict(box=a['boxes'][0].tolist(),score=float(a['scores'][0]),label='chair')];ps=reconstruct({},a,old,metric_module());self.assertEqual(ps[0]['query_index'],0)
    def test_changed_saved_prediction_rejected(self):
        a=dict(boxes=np.array([[.1,.1,.5,.5]],dtype=np.float32),scores=np.array([.8],dtype=np.float32),labels=np.array([1],dtype=np.int64))
        with self.assertRaises(ValueError):reconstruct({},a,[],metric_module())

class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=OUT/('cpu-fixture-'+uuid.uuid4().hex[:8]);cls.folder.mkdir();p=cls.folder/'synthetic.png';Image.new('RGB',(100,100),'white').save(p);cls.row=dict(photo_key='synthetic',path=str(p.relative_to(ROOT)),sha256=sha(p));cls.server=make_server(0,fixture=dict(photos=[cls.row]));cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path,method='GET',host=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5);headers={'Host':host or f'localhost:{self.server.server_port}'};c.request(method,path,headers=headers);r=c.getresponse();body=r.read();c.close();return r.status,body
    def test_photograph_hash_and_allowlist(self):self.assertEqual(self.request('/photo/synthetic')[0],200);self.assertEqual(self.request('/photo/../../models')[0],404)
    def test_nonlocal_host_rejected(self):self.assertEqual(self.request('/api/data',host='evil.invalid')[0],403)
    def test_no_training_endpoint(self):self.assertEqual(self.request('/api/train','POST')[0],405)
    def test_checkpoint_route_unavailable(self):self.assertEqual(self.request('/models/model.safetensors')[0],404)

if __name__=='__main__':unittest.main()
