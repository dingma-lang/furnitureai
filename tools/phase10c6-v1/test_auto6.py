"""CPU geometry/evidence/dashboard fixtures; never real human approvals."""
import copy,json,threading,unittest,tempfile
from unittest.mock import patch
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from common6 import *
from geometry6 import certify,simple,polygons,propose,point_inside
from evidence6 import validate_source

class GeometryTests(unittest.TestCase):
    def setUp(self):self.poly=[(0,0),(100,0),(100,100),(0,100)]
    def test_interior(self):self.assertTrue(certify([10,10,90,90],self.poly))
    def test_boundary_not_absence(self):self.assertFalse(certify([0,0,50,50],self.poly))
    def test_margin(self):self.assertFalse(certify([1,1,50,50],self.poly))
    def test_outside(self):self.assertFalse(certify([90,90,110,110],self.poly))
    def test_concave_crossing_not_accepted(self):self.assertFalse(certify([10,10,90,90],[(0,0),(100,0),(100,100),(60,100),(60,30),(40,30),(40,100),(0,100)]))
    def test_bowtie_not_simple(self):self.assertFalse(simple([(0,0),(100,100),(0,100),(100,0)]))
    def test_native_out_of_frame_rejected(self):self.assertFalse(polygons(dict(segmentation=[[-1,0,100,0,100,100,0,100]]),100,100))
    def test_unsupported_rle_does_not_get_invented(self):self.assertFalse(polygons(dict(segmentation=dict(size=[100,100],counts='unknown')),100,100))
    def test_crowd_rejected(self):self.assertFalse(polygons(dict(iscrowd=1,segmentation=[[0,0,100,0,100,100,0,100]]),100,100))
    def test_deterministic_interior_proposal(self):
        a=dict(segmentation=[[0,0,100,0,100,100,0,100]]);p=propose(a,100,100,[]);self.assertEqual(p,propose(a,100,100,[]));self.assertTrue(certify(p['box_pixels'],self.poly))
    def test_protected_boxes_block(self):self.assertIsNone(propose(dict(segmentation=[[0,0,100,0,100,100,0,100]]),100,100,[dict(box=[0,0,1,1])]))

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen=candidates();cls.observations={r['candidate_id']:read(RAW/'observations'/(r['candidate_id']+'.json')) for r in cls.frozen};cls.bundle=read(RAW/'source-negative-evidence.json');cls.rows=manifest()['rows']
    def test_real_bundle_source_geometry(self):
        from masks6 import validate_bundle
        self.assertTrue(validate_bundle(self.bundle,self.rows,self.frozen,self.observations))
    def test_all_requested_visual_inspection_is_recorded(self):
        self.assertEqual(len(self.observations),208);self.assertTrue(all(o['visual_inspected'] and not o['human_reviewed'] for o in self.observations.values()))
    def test_b_tier_not_promoted(self):
        r=next(r for r in self.frozen if self.observations[r['candidate_id']]['visual_outcome']=='automatic_only')
        with self.assertRaises(ValueError):validate_source(r,self.observations[r['candidate_id']],self.frozen,self.observations)
    def test_fake_human_flags_rejected(self):
        from masks6 import validate_bundle
        b=copy.deepcopy(self.bundle);b['records'][0]['human_reviewed']=True
        with self.assertRaises(ValueError):validate_bundle(b,self.rows,self.frozen,self.observations)
    def test_geometry_tampering_rejected(self):
        from masks6 import validate_bundle
        b=copy.deepcopy(self.bundle);b['records'][0]['box']=[0,0,1,1]
        with self.assertRaises(ValueError):validate_bundle(b,self.rows,self.frozen,self.observations)
    def test_taxonomy_tampering_rejected(self):
        from masks6 import validate_bundle
        b=copy.deepcopy(self.bundle);b['records'][0]['source_native_category']='wrong'
        with self.assertRaises(ValueError):validate_bundle(b,self.rows,self.frozen,self.observations)
    def test_duplicate_weighting_rejected(self):
        from masks6 import validate_bundle
        b=copy.deepcopy(self.bundle);b['records'].append(b['records'][0])
        with self.assertRaises(ValueError):validate_bundle(b,self.rows,self.frozen,self.observations)
    def test_validation_not_training(self):
        from masks6 import validate_bundle
        b=copy.deepcopy(self.bundle);b['records'][0]['internal_partition']='inner_development'
        with self.assertRaises(PermissionError):validate_bundle(b,self.rows,self.frozen,self.observations)
    def test_training_default_disabled(self):
        from masks6 import authorized,DEFAULT
        with self.assertRaises(PermissionError):authorized(DEFAULT,self.bundle,manifest())
    def test_real_counts_reconcile(self):
        s=read(REPORT/'summary.json');self.assertEqual(s['source_verified_regions']+s['automatically_screened_candidates']+s['uncertain_or_rejected'],208);self.assertEqual(s['source_verified_regions'],41)
    def test_old_human_review_gates_unchanged(self):
        sys.path.insert(0,str(ROOT/'tools/phase10c2-v1'))
        from trusted_loss import evidence_valid
        row=next(r for r in self.rows if r['image_id']==self.bundle['records'][0]['image_id'])
        with self.assertRaises((ValueError,PermissionError)):evidence_valid(self.bundle['records'][0],row)

class MaskTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;self.row=dict(width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
    def call(self,xy,ev,extra=(),match=False):
        from masks6 import target_mask,xyxy_to_cxcywh
        t=self.torch;b=xyxy_to_cxcywh(t.tensor(xy));logits=t.zeros(len(b),7);ii=t.tensor([0]) if match else None;labels=t.tensor([1]) if match else None
        return target_mask(logits,b,self.row,ii,labels,ev,extra)
    def test_only_category_specific_negative(self):
        self.row['image_id']='SYNTHETIC';a,m,added=self.call([[.1,.1,.2,.2]],[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3])]);self.assertEqual(added.sum().item(),1);self.assertTrue(added[0,1])
    def test_box_not_just_center_containment(self):
        self.row['image_id']='SYNTHETIC';a,m,added=self.call([[0,0,.2,.2]],[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3])]);self.assertEqual(added.sum().item(),0)
    def test_protected_positive_wins(self):
        self.row['image_id']='SYNTHETIC';a,m,added=self.call([[.75,.75,.85,.85]],[dict(image_id='SYNTHETIC',category='chair',box=[.6,.6,.95,.95])],match=True);self.assertEqual(added.sum().item(),0);self.assertEqual(a[0,1].item(),1)
    def test_ignored_region_blocks(self):
        self.row['image_id']='SYNTHETIC';a,m,added=self.call([[.1,.1,.2,.2]],[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3])],extra=[[0,0,.5,.5]]);self.assertEqual(added.sum().item(),0)
    def test_empty_evidence_identity(self):
        from masks6 import supervision,xyxy_to_cxcywh
        t=self.torch;b=xyxy_to_cxcywh(t.tensor([[.1,.1,.2,.2]]));x=t.zeros(1,7);a,m=supervision(x,b,self.row,None,None);b,bm,add=self.call([[.1,.1,.2,.2]],[]);self.assertTrue(t.equal(a,b) and t.equal(m,bm));self.assertFalse(add.any())

class ServerTests(unittest.TestCase):
    def setUp(self):
        from server6 import make_server
        self.server=make_server(0,fixture=([],dict(summary={},masks={},results=[])));self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def test_only_local(self):self.assertEqual(self.server.server_address[0],'127.0.0.1')
    def test_results_readonly(self):
        with urlopen(self.url+'/api/results') as r:self.assertEqual(json.load(r)['results'],[])
    def test_correction_csrf(self):
        with self.assertRaises(HTTPError) as cm:urlopen(Request(self.url+'/api/correction',data=b'{}',headers={'Content-Type':'application/json'}))
        self.assertEqual(cm.exception.code,403);cm.exception.close()
    def test_no_inference_route(self):
        with self.assertRaises(HTTPError) as cm:urlopen(self.url+'/api/train')
        self.assertEqual(cm.exception.code,404);cm.exception.close()

if __name__=='__main__':unittest.main()
