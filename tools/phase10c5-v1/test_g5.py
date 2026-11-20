"""Synthetic review and CPU tensor fixtures; no real human approvals created."""
import copy
import json
import threading
import unittest
import tempfile
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from g5_common import *
from g5_review import ACTION_STATEMENT,validate,valid_box,registry
import g5_server

def fixture_registry():
    photos=[];tasks=[]
    for i,role in enumerate(('optimization','optimization','internal_validation_diagnostic_only')):
        p=dict(photo_key=f'photo{i}',image_id=f'SYNTHETIC-{i}',sha256=str(i)*64,identity_group=f'fixture-group{i}',width=1000,height=800,orientation=1,role=role,path='not-a-real-photo',native=[dict(label='Chair',category='protected_furniture',box=[.7,.1,.9,.4],dataset='fixture',ignored=False)],protected=[dict(box=[.7,.1,.9,.4],reason='Chair positive')])
        photos.append(p);tasks.append(dict(task_id=f'task{i}',photo_key=p['photo_key'],box=[.1,.1,.3,.3],kind='source_annotation' if i<2 else 'saved_model_prediction',label='Person',group='people',source_dataset='fixture',source_annotation_keys=['synthetic-object'],source_grounded=i<2,eligible_for_training_review=i<2,model_prediction=i==2))
    return dict(schema='fixture-registry',photos=photos,tasks=tasks,taxonomy_sha256=digest(TAXONOMY),source_manifest_sha256='SYNTHETIC-NOT-A-REAL-MANIFEST',synthetic_fixture=True)

def submission(reg,negative=False,index=0):
    p=reg['photos'][index];t=reg['tasks'][index]
    r={k:p[k] for k in ('photo_key','image_id','sha256','identity_group','width','height','orientation')}
    r.update(task_id=t['task_id'],box=t['box'].copy(),geometry_kind='source',decision='not_chair',human_reviewed=True,confirmed=True,
        reviewer='SYNTHETIC TEST ONLY',review_date='2026-11-20',evidence='Synthetic explicit test action only',evidence_kind='explicit_action_log',hard_negative=negative,source_label_confirmed=False)
    if negative:r.update(explicit_absence_statement=ACTION_STATEMENT,confirmed_box=r['box'].copy(),review_covers_seating_stools=True)
    return dict(schema=SCHEMA,registry_sha256=digest(reg),taxonomy_sha256=digest(TAXONOMY),reviews=[r],proposals=[])

class ReviewTests(unittest.TestCase):
    def setUp(self):self.reg=fixture_registry();self.p=submission(self.reg)
    def test_object_decision_is_not_absence(self):self.assertEqual(validate(self.p,self.reg)['accepted_hard_negatives'],0)
    def test_one_explicit_absence_action(self):self.assertEqual(validate(submission(self.reg,True),self.reg)['accepted_hard_negatives'],1)
    def test_named_label_independent_of_absence(self):
        self.p['reviews'][0]['source_label_confirmed']=True;v=validate(self.p,self.reg);self.assertEqual((v['confirmed_named_source_objects'],v['accepted_hard_negatives']),(1,0))
    def test_no_silent_geometry_change(self):
        self.p['reviews'][0]['box']=[.1,.1,.4,.4]
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_changed_box_needs_confirmation(self):
        p=submission(self.reg,True);p['reviews'][0].update(box=[.1,.1,.4,.4],geometry_kind='human_corrected')
        with self.assertRaises(ValueError):validate(p,self.reg)
    def test_protected_positive_overlap(self):
        p=submission(self.reg,True);r=p['reviews'][0];r.update(box=[.6,.1,.8,.3],confirmed_box=[.6,.1,.8,.3],geometry_kind='human_corrected');self.assertEqual(len(validate(p,self.reg)['blocked']),1)
    def test_whole_photo_not_allowed(self):
        p=submission(self.reg,True);p['reviews'][0].update(box=[0,0,1,1],confirmed_box=[0,0,1,1],geometry_kind='human_corrected');self.assertTrue(validate(p,self.reg)['blocked'])
    def test_validation_cannot_train(self):
        with self.assertRaises(PermissionError):validate(submission(self.reg,True,2),self.reg)
    def test_validation_review_can_be_diagnostic(self):self.assertEqual(validate(submission(self.reg,False,2),self.reg)['accepted_hard_negatives'],0)
    def test_hash_binding(self):
        self.p['reviews'][0]['sha256']='altered'
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_no_automatic_confirmation(self):
        self.p['reviews'][0]['human_reviewed']=False
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_registry_binding(self):
        self.p['registry_sha256']='wrong'
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_stools_taxonomy_binding(self):
        self.p['taxonomy_sha256']='wrong'
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_empty_reviewer_not_evidence(self):
        self.p['reviews'][0]['reviewer']=' '
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_seating_coverage_required(self):
        p=submission(self.reg,True);p['reviews'][0]['review_covers_seating_stools']=False
        with self.assertRaises(ValueError):validate(p,self.reg)
    def test_duplicate_tasks_rejected(self):
        self.p['reviews']*=2
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_new_boxes_stay_inactive(self):
        self.p['proposals']=[dict(self.reg['photos'][0],proposal_id='fixture',box=[.4,.5,.6,.8],approved_for_training=True)]
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_proposed_overlap_protected(self):
        p=submission(self.reg,True);p['proposals']=[dict(self.reg['photos'][0],proposal_id='fixture',box=[.1,.1,.2,.2],approved_for_training=False)];self.assertTrue(validate(p,self.reg)['blocked'])
    def test_nan_outside_degenerate_rejected(self):
        for b in ([float('nan'),0,1,1],[-1,0,1,1],[.5,.5,.5,.8],[False,0,1,1]):self.assertFalse(valid_box(b))
    def test_native_taxonomy_not_generic_other(self):
        self.assertEqual(category_group('stool'),'protected_furniture');self.assertEqual(category_group('footstool'),'protected_furniture');self.assertEqual(category_group('potted plant'),'plants_flowers');self.assertEqual(category_group('flowerpot'),'tableware_small_objects')
    def test_native_coordinates_no_clipping(self):
        self.assertEqual(normalized_box([100,80,200,160],1000,800),[.1,.1,.3,.3])
        with self.assertRaises(ValueError):normalized_box([900,0,200,10],1000,800)
    def test_private_real_registry_separated(self):
        reg=registry();rows=manifest()['rows'];a=[r for r in rows if optimizer_member(r)];b=[r for r in rows if not optimizer_member(r)]
        for key in ('sha256','image_id','identity_group'):self.assertFalse({r[key] for r in a}&{r[key] for r in b})
        self.assertEqual(sum(p['role']=='optimization' for p in reg['photos']),59)
        self.assertEqual(validate(read(RAW/'empty-review.json'),reg)['accepted_hard_negatives'],0)

class ObjectiveTests(unittest.TestCase):
    def test_unknown_gradients_zero(self):
        import torch
        from g5_objective import partial_targets,masked_focal
        x=torch.zeros(1,9,requires_grad=True);y,m=partial_targets(x.shape,[(0,7)],[]);masked_focal(x,y,m).backward();self.assertEqual(x.grad[0,:7].abs().sum().item(),0);self.assertNotEqual(x.grad[0,7].item(),0)
    def test_conflicting_targets_rejected(self):
        from g5_objective import partial_targets
        with self.assertRaises(ValueError):partial_targets((1,9),[(0,1)],[(0,1)])
    def test_residual_zero_initialization_and_general_prompts(self):
        import torch
        from g5_objective import FurnitureResidual
        torch.manual_seed(2);module=FurnitureResidual(8,4);features=torch.randn(2,3,8,requires_grad=True);base=torch.randn(2,3,10,requires_grad=True);slots=list(range(7));out=module(features,base,slots);self.assertTrue(torch.equal(out,base))
        with torch.no_grad():module.adapter[-1].bias.fill_(.5)
        out=module(features,base,slots);self.assertTrue(torch.equal(out[...,7:],base[...,7:]));out.sum().backward();self.assertIsNone(features.grad);self.assertIsNone(base.grad);self.assertIsNotNone(module.adapter[-1].weight.grad)
    def test_bad_prompt_alignment_rejected(self):
        import torch
        from g5_objective import FurnitureResidual
        with self.assertRaises(ValueError):FurnitureResidual(8,4)(torch.zeros(1,8),torch.zeros(1,8),[0]*7)
    def test_nonfinite_is_not_suppressed(self):
        import torch
        from g5_objective import masked_focal
        with self.assertRaises(ValueError):masked_focal(torch.tensor([float('nan')]),torch.zeros(1),torch.ones(1))

class ExportTests(unittest.TestCase):
    def test_versioned_export_preserves_individual_evidence(self):
        import g5_review
        reg=fixture_registry();p=submission(reg,True);photo=reg['photos'][0]
        row=dict(photo,historical_partition='train',internal_partition='optimization',annotations=[dict(native_dataset='COCO')])
        # Entire registry and envelope are synthetic; temporary directory only.
        with tempfile.TemporaryDirectory() as temp,patch.object(g5_review,'RAW',Path(temp)),patch.object(g5_review,'source_photo',lambda _:None),patch.object(g5_review,'manifest',lambda:dict(rows=[row])),patch.object(g5_review,'compare_masks',lambda _:dict(changed_mask_terms=3)),patch('preflight.validate_region_record',lambda *_:True):
            a=g5_review.export_review(p,reg);b=g5_review.export_review(p,reg)
            self.assertNotEqual(a['export_id'],b['export_id']);self.assertEqual(a['hard_negatives'],1)
            out=read(Path(temp)/'exports'/a['export_id']/'hard-negative-supervision.json')
            self.assertEqual(out['accepted_Chair_negative_evidence'][0]['individual_confirmations'],p['reviews']);self.assertFalse(out['training_authorized'])
            for line in (Path(temp)/'exports'/a['export_id']/'SHA256SUMS').read_text().splitlines():
                value,name=line.split('  ',1);self.assertEqual(value,sha(Path(temp)/'exports'/a['export_id']/name))

class ServerTests(unittest.TestCase):
    def setUp(self):
        self.s=g5_server.make_server(0,fixture_registry());self.thread=threading.Thread(target=self.s.serve_forever,daemon=True);self.thread.start();self.base=f'http://127.0.0.1:{self.s.server_port}'
    def tearDown(self):self.s.shutdown();self.s.server_close();self.thread.join()
    def test_local_only_and_assets(self):
        self.assertEqual(self.s.server_address[0],'127.0.0.1')
        with urlopen(self.base+'/') as r:self.assertIn(b'Fast object review',r.read())
    def test_host_guard(self):
        with self.assertRaises(HTTPError) as cm:urlopen(Request(self.base+'/api/registry',headers={'Host':'attacker.invalid'}))
        self.assertEqual(cm.exception.code,403)
    def test_token_guard(self):
        with self.assertRaises(HTTPError) as cm:urlopen(Request(self.base+'/api/export',data=b'{}',headers={'Content-Type':'application/json'}))
        self.assertEqual(cm.exception.code,403)

if __name__=='__main__':unittest.main()
