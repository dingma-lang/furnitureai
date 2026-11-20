import copy
import io
import os
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from n2 import *
from trusted_loss import loss,negative_mask,evidence_valid,validate_bundle,legacy_loss,load_configuration
from review import validate_submission,DECISIONS

WEIGHTS=dict(classification=1,l1=5,giou=2,gamma=2,positive_alpha=.25,negative_alpha=.75)

def row():
    return dict(image_id='synthetic',sha256='fixture-sha',identity_group='fixture-group',historical_partition='train',internal_partition='optimization',width=100,height=100,
        annotations=[dict(label='chair',native_id='/m/01mzpv',native_dataset='OpenImages',ignore=False,box=[.1,.1,.3,.3])],explicit_native_negative={c:False for c in CLASSES},source_native_dataset='OpenImages')
def evidence(r,kind='complete_with_positives'):
    return dict({k:r[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')},category='chair',native_category='/m/01mzpv',kind=kind,complete_for_chair=True,
        human_reviewed=True,confirmed=True,reviewer='SYNTHETIC TEST FIXTURE',review_date='2026-11-19',evidence='Synthetic fixture; never a real human submission',trust_basis='human_category_scope_review',
        potential_unannotated_chair_regions=[],native_annotation_digest=digest(r['annotations']))
def config(r,ev):
    return dict(mode='trusted_chair_negative_v1',enabled=True,validated=True,approved=True,approval_reference='CPU SYNTHETIC FIXTURE ONLY',legacy_weights=WEIGHTS,evidence_sha256=digest(ev),row_bindings={r['image_id']:digest(r)})
def tensors():
    return torch.zeros(3,7,requires_grad=True),torch.tensor([[.2,.2,.2,.2],[.65,.65,.1,.1],[.85,.85,.1,.1]],requires_grad=True)

class LossTests(unittest.TestCase):
    def test_default_legacy_loss_and_gradients_exact(self):
        r=row();a,b=tensors();x,parts=loss(a,b,r);y,old=legacy_loss(a,b,r);self.assertTrue(torch.equal(x,y));self.assertEqual(parts,old)
        gx=torch.autograd.grad(x,(a,b),retain_graph=True);gy=torch.autograd.grad(y,(a,b));self.assertTrue(all(torch.equal(p,q) for p,q in zip(gx,gy)))
    def test_corrected_mode_disabled_and_coefficients_pinned(self):
        r=row();e=[evidence(r)];a,b=tensors();c=config(r,e)
        for flag in ('enabled','validated','approved'):
            self.assertRaises(PermissionError,loss,a,b,r,dict(c,**{flag:False}),e)
        self.assertRaises(ValueError,loss,a,b,r,dict(c,legacy_weights=dict(WEIGHTS,l1=6)),e)
    def test_known_false_chair_positive_gradient_unknown_other_classes_zero(self):
        r=row();ev=[evidence(r)];a,b=tensors();value,p=loss(a,b,r,config(r,ev),ev);value.backward()
        self.assertEqual(p['added_chair_negative_terms'],2);self.assertGreater(a.grad[1,1],0);self.assertLess(a.grad[0,1],0)
        self.assertEqual(float(a.grad[:,0].abs().sum()),0);self.assertEqual(p['normalization_native_GT'],1)
        self.assertGreater(p['new_trusted_chair_negative'],0);self.assertFalse(p['objectness_supervised'])
    def test_no_evidence_no_new_penalty(self):
        r=row();a,b=tensors();value,p=loss(a,b,r,config(r,[]),[]);old,_=legacy_loss(a,b,r);self.assertTrue(torch.equal(value,old));self.assertEqual(p['added_chair_negative_terms'],0)
    def test_verified_absence_only_after_explicit_review(self):
        r=row();r['annotations']=[];ev=[evidence(r,'full_image_absent')];a,b=tensors();v,p=loss(a,b,r,config(r,ev),ev);v.backward()
        self.assertEqual(p['added_chair_negative_terms'],3);self.assertTrue((a.grad[:,1]>0).all());self.assertEqual(p['normalization_native_GT'],1)
        ev[0]['human_reviewed']=False;self.assertRaises(PermissionError,loss,a,b,r,config(r,ev),ev)
    def test_absence_contradiction_rejected(self):
        r=row();self.assertRaises(ValueError,evidence_valid,evidence(r,'full_image_absent'),r)
    def test_potential_unannotated_chair_protected(self):
        r=row();e=evidence(r,'negative_region');e.update(regions=[[.5,.5,1,1]],potential_unannotated_chair_regions=[[.58,.58,.73,.73]])
        a,b=tensors();v,p=loss(a,b,r,config(r,[e]),[e]);v.backward();self.assertEqual(float(a.grad[1,1]),0);self.assertGreater(a.grad[2,1],0)
    def test_unconfirmed_completeness_and_missing_regions_rejected(self):
        r=row();e=evidence(r);e['complete_for_chair']=False;self.assertRaises(ValueError,evidence_valid,e,r)
        e=evidence(r);e['potential_unannotated_chair_regions']=[[.5,.5,.8,.8]];self.assertRaises(PermissionError,evidence_valid,e,r)
    def test_negative_region_requires_full_containment(self):
        r=row();e=evidence(r,'negative_region');e['regions']=[[.6,.6,.67,.67]];a,b=tensors();_,p=loss(a,b,r,config(r,[e]),[e]);self.assertEqual(p['added_chair_negative_terms'],0)
    def test_known_chair_and_ignored_regions_overlap_not_only_centers(self):
        r=row();r['annotations'].append(dict(label='table',native_dataset='OpenImages',native_id='/m/04bcr3',ignore=True,box=[.66,.66,.72,.72]))
        e=evidence(r);a,b=tensors();v,p=loss(a,b,r,config(r,[e]),[e]);v.backward();self.assertEqual(float(a.grad[1,1]),0);self.assertGreater(a.grad[2,1],0)
    def test_mixed_explicit_sofa_negative_chair_unknown(self):
        r=row();r['explicit_native_negative']['sofa']=True;a,b=tensors();v,p=loss(a,b,r,config(r,[]),[]);v.backward()
        self.assertTrue((a.grad[:,0]>0).all());self.assertEqual(float(a.grad[1:,1].abs().sum()),0)
    def test_normalization_native_positive_count_unchanged(self):
        r=row();r['annotations'].append(dict(label='bed',native_dataset='OpenImages',native_id='/m/03ssj5',ignore=False,box=[.6,.6,.7,.7]));e=evidence(r)
        a,b=tensors();v,p=loss(a,b,r,config(r,[e]),[e]);self.assertEqual(p['normalization_native_GT'],2);self.assertEqual(p['valid_positive_terms'],2)
    def test_padding_is_not_trusted_negative(self):
        r=row();r['width']=200;r['height']=100;e=evidence(r);a,b=tensors();_,p=loss(a,b,r,config(r,[e]),[e]);self.assertEqual(p['added_chair_negative_terms'],0)
    def test_hash_annotation_date_and_category_mismatch_rejected(self):
        r=row();e=evidence(r)
        for key,value in [('sha256','changed'),('native_category',62),('native_annotation_digest','changed'),('review_date','not-a-date')]:
            self.assertRaises(ValueError,evidence_valid,dict(e,**{key:value}),r)
        a,b=tensors();self.assertRaises(ValueError,loss,a,b,dict(r,width=99),config(r,[e]),[e])
    def test_validation_and_reserved_evidence_rejected(self):
        for part in ('inner_development','frozen_evaluation','development'):
            r=row();r['internal_partition']=part;self.assertRaises(PermissionError,evidence_valid,evidence(r),r)
    def test_bundle_new_data_and_group_leakage_rejected(self):
        r=row();e=evidence(r);m=dict(rows=[r]);c=dict(config(r,[e]),dataset_sha256=digest(m));self.assertTrue(validate_bundle(c,[e],m))
        self.assertRaises(PermissionError,validate_bundle,dict(config(r,[]),dataset_sha256=digest(m)),[],m)
        other=dict(r,image_id='validation',internal_partition='inner_development');m['rows'].append(other);c['dataset_sha256']=digest(m);self.assertRaises(PermissionError,validate_bundle,c,[e],m)
    def test_determinism(self):
        r=row();e=[evidence(r)];a,b=tensors();x,p=loss(a,b,r,config(r,e),e);y,q=loss(a,b,r,config(r,e),e);self.assertTrue(torch.equal(x,y));self.assertEqual(p,q)
    def test_real_head_gradient_scope_no_optimizer_updates_checkpoint_compatible(self):
        from transformers import Owlv2Config
        from transformers.models.owlv2.modeling_owlv2 import Owlv2ClassPredictionHead,Owlv2BoxPredictionHead
        from expanded import configure_heads,load_delta,HEAD_KEYS
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__();self.owlv2=torch.nn.Linear(768,768);self.objectness_head=torch.nn.Linear(768,1);c=Owlv2Config();self.class_head=Owlv2ClassPredictionHead(c);self.box_head=Owlv2BoxPredictionHead(c)
        torch.manual_seed(20261119);m=Stub();configure_heads(m);before={n:p.detach().clone() for n,p in m.named_parameters()};r=row();e=[evidence(r)]
        features=m.owlv2(torch.randn(1,8,768));logits=m.class_head(features,torch.randn(1,7,512),None)[0][0];boxes=m.box_head(features)[0].sigmoid()
        value,_=loss(logits,boxes,r,config(r,e),e);value.backward()
        self.assertEqual({n for n,p in m.named_parameters() if p.requires_grad},HEAD_KEYS)
        self.assertTrue(all(p.grad is None for p in m.parameters() if not p.requires_grad));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
        self.assertTrue(all(torch.equal(before[n],p) for n,p in m.named_parameters()))
        buf=io.BytesIO();torch.save(dict(weights={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}),buf);buf.seek(0);payload=torch.load(buf,map_location='cpu',weights_only=True);load_delta(m,payload)
        self.assertTrue(all(torch.equal(before[n],p) for n,p in m.named_parameters()));self.assertFalse(torch.cuda.is_initialized())

class PreparationReviewTests(unittest.TestCase):
    def test_actual_inventory_no_new_trusted_negatives(self):
        x=read(REPORT/'candidate-manifest-index.json')['counts'];self.assertEqual(x['priority_training_scope_reviews'],9);self.assertEqual(x['trustworthy_new_chair_negative_records'],0)
        self.assertEqual(len(read(RAW/'review-queue.json')['records']),65)
        self.assertFalse(load_configuration(REPORT/'arm-B-corrected.json')['enabled']);self.assertEqual(read(RAW/'trusted-chair-evidence-v1.json')['records'],[])
    def payload(self,item,decision):
        d={k:item[k] for k in ('item_id','image_id','sha256','identity_group','historical_partition','internal_partition','kind')}
        return dict(d,decision=decision,reviewer='SYNTHETIC FIXTURE',review_date='2026-11-19',evidence='Synthetic validation example only',human_reviewed=True,confirmed=True)
    def test_six_diagnostic_options_never_enter_training(self):
        q=read(RAW/'review-queue.json');item=q['records'][0]
        for d in DECISIONS:
            p=dict(schema='phase10c2-human-review-v1',queue_sha256=digest(q),records=[self.payload(item,d)])
            result=validate_submission(p,q,manifest()['rows']);self.assertEqual(result['trusted_records'],[]);self.assertFalse(result['training_authorized'])
    def test_explicit_scope_review_creates_only_proposed_evidence(self):
        q=read(RAW/'review-queue.json');item=next(x for x in q['records'] if x['kind']=='training_scope_review')
        p=dict(schema='phase10c2-human-review-v1',queue_sha256=digest(q),records=[self.payload(item,'complete_for_chair')])
        result=validate_submission(p,q,manifest()['rows']);self.assertEqual(len(result['trusted_records']),1);self.assertFalse(result['configuration_automatically_enabled'])
        evidence_valid(result['trusted_records'][0],next(r for r in manifest()['rows'] if r['image_id']==item['image_id']))
        p['records'][0]['confirmed']=False;self.assertRaises(ValueError,validate_submission,p,q,manifest()['rows'])
    def test_review_tampering_rejected(self):
        q=read(RAW/'review-queue.json');item=q['records'][0];d=self.payload(item,'background_non_furniture')
        p=dict(schema='phase10c2-human-review-v1',queue_sha256=digest(q),records=[dict(d,internal_partition='optimization')]);self.assertRaises(ValueError,validate_submission,p,q,manifest()['rows'])
        p['records']=[dict(d,decision='invented')];self.assertRaises(ValueError,validate_submission,p,q,manifest()['rows'])

if __name__=='__main__':unittest.main()
