"""CPU contracts and synthetic tensor regression tests,not model improvement evidence."""
import copy,tempfile,unittest,types
from unittest.mock import patch
import numpy as np
import torch
from commonE import *
from teacherE import load_teacher,verify_alignment
from objectiveE import selective_mask,distill,objective,base_objective
from scopeE import StudentScope,TRAIN_KEYS,CALIBRATION_KEYS,original_heads,frozen_state
from configE import require_execution,TRAIN_LIMITS,SMOKE_LIMITS
sys.path.insert(0,str(ROOT/'tools/phase10c10-v1'))
from test_preflight10 import fixture

def data():
    row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
    from expanded import xyxy_to_cxcywh
    b=xyxy_to_cxcywh(torch.tensor([[.69,.69,.91,.91],[.1,.1,.2,.2],[.4,.4,.5,.5]]))
    z=torch.full((3,7),-5.);z[0,1]=2.
    e=dict(row=row,source_regions=[dict(candidate_id='TEST',image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],source_verified_under_policy=True)],legacy_protected=[],distill_protected=[])
    e['distill_regions']=copy.deepcopy(e['source_regions'])
    return row,z,b,e

class MaskLossTests(unittest.TestCase):
    def setUp(self):self.row,self.tz,self.tb,self.e=data();self.z=self.tz.clone().requires_grad_();self.b=self.tb.clone().requires_grad_()
    def mask(self,**kw):return selective_mask(self.z,self.b,kw.get('tz',self.tz),kw.get('tb',self.tb),self.row,kw.get('regions',self.e['distill_regions']),kw.get('protected',[]))[0]
    def test_only_supported_chair_query_not_seven_classes(self):
        m=self.mask();self.assertEqual(m.sum(),1);self.assertTrue(m[1,1]);self.assertFalse(m[:,[0,2,3,4,5,6]].any())
    def test_low_teacher_confidence_not_absence(self):self.assertFalse(self.mask(regions=[]).any())
    def test_protected_unknown_and_ignored_block(self):
        self.assertFalse(self.mask(protected=[[0,0,.4,.4]]).any())
        self.row['annotations'].append(dict(label='lamp',ignore=True,box=[0,0,.4,.4]));self.assertFalse(self.mask().any())
    def test_teacher_negative_conflict_not_copied(self):
        z=self.tz.clone();z[1,1]=2
        m,s=selective_mask(self.z,self.b,z,self.tb,self.row,self.e['distill_regions'],[])
        self.assertFalse(m.any());self.assertEqual(s['teacher_conflicts_excluded'],1)
    def test_positive_chairs_have_priority_even_teacher_wrong(self):
        self.e['distill_regions'][0]['box']=[0,0,1,1]
        z=self.tz.clone();z[0,1]=-20
        m=self.mask(tz=z);self.assertFalse(m[0].any())
    def test_both_box_containment_not_center(self):
        b=self.tb.clone();b[1,2]=.7
        self.assertFalse(self.mask(tb=b).any())
        with torch.no_grad():self.b[1,2]=.7
        self.assertFalse(self.mask().any())
    def test_teacher_targets_cannot_receive_gradients(self):
        self.assertRaises(ValueError,self.mask,tz=self.tz.clone().requires_grad_())
    def test_identical_loss_and_gradient_zero(self):
        l=distill(self.z,self.tz,self.mask());self.assertEqual(float(l.detach()),0)
        g=torch.autograd.grad(l,self.z)[0];self.assertFalse(g.any())
    def test_controlled_deviation_and_normalization(self):
        m=self.mask();z=self.z+2*m
        loss=distill(z,self.tz,m);self.assertEqual(float(loss.detach()),4.)
        grad=torch.autograd.grad(loss,self.z)[0];self.assertAlmostEqual(float(grad[1,1]),4)
        self.assertEqual(int((grad!=0).sum()),1)
        self.assertEqual(float(distill(z.repeat(2,1),self.tz.repeat(2,1),m.repeat(2,1)).detach()),4.)
    def test_empty_mask_finite_zero_connected(self):
        m=torch.zeros_like(self.z,dtype=torch.bool);l=distill(self.z,self.tz,m)
        self.assertEqual(float(l.detach()),0);self.assertFalse(torch.autograd.grad(l,self.z)[0].any())
    def test_distillation_routes_no_box_gradient(self):
        m=self.mask();l=distill(self.z+2*m,self.tz,m)
        g=torch.autograd.grad(l,(self.z,self.b),allow_unused=True);self.assertIsNone(g[1])
    def test_unknown_class_drift_not_claimed_constrained(self):
        shift=torch.zeros_like(self.z);shift[:,6]=10
        self.assertEqual(float(distill(self.z+shift,self.tz,self.mask()).detach()),0)
    def test_shape_class_and_nonfinite_rejected(self):
        self.assertRaises(ValueError,selective_mask,self.z[:,:6],self.b,self.tz,self.tb,self.row,self.e['distill_regions'],[])
        bad=self.tz.clone();bad[0,0]=float('nan');self.assertRaises(ValueError,self.mask,tz=bad)
        m=self.mask();m[:,2]=True;self.assertRaises(ValueError,distill,self.z,self.tz,m)
    def test_zero_coefficient_replays_existing_positive_and_box_gradients(self):
        old,_,_=base_objective(self.z,self.b,self.row,self.e['source_regions'],[],True)
        new,parts,_=objective(self.z,self.b,self.row,self.tz,self.tb,self.e,0.)
        self.assertTrue(torch.equal(old,new))
        a=torch.autograd.grad(old,(self.z,self.b),retain_graph=True);b=torch.autograd.grad(new,(self.z,self.b))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
    def test_missing_coefficient_and_fake_source_rejected(self):
        self.assertRaises(ValueError,objective,self.z,self.b,self.row,self.tz,self.tb,self.e,None)
        self.e['distill_regions'][0]['source_verified_under_policy']=False;self.assertRaises(ValueError,self.mask)

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index=read(PREP/'teacher-index.json');cls.rows={e['row']['image_id']:e['row'] for e in read(PREP/'training-context.json')['entries']}
    def test_exact59_and_training_only_hash_alignment(self):
        self.assertEqual(len(self.index['records']),59)
        self.assertEqual(len({r['source_group'] for r in self.index['records']}),56)
        for r in self.index['records']:
            teacher=load_teacher(r,self.rows[r['image_id']],self.index['processor_contract_sha256'])
            self.assertEqual(teacher.logits.shape,(3600,7));self.assertEqual(teacher.boxes.shape,(3600,4))
    def test_immutability_and_detached_tensor_copy(self):
        r=self.index['records'][0];t=load_teacher(r,self.rows[r['image_id']],self.index['processor_contract_sha256'])
        with self.assertRaises(ValueError):t.logits[0,0]=1
        before=t.logits.copy();z,b=t.tensors();z[0,0]=99
        np.testing.assert_array_equal(t.logits,before);self.assertFalse(z.requires_grad);self.assertFalse(b.requires_grad)
    def test_all_correspondence_fields_fail_closed(self):
        r=self.index['records'][0];row=self.rows[r['image_id']]
        for key,bad in [('source_sha256','0'*64),('source_group','wrong'),('width',1),('precision','FP32'),('query_layout',[30,30]),('augmentation','flip'),('prompts',PROMPTS[::-1]),('classes',list(CLASSES)[::-1]),('processor_contract_sha256','bad'),('internal_partition','inner_development')]:
            q=copy.deepcopy(r);q[key]=bad
            self.assertRaises((ValueError,PermissionError),verify_alignment,q,row,self.index['processor_contract_sha256'])
    def test_array_hash_tampering_rejected(self):
        r=copy.deepcopy(self.index['records'][0]);r['cache_sha256']='0'*64
        self.assertRaises(ValueError,load_teacher,r,self.rows[r['image_id']],self.index['processor_contract_sha256'])
    def test_no_training_development_group_overlap(self):
        d=read(PREP/'training-context.json');train=[e['row'] for e in d['entries']]
        for k in ('image_id','sha256','identity_group'):self.assertFalse({r[k] for r in train}&{r[k] for r in d['development']})

class ScopeGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.model=fixture(original_heads());self.model.box_bias=torch.zeros(2,4);self.scope=StudentScope(self.model);self.opt=self.scope.optimizer()
    def test_exact_eight_and_projection_trainable(self):
        self.assertEqual({n for n,p in self.model.named_parameters() if p.requires_grad},TRAIN_KEYS)
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),1577988)
        self.assertTrue(self.model.class_head.dense0.weight.requires_grad)
        self.assertTrue(all(not dict(self.model.named_parameters())[n].requires_grad for n in CALIBRATION_KEYS))
    def test_mode_transitions_and_illegal_reenable(self):
        self.scope.eval_mode();self.scope.train_mode();self.scope.check(self.opt,hash_frozen=True)
        self.model.class_head.logit_shift.weight.requires_grad_(True);self.assertRaises(ValueError,self.scope.train_mode)
    def test_frozen_values_buffers_grid_protected(self):
        self.model.box_bias[0,0]=1;self.assertRaises(ValueError,self.scope.check,self.opt,True)
    def test_checkpoint_compatibility(self):
        p=self.scope.payload(self.opt,0,'a'*64);self.scope.restore(p,self.opt,'a'*64)
        self.assertRaises(ValueError,self.scope.restore,p,self.opt,'b'*64)
        p['immutable_state']='wrong';self.assertRaises(ValueError,self.scope.restore,p)
    def test_optimizer_excludes_teacher_and_calibration(self):
        ids={id(p) for g in self.opt.param_groups for p in g['params']}
        self.assertEqual(len(ids),8)
        for n in CALIBRATION_KEYS:self.assertNotIn(id(dict(self.model.named_parameters())[n]),ids)
    def test_gate_denies_before_runtime_dirs_or_cuda(self):
        import workerE,launchE
        self.assertRaises(PermissionError,workerE.run,{},'training')
        self.assertRaises(PermissionError,workerE.run,{},'gradient_smoke')
        self.assertFalse(FUTURE.exists());self.assertFalse(SMOKE.exists());self.assertFalse(torch.cuda.is_initialized())
    def test_default_proposals_do_not_invent_strength(self):
        for name in ('gpu-training-disabled.json','gradient-smoke-disabled.json'):
            d=read(REPORT/name);self.assertFalse(d['approved']);self.assertFalse(d['enabled']);self.assertIsNone(d['coefficient'])
        self.assertEqual(TRAIN_LIMITS['forwards'],59+295+19);self.assertEqual(SMOKE_LIMITS['updates'],0)
    def test_every_prepared_module_compiles(self):
        for path in HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
    def test_execution_and_evaluation_imports_are_torch_free(self):
        code='import workerE,launchE,evaluateE; import sys; assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
