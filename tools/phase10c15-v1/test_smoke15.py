"""CPU-only synthetic loss,gradient isolation,budget and provenance tests."""
import ast,subprocess,unittest,uuid
from unittest.mock import patch
import torch
import common15
from common15 import *
from loss15 import controlled_losses,gradients
from expanded import xyxy_to_cxcywh


def fixture():
    row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
    b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.1,.1,.2,.2],[.4,.1,.6,.3]],dtype=torch.float32)).requires_grad_()
    z=torch.full((3,7),-4.,requires_grad=True);e=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC-NEGATIVE')]
    return z,b,row,e


class LossTests(unittest.TestCase):
    def test_same_A_masks_positive_and_box_loss(self):
        z,b,r,e=fixture();losses,m,s=controlled_losses(z,b,r,e,[])
        self.assertEqual(s['positive_boxes'],1);self.assertEqual(int(m['added'].sum()),1);self.assertEqual(float(losses['class_A']),s['classification_loss']);self.assertEqual(float(losses['box']),s['box_loss'])
    def test_BCE_only_existing_added_source_terms(self):
        z,b,r,e=fixture();losses,m,s=controlled_losses(z,b,r,e,[]);g=torch.autograd.grad(losses['source_BCE'],z)[0]
        self.assertFalse(g[~m['added']].any());self.assertGreater(float(g[1,1]),0);self.assertEqual(int(torch.count_nonzero(g)),1)
    def test_unknown_high_score_query_stays_ignored(self):
        z,b,r,e=fixture()
        with torch.no_grad():z[2,1]=12
        losses,m,_=controlled_losses(z,b,r,e,[]);g=torch.autograd.grad(losses['class_B'],z)[0];self.assertFalse(m['mask'][2].any());self.assertFalse(g[2].any())
    def test_protected_seating_stool_cannot_be_negative(self):
        z,b,r,e=fixture();losses,m,_=controlled_losses(z,b,r,e,[[.08,.08,.25,.25]])
        self.assertFalse(m['added'].any());self.assertEqual(float(losses['source_BCE']),0)
    def test_known_positive_cannot_receive_source_negative(self):
        z,b,r,e=fixture();e[0]['box']=[0,0,1,1];_,m,_=controlled_losses(z,b,r,e,[]);self.assertFalse((m['added']&m['target'].bool()).any())
    def test_alpha_and_nGT_divisor_preserved(self):
        z,b,r,e=fixture();losses,_,_=controlled_losses(z,b,r,e,[]);self.assertTrue(torch.allclose(losses['source_BCE'],.75*torch.nn.functional.softplus(z[1,1]),atol=1e-7,rtol=1e-6))
    def test_native_negatives_not_replaced(self):
        z,b,r,e=fixture();r['explicit_native_negative']={'sofa':True};losses,m,_=controlled_losses(z,b,r,e,[])
        a=torch.autograd.grad(losses['class_A'],z,retain_graph=True)[0];q=torch.autograd.grad(losses['class_B'],z)[0];self.assertTrue(torch.equal(a[:,0],q[:,0]));self.assertFalse(m['added'][:,0].any())
    def test_source_BCE_loss_and_gradient_stronger_without_weight_search(self):
        z,b,r,e=fixture();losses,_,_=controlled_losses(z,b,r,e,[]);a=torch.autograd.grad(losses['source_focal'],z,retain_graph=True)[0];q=torch.autograd.grad(losses['source_BCE'],z)[0];self.assertGreater(float(q.abs().sum()),100*float(a.abs().sum()))
    def test_parameter_gradients_are_real_and_do_not_accumulate(self):
        z,b,r,e=fixture();features=torch.tensor([[1.,2.],[3.,1.],[2.,4.]]);w=torch.nn.Parameter(torch.ones(2,7)*-.5);zz=features@w;losses,_,_=controlled_losses(zz,b,r,e,[])
        a=gradients(losses['source_focal'],zz,b,[w]);q=gradients(losses['source_BCE'],zz,b,[w]);self.assertIsNone(a[1]);self.assertGreater(float(q[2].norm()),float(a[2].norm()));self.assertIsNone(w.grad);self.assertIsNone(b.grad)
    def test_FP16_not_silently_used_for_loss(self):
        z,b,r,e=fixture()
        with self.assertRaises(ValueError):controlled_losses(z.half(),b,r,e,[])
    def test_nonfinite_loss_input_rejected(self):
        z,b,r,e=fixture()
        with torch.no_grad():z[0,0]=float('nan')
        with self.assertRaises(ValueError):controlled_losses(z,b,r,e,[])
    def test_BCE_numerically_stable_on_extreme_logits(self):
        for value in [-100.,100.]:
            z,b,r,e=fixture()
            with torch.no_grad():z[1,1]=value
            losses,_,_=controlled_losses(z,b,r,e,[]);g=torch.autograd.grad(losses['source_BCE'],z)[0];self.assertTrue(torch.isfinite(g).all())


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.folder=RUN15/('cpu-fixture-'+uuid.uuid4().hex[:8]);cls.folder.mkdir()
    def test_atomic_records_cannot_overwrite(self):
        p=self.folder/'immutable.json'
        with patch.object(common15,'RUN15',self.folder):
            emit(p,dict(synthetic=True))
            with self.assertRaises(FileExistsError):emit(p,dict(synthetic=False))
        self.assertTrue(read(p)['synthetic'])
    def test_output_path_escape_rejected(self):
        with self.assertRaises(PermissionError):emit(ROOT/'escape-never-created.json',{})
    def test_allocation_budget_fail_closed(self):
        with patch.object(common15,'MAX_BYTES',1):
            with self.assertRaises(RuntimeError):emit(self.folder/'limit-never-created.json',{})
    def test_gpu_code_has_no_optimizer_or_backward_call(self):
        tree=ast.parse((HERE15/'gpu15.py').read_text());calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'step','backward','optimizer','zero_grad'});self.assertIn('autograd.grad',(HERE15/'loss15.py').read_text())
    def test_no_inference_or_CUDA_initialization_on_module_import(self):
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c','import gpu15,launch15,torch;assert not torch.cuda.is_initialized()'],cwd=HERE15,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
    def test_exact_approved_training_whitelist_and_evidence(self):
        cfg=read(RUN15/'authorized-config.json');_,train,dev,_,_=training_context();rows=[{r['image_id']:r for r in train}[i] for i in cfg['photo_ids']]
        for row in rows:require_training_row(row,train,dev)
        self.assertEqual(len(rows),24);self.assertEqual(len({r['identity_group'] for r in rows}),24);self.assertTrue(all(r['internal_partition']=='optimization' for r in rows));self.assertFalse(cfg['training_authorized'])
    def test_failed_run_binding_rejects_corrected_code_without_new_approval(self):
        with self.assertRaises(ValueError):inputs15()


class CachedRegressionTests(unittest.TestCase):
    def test_all24_saved_C11_training_outputs_preserve_exact_executed_box_loss(self):
        from common6 import candidates
        _,train,dev,_,reg=training_context();byid={r['image_id']:r for r in train};cfg=read(RUN15/'authorized-config.json');bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
        for i,ident in enumerate(cfg['photo_ids'],1):
            row=byid[ident];a=load_npz(C11_RUN/'probes/epoch-5'/(str(i).zfill(2)+'.npz'));selected=[e for e in bundle['records'] if e['image_id']==ident];protected=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
            losses,m,stats=controlled_losses(torch.from_numpy(a['logits']).requires_grad_(),torch.from_numpy(a['square_cxcywh']).requires_grad_(),row,selected,protected)
            self.assertEqual(float(losses['box']),stats['box_loss']);self.assertEqual(stats['protected_violations'],0)


if __name__=='__main__':unittest.main()
