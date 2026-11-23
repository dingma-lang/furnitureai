"""CPU-only authorization and measured-gradient preflight for the future candidate."""
import ast
import unittest
from common4 import *
from future4 import gate,worker,launch,FUTURE


class FutureTests(unittest.TestCase):
    def test_absent_training_authorization_blocks_before_output_or_torch(self):
        for approval in (None,{},dict(approved=True,mode='gradient_smoke')):
            self.assertRaises(PermissionError,worker,approval)
            self.assertRaises(PermissionError,launch,approval)
        self.assertFalse(FUTURE.exists())
        code='import future4;import sys;assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_one_arm_and_bounds(self):
        plan=read(REPORT/'training-disabled.json')
        self.assertFalse(plan['approved']);self.assertFalse(plan['enabled']);self.assertFalse(plan['training_authorized'])
        self.assertEqual(plan['configurations'],1);self.assertEqual(plan['learning_rate'],1e-5)
        self.assertEqual(plan['limits']['updates'],75);self.assertEqual(plan['limits']['epochs'],5)
        self.assertEqual(plan['limits']['seconds'],600);self.assertEqual(len(plan['trainable_keys']),8)
        self.assertEqual(plan['training_photos'],59);self.assertEqual(plan['positive_boxes'],182)
        self.assertEqual(plan['precision']['loss_scale'],1.);self.assertFalse(plan['precision']['GradScaler'])

    def test_proposed_coefficient_non_domination_on_all_actual_groups(self):
        plan=read(REPORT/'training-disabled.json');details=read(REPORT/'per-photo.json')['records']
        self.assertEqual(len(details),15)
        for row in details:
            self.assertLessEqual(plan['coefficient']*row['unit_KD_L2']/row['positive_L2'],.5+1e-12)
            self.assertGreaterEqual(row['positive_descent_fraction'],.5)
        self.assertEqual(plan['coefficient'],read(REPORT/'gradient-findings.json')['coefficient'])

    def test_pinned_files_and_original_sources(self):
        plan=read(REPORT/'training-disabled.json')
        for rel,h in plan['files'].items():self.assertEqual(sha(ROOT/rel),h,rel)
        self.assertEqual(sha(BASE/'model.safetensors'),BASE_SHA)

    def test_wrong_training_permit_limits_or_coefficient_rejected(self):
        path=REPORT/'training-disabled.json';plan=read(path)
        a=dict(approved=True,mode='teacher_distillation_training',proposal_sha256=sha(path),limits=plan['limits'],coefficient=plan['coefficient'])
        bad=dict(a,coefficient=1.);self.assertRaises(PermissionError,gate,bad)
        bad=dict(a,limits={});self.assertRaises(PermissionError,gate,bad)

    def test_future_has_explicit_scale_and_cpu_syntax(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
        text=(HERE/'future4.py').read_text()
        self.assertNotIn('GradScaler(',text);self.assertNotIn('loss*128',text)
        self.assertIn('torch.autograd.grad(loss,params)',text)
        self.assertIn('(loss/len(group)).backward()',text)


if __name__=='__main__':unittest.main()
