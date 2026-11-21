"""CPU checks only; no worker launch, real training or model forward calls."""
import copy,unittest
import torch
from common13 import *
from config12 import check_spec,require_training_approval


class ExecutionTests(unittest.TestCase):
    def test_authorized_decimal_is_exactly_same_FP32_coefficient_and_gradient(self):
        cfg=read(ROOT/'reports/phase10c12-v1/experiment-disabled.json');stored=cfg['projection_preservation']['lambda_value']
        self.assertTrue(torch.equal(torch.tensor(stored,dtype=torch.float32),torch.tensor(AUTHORIZED_DECIMAL,dtype=torch.float32)))
        x=torch.tensor([.0001,-.002,.01],requires_grad=True);a=stored*.5*x.square().sum();b=AUTHORIZED_DECIMAL*.5*x.square().sum();self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0]))

    def test_prepared_and_C11_training_and_evaluation_settings_identical(self):
        cfg=read(ROOT/'reports/phase10c12-v1/experiment-disabled.json');old=read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json')
        for key in ('initialization','data','optimization_settings','evaluation','limits','trainable_keys','frozen_calibration_keys'):self.assertEqual(cfg[key],old[key])

    def test_all75_training_orders_match_C11(self):
        from common8 import grouped_epoch
        _,train,dev=verify_inputs13();orders=[g for epoch in range(5) for g in grouped_epoch(train,epoch)]
        self.assertEqual(len(orders),75)
        for i,g in enumerate(orders,1):self.assertEqual([r['image_id'] for r in g],read(C11_RUN/('update-'+str(i).zfill(3)+'.json'))['source_order'])

    def test_C8_C11_and_original_scoring_manifests_exactly_match(self):
        _,_,dev=verify_inputs13()
        for path in (C8/'A-control/evaluation/epoch-0',C8/'B-corrected/evaluation/epoch-5',C11_RUN/'evaluation/epoch-5'):
            raw=read(path/'raw-complete.json');self.assertEqual(raw['rows_sha256'],digest(dev));self.assertTrue(raw['FP32']);self.assertEqual([(r['image_id'],r['source_sha256']) for r in raw['records']],[(r['image_id'],r['sha256']) for r in dev])

    def test_activation_changes_only_recorded_approval_flags(self):
        old=read(ROOT/'reports/phase10c12-v1/experiment-disabled.json');cfg=copy.deepcopy(old);cfg['training_authorized']=True;cfg['projection_preservation']['enabled']=True;cfg['source_configuration'].update(enabled=True,approved=True,approval_reference='Explicit current C13 authorization')
        self.assertTrue(check_spec(cfg))
        with self.assertRaises(PermissionError):require_training_approval(cfg)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
