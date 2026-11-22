"""CPU mathematical, floating-point, corruption and preservation regressions."""
import math
import unittest
from unittest.mock import patch
import torch
from common19 import *
import guard19
from guard19 import compose_checked,verify_direct,norm64,tolerance_for,CapViolation
from balanced16 import CLASS_KEYS,Accumulator,Policy,compose,PositiveAnchorUnavailable
from measure17 import finish_accumulator


def fixture(base=1.,positive=1.,negative=-10.,M=1):
    b={k:torch.full((4,),base,dtype=torch.float32) for k in sorted(HEAD_KEYS)}
    p={k:torch.full((4,),positive,dtype=torch.float32) for k in sorted(CLASS_KEYS)}
    n={k:torch.full((4,),negative,dtype=torch.float32) for k in sorted(CLASS_KEYS)}
    a=Accumulator(M,128)
    for _ in range(M):a.add({k:v*(128/M) for k,v in b.items()},{k:v*(128/M) for k,v in p.items()},{k:v*(128/M) for k,v in n.items()},1)
    return a


class GuardTests(unittest.TestCase):
    def test_C18_false_failure_repaired_without_changed_output(self):
        from update18 import compose_checked as old
        a=fixture(base=64,positive=.001)
        with self.assertRaisesRegex(ValueError,'Measured cap violation'):old(a)
        expected,cap=finish_accumulator(a);out,m=compose_checked(a)
        self.assertTrue(all(torch.equal(out[k],expected[k]) for k in out));self.assertTrue(m['direct_guard']['cap_verified']);self.assertLess(cap['scale'],1)
    def test_tiny_valid_positive_anchors(self):
        for p in (1e-7,1e-5,.001):
            a=fixture(base=100,positive=p,negative=-1e3);out,m=compose_checked(a)
            self.assertLessEqual(m['source_post']['norm'],m['cap']['budget']+m['direct_guard']['numerical_tolerance']['total'])
    def test_large_gradients_with_finite_FP64_reference(self):
        for p,n in ((1e15,-1e25),(1e30,-1e32)):
            out,m=compose_checked(fixture(base=1e25,positive=p,negative=n));self.assertTrue(m['direct_guard']['cap_verified']);self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
    def test_mixed_magnitudes_and_cancellation(self):
        a=fixture()
        for index,k in enumerate(sorted(CLASS_KEYS)):
            a.base[k]=torch.tensor([1e20,64.,-1e20,1e-20])*128
            a.positive[k]=torch.tensor([1e10,.001,1e-6,1e-15])*128
            a.negative[k]=torch.tensor([-1e15,-1e5,1e-10,-1e-20])*128
        expected,_=finish_accumulator(a);out,m=compose_checked(a)
        self.assertTrue(all(torch.equal(out[k],expected[k]) for k in out));self.assertTrue(m['direct_guard']['cap_verified'])
    def test_normal_and_subnormal_vector_entries_supported(self):
        a=fixture()
        for k in CLASS_KEYS:a.negative[k]=torch.tensor([1.,1e-38,1e-44,0.])*128
        out,m=compose_checked(a);self.assertTrue(m['direct_guard']['cap_verified'])
    def test_consistent_scope_norm_vs_math_fsum(self):
        values={k:torch.tensor([1e20,1.,1e-20,-1e18],dtype=torch.float32) for k in CLASS_KEYS}
        reference=math.sqrt(math.fsum(float(x)**2 for v in values.values() for x in v.tolist()))
        self.assertTrue(math.isclose(norm64(values),reference,rel_tol=5e-15))
    def test_genuine_cap_violation_not_excused(self):
        p={k:torch.ones(2) for k in CLASS_KEYS};n={k:torch.full((2,),100.) for k in CLASS_KEYS};post={k:torch.full((2,),.25001) for k in CLASS_KEYS}
        with self.assertRaisesRegex(CapViolation,'exceeds derived cap'):verify_direct(p,n,post,.0025,1)
    def test_wrong_coefficient_or_small_but_corrupt_component_stops(self):
        p={k:torch.ones(2) for k in CLASS_KEYS};n={k:torch.full((2,),100.) for k in CLASS_KEYS};post={k:torch.full((2,),.25) for k in CLASS_KEYS}
        with self.assertRaisesRegex(CapViolation,'coefficient'):verify_direct(p,n,post,.005,1)
        post={k:torch.full((2,),.1) for k in CLASS_KEYS}
        with self.assertRaisesRegex(CapViolation,'exact original'):verify_direct(p,n,post,.0025,1)
    def test_legitimate_boundary_is_accepted(self):
        out,m=compose_checked(fixture(negative=-4));self.assertAlmostEqual(m['source_post']['negative_positive_ratio'],.25,places=6)
    def test_derived_tolerance_is_not_an_empirical_threshold(self):
        a=tolerance_for(1,24,.5);b=tolerance_for(1,1000000,.5)
        self.assertAlmostEqual(a['unit_roundoff_FP32'],2**-24);self.assertLess(a['relative'],1.3e-7);self.assertGreater(b['relative'],a['relative']);self.assertEqual(a['scalar_conversion_and_multiply_roundings'],2)
        self.assertAlmostEqual(tolerance_for(4,24,.5)['total'],4*a['total'],delta=1e-30)
    def test_underresolved_coefficient_stops_instead_of_widening_tolerance(self):
        with self.assertRaisesRegex(ValueError,'resolution'):tolerance_for(1,24,1e-44)
    def test_tolerance_matches_high_precision_analytic_formula(self):
        from decimal import Decimal,localcontext
        with localcontext() as context:
            context.prec=70
            for count in (24,395266,1000000):
                d=tolerance_for(1,count,.5);u32=Decimal(2)**-24;u64=Decimal(2)**-53;n=Decimal(d['reduction_operation_bound']);gamma=n*u64/(1-n*u64)
                reference=(1+u32)**2*(1+gamma)**2*(1+u64)**4/(1-gamma)-1
                self.assertTrue(math.isclose(d['relative'],float(reference),rel_tol=2e-15))
    def test_no_negative_gradients_preserve_entire_base(self):
        a=fixture(negative=0);expected={k:v/128 for k,v in a.base.items()};out,m=compose_checked(a)
        self.assertTrue(all(torch.equal(out[k],expected[k]) for k in out));self.assertEqual(m['cap']['scale'],1)
    def test_unusable_positive_anchor_stops(self):
        for value in (0.,1e-12):
            with self.assertRaises(PositiveAnchorUnavailable):compose_checked(fixture(positive=value))
    def test_negative_only_absolute_budget_unchanged(self):
        a=fixture(positive=0);a.positive_count=0;out,m=compose_checked(a);self.assertAlmostEqual(m['source_post']['norm'],.01,places=8)
    def test_positive_and_box_tensors_bitwise_preserved(self):
        a=fixture(M=3);p={k:v.clone() for k,v in a.positive.items()};b={k:v.clone() for k,v in a.base.items()};out,m=compose_checked(a)
        self.assertTrue(all(torch.equal(v,a.positive[k]) for k,v in p.items()));self.assertTrue(all(torch.equal(v,a.base[k]) for k,v in b.items()))
        self.assertTrue(all(torch.equal(out[k],b[k]/128) for k in out if k.startswith('box_head.')))
    def test_corrupt_combined_or_box_vector_is_detected(self):
        original=guard19.finish_accumulator
        for name in ('class_head.dense0.weight','box_head.dense0.weight'):
            def corrupt(a):
                out,cap=original(a);out[name]=out[name]+1;return out,cap
            with patch.object(guard19,'finish_accumulator',corrupt):
                with self.assertRaisesRegex(CapViolation,'corruption'):compose_checked(fixture())
    def test_nonfinite_and_wrong_parameter_scope_stops(self):
        a=fixture();a.negative[next(iter(CLASS_KEYS))][0]=float('inf')
        with self.assertRaises(ValueError):compose_checked(a)
        p={k:torch.ones(2) for k in CLASS_KEYS};n=dict(p);n.pop(next(iter(n)))
        with self.assertRaises(ValueError):verify_direct(p,n,p,1,1)
    def test_source_group_duplicate_normalization_and_protected_targets_unchanged(self):
        from test_balanced16 import fixture as source_fixture
        from balanced16 import catalog,branches
        z,b,row,e,ctx=source_fixture();first=branches(z,b,row,e,[],ctx,Policy(enabled=True));alias=dict(e[0],candidate_id='SYNTHETIC-ALIAS');other=branches(z,b,row,e+[alias],[],catalog(e+[alias],[row]),Policy(enabled=True))
        self.assertTrue(torch.equal(first['source_raw'],other['source_raw']));self.assertTrue(torch.equal(first['masks']['added'],other['masks']['added']))
        protected=branches(z,b,row,e,[[0,0,.3,.3]],ctx,Policy(enabled=True));self.assertFalse(protected['masks']['added'].any())
    def test_guard_does_not_initialize_CUDA(self):self.assertFalse(torch.cuda.is_initialized())
    def test_actual_full_parameter_shapes_at_active_cap(self):
        reference=original_heads();generator=torch.Generator().manual_seed(1901)
        base={k:torch.full_like(v,64.) for k,v in reference.items()}
        positive={k:torch.randn(reference[k].shape,generator=generator)*.001 for k in CLASS_KEYS}
        negative={k:torch.randn(reference[k].shape,generator=generator)*10 for k in CLASS_KEYS}
        a=Accumulator(1,128);a.add({k:v*128 for k,v in base.items()},{k:v*128 for k,v in positive.items()},{k:v*128 for k,v in negative.items()},1)
        expected,_=finish_accumulator(a);out,m=compose_checked(a)
        self.assertTrue(all(torch.equal(out[k],expected[k]) for k in out));self.assertTrue(m['direct_guard']['cap_verified']);self.assertLess(m['cap']['scale'],1)


if __name__=='__main__':unittest.main()
