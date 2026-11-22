"""CPU-only synthetic mathematics plus actual cached-data integrity tests."""
import copy,unittest
import torch
from common16 import *
from balanced16 import *
from expanded import xyxy_to_cxcywh


def fixture():
    row=dict(image_id='SYNTHETIC',sha256='SYNTHETIC-SHA',identity_group='SYNTHETIC-GROUP',width=100,height=100,internal_partition='optimization',historical_partition='train',annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
    e=[dict(candidate_id='SYNTHETIC-R',image_id=row['image_id'],identity_group=row['identity_group'],category='chair',source_photo_sha256=row['sha256'],source_verified_under_policy=True,human_reviewed=False,box=[.05,.05,.3,.3])]
    z=torch.full((3,7),-4.,requires_grad=True);b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.1,.1,.2,.2],[.4,.1,.5,.2]],dtype=torch.float32)).requires_grad_();return z,b,row,e,catalog(e,[row])


def grad_maps(p=1.,n=-10.):
    base={k:torch.ones(2,dtype=torch.float32) for k in sorted(HEAD_KEYS)};pos={k:torch.ones(2)*p for k in sorted(CLASS_KEYS)};neg={k:torch.ones(2)*n for k in sorted(CLASS_KEYS)};return base,pos,neg


class NormalizationTests(unittest.TestCase):
    def test_disabled_objective_bit_identical_loss_and_gradients(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[],c);old,_,_=baseline_objective(z,b,r,e,[],True);self.assertTrue(torch.equal(q['baseline_total'],old));a=torch.autograd.grad(old,(z,b),retain_graph=True);d=torch.autograd.grad(q['baseline_total'],(z,b));self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,d)));self.assertFalse(q['enabled'])
    def test_region_mean_and_positive_preservation(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[],c,Policy(enabled=True));self.assertEqual(float(q['weights'].sum()),1.);self.assertTrue(torch.equal(q['source_raw'],.75*torch.nn.functional.binary_cross_entropy_with_logits(z[1,1],torch.tensor(0.))));self.assertEqual(q['normalization']['group_region_count'],1)
    def test_ignored_and_other_six_classes_zero_source_gradient(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[],c,Policy(enabled=True));g=torch.autograd.grad(q['source_raw'],z)[0];self.assertEqual(int(torch.count_nonzero(g)),1);self.assertGreater(float(g[1,1]),0);self.assertFalse(g[0].any());self.assertFalse(g[2].any())
    def test_duplicate_region_aliases_do_not_increase_loss(self):
        z,b,r,e,c=fixture();dupe=dict(e[0],candidate_id='ALIAS');cc=catalog(e+[dupe,e[0]],[r]);self.assertEqual(cc['unique_supervision_cells'],1);a=branches(z,b,r,e,[],c,Policy(enabled=True));bb=branches(z,b,r,e,[],cc,Policy(enabled=True));self.assertTrue(torch.equal(a['source_raw'],bb['source_raw']));self.assertEqual(len(cc['photos'][r['image_id']][0]['aliases']),2)
    def test_conflicting_duplicate_region_ID_rejected(self):
        z,b,r,e,c=fixture()
        with self.assertRaises(ValueError):catalog(e+[dict(e[0],box=[.1,.1,.4,.4])],[r])
    def test_exact_duplicate_queries_leave_loss_and_output_mass_unchanged(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[],c,Policy(enabled=True));zz=torch.cat([z.detach(),z.detach()[1:2]]).requires_grad_();bb=torch.cat([b.detach(),b.detach()[1:2]]).requires_grad_();qq=branches(zz,bb,r,e,[],c,Policy(enabled=True));self.assertTrue(torch.equal(q['source_raw'],qq['source_raw']));self.assertEqual(qq['normalization']['duplicate_query_signatures_collapsed'],1);a=torch.autograd.grad(q['source_raw'],z)[0];d=torch.autograd.grad(qq['source_raw'],zz)[0];self.assertAlmostEqual(float(a.abs().sum()),float(d.abs().sum()),places=7)
    def test_overlapping_regions_assign_query_once(self):
        z,b,r,e,c=fixture();ee=e+[dict(e[0],candidate_id='SECOND',box=[.06,.06,.29,.29])];cc=catalog(ee,[r]);q=branches(z,b,r,ee,[],cc,Policy(enabled=True));self.assertEqual(sum(len(x['query_indices']) for x in q['normalization']['regions']),1);self.assertAlmostEqual(float(q['weights'].sum()),.5)
    def test_uncovered_regions_stay_zero_and_stay_in_denominator(self):
        z,b,r,e,c=fixture();ee=e+[dict(e[0],candidate_id='EMPTY',box=[.6,.1,.65,.2])];cc=catalog(ee,[r]);q=branches(z,b,r,ee,[],cc,Policy(enabled=True));self.assertEqual(float(q['weights'].sum()),.5);self.assertEqual(sum(x['covered_signatures']==0 for x in q['normalization']['regions']),1)
    def test_no_coverage_source_loss_and_gradient_zero(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[[0,0,.3,.3]],c,Policy(enabled=True));self.assertEqual(float(q['source_raw']),0);self.assertFalse(torch.autograd.grad(q['source_raw'],z)[0].any())
    def test_group_normalization_spans_photographs_not_just_minibatch(self):
        z,b,r,e,c=fixture();row2=dict(r,image_id='SYNTHETIC-2');e2=dict(e[0],image_id=row2['image_id'],candidate_id='R2');cc=catalog(e+[e2],[r,row2]);q=branches(z,b,r,e,[],cc,Policy(enabled=True));self.assertEqual(q['normalization']['group_region_count'],2);self.assertEqual(float(q['weights'].sum()),.5)
    def test_new_source_divisor_is_not_positive_box_count(self):
        z,b,r,e,c=fixture();a=branches(z,b,r,e,[],c,Policy(enabled=True));r2=copy.deepcopy(r);r2['annotations'].append(dict(label='sofa',box=[.4,.1,.5,.2],ignore=False));q=branches(z,b,r2,e,[],c,Policy(enabled=True));self.assertTrue(torch.equal(a['source_raw'],q['source_raw']));self.assertEqual(q['stats']['positive_boxes'],2)
    def test_protected_chair_or_stool_preserved(self):
        z,b,r,e,c=fixture();q=branches(z,b,r,e,[[.08,.08,.24,.24]],c,Policy(enabled=True));self.assertFalse(q['masks']['added'].any());self.assertEqual(float(q['source_raw']),0)
    def test_deterministic_region_order(self):
        z,b,r,e,c=fixture();ee=e+[dict(e[0],candidate_id='EMPTY',box=[.6,.1,.65,.2])];a=branches(z,b,r,ee,[],catalog(ee,[r]),Policy(enabled=True));q=branches(z,b,r,list(reversed(ee)),[],catalog(list(reversed(ee)),[r]),Policy(enabled=True));self.assertTrue(torch.equal(a['weights'],q['weights']))
    def test_nonfinite_and_unverified_sources_rejected(self):
        z,b,r,e,c=fixture()
        with self.assertRaises(ValueError):catalog([dict(e[0],source_verified_under_policy=False)],[r])
        with torch.no_grad():z[0,0]=float('nan')
        with self.assertRaises(ValueError):branches(z,b,r,e,[],c,Policy(enabled=True))
    def test_validation_source_cannot_enter_catalog(self):
        z,b,r,e,c=fixture()
        with self.assertRaises(PermissionError):catalog(e,[dict(r,internal_partition='inner_development')])
    def test_extreme_logits_stable(self):
        for value in (-100.,100.):
            z,b,r,e,c=fixture()
            with torch.no_grad():z[1,1]=value
            q=branches(z,b,r,e,[],c,Policy(enabled=True));self.assertTrue(torch.isfinite(q['source_raw']));self.assertTrue(torch.isfinite(torch.autograd.grad(q['source_raw'],z)[0]).all())


class CapTests(unittest.TestCase):
    def test_disabled_and_incomplete_C11_scope_rejected(self):
        base,p,n=grad_maps()
        with self.assertRaises(PermissionError):compose(base,p,n,1)
        with self.assertRaises(ValueError):compose(base,{k:v for k,v in p.items() if 'dense0' in k},n,1,Policy(enabled=True))
    def test_cap_only_negative_preserves_base_and_box_buffers(self):
        base,p,n=grad_maps();snapshot={k:v.clone() for k,v in base.items()};out,d=compose(base,p,n,1,Policy(enabled=True));self.assertLessEqual(d['used_negative_norm'],.25*d['positive_norm']*(1+1e-6));self.assertTrue(all(torch.equal(base[k],snapshot[k]) for k in base));self.assertTrue(all(torch.equal(out[k],base[k]) for k in base if k.startswith('box_head.')))
    def test_strong_conflict_preserves_positive_plus_source_direction(self):
        base,p,n=grad_maps();out,d=compose(base,p,n,1,Policy(enabled=True));dot=sum(float((p[k].double()*(p[k]+n[k]*d['scale']).double()).sum()) for k in p);self.assertGreaterEqual(dot,.75*d['positive_norm']**2-1e-5)
    def test_absent_chair_positives_do_not_disable_furniture_anchor(self):
        base,p,n=grad_maps();out,d=compose(base,p,n,2,Policy(enabled=True));self.assertGreater(d['used_negative_norm'],0);self.assertEqual(d['mode'],'relative-to-all-verified-furniture-positives')
    def test_verified_negative_only_budget_is_nonzero_and_absolute(self):
        base,p,n=grad_maps(p=0);out,d=compose(base,p,n,0,Policy(enabled=True));self.assertAlmostEqual(d['used_negative_norm'],.01,places=7);self.assertTrue(d['mode'].startswith('verified-negative-only'))
    def test_nearzero_known_positive_anchor_stops_without_dropping_evidence(self):
        base,p,n=grad_maps(p=1e-12)
        with self.assertRaises(PositiveAnchorUnavailable):compose(base,p,n,1,Policy(enabled=True))
    def test_no_negative_gradient_has_no_effect(self):
        base,p,n=grad_maps(n=0);out,d=compose(base,p,n,1,Policy(enabled=True));self.assertEqual(d['scale'],1);self.assertTrue(all(torch.equal(out[k],base[k]) for k in out))
    def test_nonfinite_parameter_gradient_rejected(self):
        base,p,n=grad_maps();n[next(iter(n))][0]=float('inf')
        with self.assertRaises(ValueError):compose(base,p,n,1,Policy(enabled=True))
    def test_actual_toy_parameter_gradients_not_just_output_gradients(self):
        # Synthetic tensor graph: six class tensors contribute to the same logit.
        params={k:torch.tensor([.1,.2],requires_grad=True) for k in sorted(CLASS_KEYS)};v=sum(x.sum() for x in params.values());lp=F.softplus(-v);ln=F.softplus(v);p=dict(zip(params,torch.autograd.grad(lp,tuple(params.values()),retain_graph=True)));n=dict(zip(params,torch.autograd.grad(ln,tuple(params.values()))));base={k:(p[k] if k in p else torch.zeros(2)) for k in sorted(HEAD_KEYS)};out,d=compose(base,p,n,1,Policy(enabled=True));self.assertLessEqual(d['used_negative_norm'],.25*d['positive_norm']*(1+1e-6));self.assertTrue(all(x.grad is None for x in params.values()))
    def test_cap_after_accumulation_handles_cancelling_positive_directions(self):
        acc=Accumulator(2,128.)
        for sign in (1.,-.99):
            b,p,n=grad_maps(p=sign,n=2.);acc.add({k:v*64 for k,v in b.items()},{k:v*64 for k,v in p.items()},{k:v*64 for k,v in n.items()},1)
        result,d=acc.finish(Policy(enabled=True));self.assertLessEqual(d['used_negative_norm'],.25*d['positive_norm']*(1+1e-6));self.assertLess(d['positive_norm'],.1);self.assertEqual(d['actual_microbatches'],2)
    def test_last_three_microbatch_order_and_base_preservation(self):
        acc=Accumulator(3,128.)
        for _ in range(3):
            b,p,n=grad_maps(n=0);acc.add({k:v*(128/3) for k,v in b.items()},{k:v*(128/3) for k,v in p.items()},{k:v*(128/3) for k,v in n.items()},1)
        out,d=acc.finish(Policy(enabled=True));self.assertTrue(all(torch.allclose(v,torch.ones(2)) for v in out.values()));self.assertEqual(d['positive_targets'],3)
    def test_incomplete_or_extra_accumulation_fails_closed(self):
        acc=Accumulator(1,128.)
        with self.assertRaises(ValueError):acc.finish(Policy(enabled=True))
        b,p,n=grad_maps();acc.add(b,p,n,1)
        with self.assertRaises(ValueError):acc.add(b,p,n,1)
    def test_parameter_key_order_does_not_change_cap_or_results(self):
        b,p,n=grad_maps()
        for i,k in enumerate(sorted(p)):p[k]=p[k]*(i+1);n[k]=n[k]*(i+.3)
        a,d=compose(b,p,n,1,Policy(enabled=True));q,e=compose(dict(reversed(list(b.items()))),dict(reversed(list(p.items()))),dict(reversed(list(n.items()))),1,Policy(enabled=True));self.assertEqual(d,e);self.assertTrue(all(torch.equal(a[k],q[k]) for k in a))


class ActualEvidenceTests(unittest.TestCase):
    def test_real_source_masks_and_coverage_unchanged(self):
        d=read(REPORT16/'comparison.json');self.assertTrue(d['ignored_and_positive_masks_unchanged']);self.assertEqual(d['source_queries'],281);self.assertIsNone(d['parameter_space_limitations']['new_balanced_real_model_parameter_norm'])
    def test_future_config_preserves_C8_calibration_and_original_scope(self):
        from spec16 import validate_spec
        s=read(REPORT16/'future-experiment-disabled.json');self.assertTrue(validate_spec(s));self.assertEqual(len(s['trainable_keys']),12);self.assertEqual(len(s['class_gradient_keys']),6);self.assertFalse(s['calibration_keys_frozen']);self.assertFalse(s['L2SP_enabled']);self.assertEqual(s['initialization']['head_delta'],None)
    def test_no_CUDA_initialized_by_CPU_suite(self):self.assertFalse(torch.cuda.is_initialized())
    def test_original_C8_checkpoint_load_interface_and_all12_scope(self):
        from test_preflight10 import fixture as head_fixture
        from expanded import configure_heads,load_delta
        model=head_fixture();configure_heads(model);self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},HEAD_KEYS);self.assertTrue(all(dict(model.named_parameters())[n].requires_grad for n in CLASS_KEYS));done=read(C8/'B-corrected/complete.json');path=C8/done['final_checkpoint']['file'];before=sha(path);payload=torch.load(path,map_location='cpu',weights_only=True);load_delta(model,payload);self.assertEqual(sha(path),before);self.assertTrue(all(torch.equal(dict(model.named_parameters())[n].detach(),payload['weights'][n]) for n in HEAD_KEYS))


if __name__=='__main__':unittest.main()
