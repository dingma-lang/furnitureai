"""Read-only checks of the actual safety-stopped C18 attempt."""
import unittest
import torch
from common18 import *


class CompletionTests(unittest.TestCase):
    def test_single_arm_and_exact_counters(self):
        f=read(RUN18/'failure.json');d=f['state']
        self.assertEqual((d['updates'],d['optimizer_hook_calls']),(28,28));self.assertEqual((d['forwards'],d['training_VJPs'],d['probe_VJPs']),(182,345,144));self.assertFalse(f['automatic_retry']);self.assertFalse((RUN18/'complete.json').exists())
        self.assertLess(d['seconds'],600);self.assertLess(d['peak_allocated_bytes'],MAX_VRAM17);self.assertLess(d['peak_reserved_bytes'],MAX_VRAM17);self.assertLess(d['peak_device_bytes'],MAX_VRAM17);self.assertLess(allocation18(),MAX_OUTPUT18)
    def test_all28_order_protection_caps_and_committed_receipts(self):
        cfg,*_=inputs18()
        for i in range(1,29):
            u=read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json'));m=u['metrics']
            self.assertEqual(u['source_order'],cfg['orders'][(i-1)//15][(i-1)%15]);self.assertEqual(m['cap_violations'],0);self.assertTrue(m['positive_buffers_unchanged']);self.assertTrue(m['box_buffers_unchanged'])
            self.assertLessEqual(m['source_post']['norm'],.25*m['positive']['norm']*(1+1e-6)+1e-12)
            self.assertEqual(u['scaler_scale'],128)
            self.assertFalse(any(r['protected_violations'] for r in u['images']))
            self.assertEqual(sha(RUN18/u['checkpoint']['file']),u['checkpoint']['sha256'])
        self.assertFalse((RUN18/'updates/update-029.intent.json').exists())
    def test_checkpoint28_all12_heads_and_calibration_updated(self):
        rec=read(RUN18/'updates/update-028.json')['checkpoint'];p=RUN18/rec['file'];self.assertEqual(sha(p),rec['sha256']);s=torch.load(p,map_location='cpu',weights_only=True)
        self.assertEqual(set(s['weights']),HEAD_KEYS);self.assertEqual(s['updates'],28);self.assertTrue(all(int(v['step'])==28 for v in s['optimizer']['state'].values()))
        original=original_heads()
        for k in s['weights']:
            self.assertTrue(torch.isfinite(s['weights'][k]).all())
            if 'logit_shift' in k or 'logit_scale' in k:self.assertFalse(torch.equal(original[k],s['weights'][k]))
    def test_all_evaluation_raw_receipts_exact19_FP32(self):
        from analyze18 import raw_outputs
        _,_,_,dev,_=inputs18()
        self.assertEqual(len(raw_outputs(RUN18/'evaluation/epoch-1',dev)),19)
        self.assertFalse((RUN18/'evaluation/epoch-3').exists());self.assertFalse((RUN18/'evaluation/epoch-5').exists())
        r=read(REPORT18/'results.json');self.assertEqual(r['protocol']['supported_classes'],['sofa','chair','bed']);self.assertFalse(r['protocol']['independent_accuracy']);self.assertFalse(r['protocol']['frozen_reserved_official_scored'])
    def test_training_is59_only_with_exact_counts_and_no_new_regions(self):
        cfg,_,train,dev,_=inputs18();allowed={r['image_id'] for r in train};evidence={r['candidate_id'] for r in read(SOURCE/'source-negative-evidence.json')['records']}
        images=[r for i in range(1,29) for r in read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json'))['images']]
        self.assertEqual(len(images),111);self.assertEqual({r['image_id'] for r in images},allowed)
        self.assertFalse(allowed&{r['image_id'] for r in dev})
        self.assertTrue(all(set(r['region_query_counts'])<=evidence for r in images))
    def test_probe_weights_unchanged_and_exact24_training_only(self):
        ids={p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']}
        for epoch in (0,1):
            d=read(RUN18/'probes'/('epoch-'+str(epoch))/'complete.json');self.assertTrue(d['weights_unchanged']);self.assertEqual(len(d['records']),24);self.assertEqual({r['image_id'] for r in d['records']},ids)
    def test_original_disk_hash_and_original_C8_initialization(self):
        cfg,*_=inputs18();init=read(RUN18/'initialization.json');old=read(C8/'B-corrected/initialization.json')
        self.assertEqual(sha(BASE/'model.safetensors'),cfg['initialization']['SHA256']);self.assertEqual(init['initial_heads_sha256'],old['initial_heads_sha256']);self.assertEqual(init['frozen_sha256'],old['frozen_sha256']);self.assertTrue(init['calibration_trainable'])
    def test_CPU_metrics_complete_without_GPU_or_protocol_drift(self):
        d=read(RUN18/'cpu-partial-analysis-v1/complete.json');self.assertEqual(d['status'],'PASS');self.assertEqual(d['new_GPU_forwards'],0);self.assertTrue(d['historical_reference_metrics_reproduced_exactly'])
        self.assertFalse(torch.cuda.is_initialized())
    def test_effective_difference_guard_can_false_positive_on_FP32_addition(self):
        from balanced16 import compose,CLASS_KEYS,Policy,Accumulator
        from update18 import compose_checked
        base={k:torch.full((2,),64.) for k in HEAD_KEYS};p={k:torch.full((2,),.001) for k in CLASS_KEYS};n={k:torch.full((2,),-10.) for k in CLASS_KEYS};a=Accumulator(1,128)
        a.add({k:v*128 for k,v in base.items()},{k:v*128 for k,v in p.items()},{k:v*128 for k,v in n.items()},1)
        out,cap=compose(base,p,n,1,Policy(enabled=True));self.assertLessEqual(cap['used_negative_norm'],cap['budget']*(1+1e-6)+1e-12)
        with self.assertRaisesRegex(ValueError,'Measured cap violation'):compose_checked(a)


if __name__=='__main__':unittest.main()
