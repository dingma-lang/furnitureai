"""CPU regression tests. Synthetic module checks are not detector experiments."""
import ast
import copy
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from common21 import *
from box21 import PretrainedBoxScope, BOX_KEYS, restore_delta, frozen_fingerprint
from evaluation21 import validate_record, validate_manifest, registry21
from review21 import semantic_protected
from geometry6 import certify
from g5_common import overlap
from teacher21 import teacher_index, query_arrays

sys.path.insert(0, str(ROOT/'tools/phase10c10-v1'))
from test_preflight10 import fixture


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.train, cls.dev, cls.reg, _, _ = context21()
        cls.ids = {r['image_id'] for r in cls.train}
        cls.manifest = read(DATA21/'hard-negative-manifest-v1.json')
        cls.accepted = [r for r in cls.manifest['records'] if r['evidence_tier']=='A']

    def test_exact_optimization_and_group_isolation(self):
        self.assertEqual((len(self.train), len(self.dev)), (59,19))
        self.assertEqual(len({r['identity_group'] for r in self.train}),56)
        for field in ('image_id','identity_group','sha256'):
            self.assertFalse({r[field] for r in self.train}&{r[field] for r in self.dev})
        self.assertRaises(PermissionError, require_training_row, self.dev[0], self.train, self.dev)

    def test_complete_source_join_keeps_inactive_rows_inactive(self):
        full = read(DATA21/'full-local-native-inventory.json')
        report = read(REPORT21/'full-native-audit.json')
        self.assertEqual(len(full['photos']),101)
        self.assertEqual(sum(len(p['annotations']) for p in full['photos']),2272)
        self.assertEqual(report['optimization_annotation_records'],1386)
        self.assertEqual(report['newly_visible_native_records'],381)
        self.assertFalse(full['source_annotations_modified'])
        self.assertTrue(report['inactive42_scanned_metadata_but_not_mined'])
        for p in full['photos']:
            for a in p['annotations']:
                self.assertEqual(a['original_annotation_sha256'],digest(a['original_annotation']))
                self.assertFalse(a['Chair_absence_inferred'])

    def test_only_five_new_local_regions_four_groups(self):
        self.assertEqual(len(self.manifest['records']),18)
        self.assertEqual(len(self.accepted),5)
        self.assertEqual(len({r['image_id'] for r in self.accepted}),4)
        self.assertEqual(len({r['source_group'] for r in self.accepted}),4)
        self.assertEqual(len(self.manifest['duplicates']),12)
        self.assertTrue({r['image_id'] for r in self.manifest['records']}<=self.ids)
        self.assertFalse(self.manifest['training_authorized'])

    def test_evidence_identity_geometry_and_licence(self):
        photos = {p['image_id']:p for p in read(DATA21/'full-local-native-inventory.json')['photos']}
        for r in self.accepted:
            self.assertEqual(sha(ROOT/r['path']),r['source_sha256'])
            self.assertEqual(sha(ROOT/r['licence']['evidence_file']),r['licence']['evidence_sha256'])
            self.assertTrue(certify(r['box_pixels'],[tuple(p) for p in r['native_polygon']]))
            a = next(a for a in photos[r['image_id']]['annotations'] if a['annotation_key']==r['annotation_key'])
            self.assertEqual(digest(a['original_annotation']),r['annotation_sha256'])
            self.assertFalse(any(overlap(r['box'],a['box']) for a in photos[r['image_id']]['annotations'] if semantic_protected(a) or a['native_ignored']))
            self.assertFalse(any(overlap(r['box'],a['box']) for a in self.reg[r['image_id']]['protected']))
        for i,r in enumerate(self.accepted):
            self.assertFalse(any(r['image_id']==s['image_id'] and overlap(r['box'],s['box']) for s in self.accepted[i+1:]))

    def test_no_fake_human_review_or_generic_background(self):
        obs = read(DATA21/'query-local-visual-observations.json')['records']
        indexed = {r['candidate_id']:r for r in obs}
        for r in self.manifest['records']:
            self.assertFalse(r['human_reviewed']);self.assertFalse(r['human_confirmed'])
            self.assertFalse(r['training_activated']);self.assertEqual(r['category'],'chair')
            self.assertFalse(r['whole_photo_completeness'])
            self.assertFalse(r['other_furniture_negative_claim'])
            self.assertEqual(r['observation_sha256'],digest(indexed[r['candidate_id']]))
            self.assertEqual(sha(ROOT/indexed[r['candidate_id']]['plate']),indexed[r['candidate_id']]['plate_sha256'])
        self.assertFalse(self.manifest['primary_box_only_experiment_consumes_these_negatives'])

    def test_new_coverage_not_high_confidence_or_broad_background(self):
        old = read(SOURCE/'source-negative-evidence.json')['records']
        for r in self.accepted:
            self.assertLess(max(q['Chair_score'] for q in r['query_observations']),.5)
            self.assertFalse(any(s['image_id']==r['image_id'] and all(r['box'][j]>=s['box'][j] for j in (0,1)) and all(r['box'][j]<=s['box'][j] for j in (2,3)) for s in old))
        self.assertEqual(Counter(r['native_label'] for r in self.accepted),{'person':2,'piano':1,'television_set':1,'handbag':1})

    def test_context_overlap_never_establishes_negative(self):
        d=read(DATA21/'automatic-context-candidates-v1.json')
        self.assertEqual(len(d['records']),658)
        self.assertEqual(Counter(r['evidence_tier'] for r in d['records']),{'B':373,'C':285})
        for r in d['records']:
            self.assertIn(r['image_id'],self.ids)
            self.assertFalse(r['training_eligible']);self.assertFalse(r['Chair_negative_proven'])
            self.assertFalse(r['training_activated'])

    def test_ambiguous_parts_and_seating_protected(self):
        for label in ('beanbag','wardrobe','headboard','lampshade','cushion','knob','chandelier'):
            self.assertTrue(semantic_protected(dict(native_label=label,protect_furniture=False,part_or_semantics_unresolved=False)))
        for label in ('Chair','Stool','stool','footstool'):
            self.assertTrue(semantic_protected(dict(native_label=label,protect_furniture=True,part_or_semantics_unresolved=False)))

    def test_exact_teacher_cache_hashes_and_24_35_provenance(self):
        records=teacher_index()
        self.assertEqual(records,read(DATA21/'teacher-cache-index.json')['records'])
        self.assertEqual(len(records),59)
        self.assertEqual(sum(r['matches_original24_cache'] is True for r in records),24)
        self.assertEqual(sum(r['matches_original24_cache'] is None for r in records),35)
        self.assertEqual({r['image_id'] for r in records},self.ids)
        self.assertTrue(all('FP16' in r['precision'] and r['query_layout']==[60,60] for r in records))

    def test_teacher_tampered_hash_fails(self):
        r=read(DATA21/'teacher-cache-index.json')['records'][0]
        self.assertRaises(ValueError,query_arrays,ROOT/r['cache'],'0'*64)


class ScopeTests21(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.reference=original_heads()

    def setUp(self):
        self.model=fixture(self.reference)
        self.scope=PretrainedBoxScope(self.model)
        self.opt=self.scope.optimizer()

    def test_exact_six_box_tensors(self):
        self.assertEqual({n for n,p in self.model.named_parameters() if p.requires_grad},BOX_KEYS)
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),1184260)
        self.scope.check(self.opt,check_hash=True)
        self.assertFalse(torch.cuda.is_initialized())

    def test_random_or_adapted_classification_rejected(self):
        self.assertRaises(ValueError,PretrainedBoxScope,fixture())
        with torch.no_grad():self.model.class_head.logit_shift.bias.add_(.01)
        self.assertRaises(ValueError,PretrainedBoxScope,self.model)

    def test_train_mode_drift_detected(self):
        self.model.train();self.assertRaises(ValueError,self.scope.check,self.opt)

    def test_optimizer_parameter_or_hyperparameter_drift_detected(self):
        self.opt.param_groups[0]['lr']=3e-5
        self.assertRaises(ValueError,self.scope.check,self.opt)
        self.opt.param_groups[0]['lr']=1e-5
        self.opt.add_param_group({'params':[self.model.class_head.dense0.weight]})
        self.assertRaises(ValueError,self.scope.check,self.opt)

    def test_frozen_grad_and_nonfinite_gradient_detected(self):
        self.model.class_head.dense0.weight.grad=torch.zeros_like(self.model.class_head.dense0.weight)
        self.assertRaises(ValueError,self.scope.check,self.opt)
        self.model.class_head.dense0.weight.grad=None
        for n,p in self.model.named_parameters():
            if n in BOX_KEYS:p.grad=torch.ones_like(p)
        self.model.box_head.dense2.bias.grad[0]=float('nan')
        self.assertRaises(ValueError,self.scope.check,self.opt,True)

    def test_frozen_backbone_and_buffer_hash(self):
        with torch.no_grad():self.model.backbone.weight.add_(1)
        self.assertRaises(ValueError,self.scope.check,self.opt,check_hash=True)
        self.model=fixture(self.reference);self.model.register_buffer('nonpersistent_probe',torch.ones(1),persistent=False)
        s=PretrainedBoxScope(self.model)
        self.model.nonpersistent_probe.add_(1)
        self.assertRaises(ValueError,s.check,check_hash=True)

    def test_synthetic_box_step_preserves_classifier_and_all_frozen_values(self):
        # CPU toy feature vector,not a photo or real OWLv2 forward.
        x=torch.randn(3,768)
        before=self.model.class_head.dense0(x).detach().clone()
        out=self.model.box_head.dense2(torch.relu(self.model.box_head.dense1(torch.relu(self.model.box_head.dense0(x)))))
        out.square().mean().backward()
        self.scope.check(self.opt,require_gradients=True)
        self.opt.step();self.scope.check(self.opt,check_hash=True)
        self.assertTrue(torch.equal(before,self.model.class_head.dense0(x)))
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if n not in BOX_KEYS))

    def test_six_tensor_checkpoint_roundtrip_and_binding(self):
        payload=self.scope.checkpoint_payload(0,'a'*64)
        stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
        loaded=torch.load(stream,map_location='cpu',weights_only=True)
        m=fixture(self.reference);restored=restore_delta(m,loaded,'a'*64)
        self.assertEqual(frozen_fingerprint(m),restored.expected_frozen)
        self.assertEqual(set(loaded['weights']),BOX_KEYS)
        self.assertRaises(ValueError,restore_delta,m,loaded,'b'*64)
        self.assertRaises(ValueError,self.scope.checkpoint_payload,76,'a'*64)
        self.assertRaises(ValueError,self.scope.checkpoint_payload,0,'missing')

    def test_invalid_checkpoint_validated_before_weight_copy(self):
        p=self.scope.checkpoint_payload(0,'a'*64)
        p['weights']['box_head.dense2.bias'][0]=float('nan')
        before={n:v.detach().clone() for n,v in self.model.named_parameters()}
        self.assertRaises(ValueError,restore_delta,self.model,p,'a'*64)
        self.assertTrue(all(torch.equal(before[n],v) for n,v in self.model.named_parameters()))

    def test_loss_detaches_classification_and_ignores_missing_classes(self):
        row=dict(width=640,height=480,annotations=[dict(label='chair',box=[.2,.2,.6,.6],ignore=False),dict(label='lamp',box=[.7,.1,.9,.5],ignore=True)],explicit_native_negative={})
        logits=torch.zeros(5,7,requires_grad=True);boxes=torch.full((5,4),.4,requires_grad=True)
        loss,parts=self.scope.loss(logits,boxes,row);loss.backward()
        self.assertIsNone(logits.grad);self.assertIsNotNone(boxes.grad)
        self.assertEqual(parts['matched'],1);self.assertFalse(parts['unknowns_as_background'])
        self.assertEqual(parts['classification'],0)
        row['annotations'][0]['label']='table'
        self.assertRaises(ValueError,self.scope.loss,logits,boxes,row)

    def test_coordinate_registration_landscape_portrait(self):
        from expanded import to_square,xyxy_to_cxcywh
        from common9 import registered_box
        import numpy as np
        b=torch.tensor([[.1,.2,.6,.8]])
        for w,h in ((640,480),(480,640),(511,511)):
            square=xyxy_to_cxcywh(to_square(b,w,h))
            np.testing.assert_allclose(registered_box(square.numpy(),w,h,clip=False),b.numpy(),atol=1e-6)


class EvaluationTests21(unittest.TestCase):
    def setUp(self):
        self.registry=registry21()
        self.record=dict(image_id='SYNTHETIC-TEST-ONLY',sha256='a'*64,source_photo_id='synthetic-source',source_group='synthetic-creator',capture_session_group='synthetic-session',source_group_independence_verified=True,source_group_evidence='SYNTHETIC TEST ONLY',rights_verified=True,rights_evidence_sha256='b'*64,licence_url='https://example.invalid/synthetic',role='independent_acceptance',category_complete={c:True for c in CLASSES},Chair_includes_seating_and_bar_stools=True,whole_object_taxonomy_confirmed=True,annotations=[dict(label='chair',box=[.1,.2,.3,.4],ignore=False)],review=dict(human_reviewed=True,blind_to_model_predictions=True,reviewer='SYNTHETIC-TEST-ONLY',date='2026-11-22',evidence='SYNTHETIC TEST ONLY'))

    def test_actual_independent_evidence_is_zero_no_reviews_filled(self):
        self.assertEqual(read(REPORT21/'independent-evaluation-readiness.json')['actual_independent_complete_eligible_photos'],0)
        self.assertEqual(read(DATA21/'independent-review-template.json')['records'],[])

    def test_synthetic_validator_cannot_claim_file_verified(self):
        self.assertTrue(validate_record(self.record,self.registry,verify_files=False)['synthetic_test_only'])
        self.assertRaises(KeyError,validate_record,self.record,self.registry)

    def test_historical_id_hash_source_and_creator_rejected(self):
        for field,values in [('image_id','image_ids'),('sha256','sha256'),('source_photo_id','source_photo_ids'),('source_group','source_groups'),('capture_session_group','source_groups')]:
            r=copy.deepcopy(self.record);r[field]=self.registry[values][0]
            self.assertRaises(PermissionError,validate_record,r,self.registry,False)

    def test_missing_review_taxonomy_completeness_and_bad_boxes_rejected(self):
        for mutate in (lambda r:r['review'].update(human_reviewed=False),lambda r:r['review'].update(date='not a date'),lambda r:r['category_complete'].update(lamp=False),lambda r:r.update(Chair_includes_seating_and_bar_stools=False),lambda r:r['annotations'][0].update(box=[0,0,2,1]),lambda r:r.update(source_group=''),lambda r:r.update(sha256='missing')):
            r=copy.deepcopy(self.record);mutate(r)
            self.assertRaises(ValueError,validate_record,r,self.registry,False)

    def test_duplicate_source_group_in_future_manifest_rejected(self):
        self.assertRaises(ValueError,validate_manifest,[self.record,self.record],self.registry,False)


class ProposalTests21(unittest.TestCase):
    def test_public_seal_uses_repository_relative_paths(self):
        from verify21 import public_seal_bytes
        path=REPORT21/'preflight.json'
        data=public_seal_bytes([path]).decode()
        h,rel=data.strip().split('  ',1)
        self.assertEqual(rel,'reports/phase10c21-v1/preflight.json')
        self.assertEqual(sha(ROOT/rel),h)

    def test_public_seal_rejects_outside_root(self):
        from verify21 import public_seal_bytes
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file.json';p.write_text('{}')
            self.assertRaises(PermissionError,public_seal_bytes,[p])

    def test_single_disabled_arm_exact_original_and_resource_limits(self):
        cfg=read(REPORT21/'box-only-disabled.json')
        self.assertFalse(cfg['approved']);self.assertFalse(cfg['enabled']);self.assertFalse(cfg['Phase10D_authorized'])
        self.assertFalse(cfg['C20_or_adapted_head_initialization'])
        self.assertEqual(cfg['starting_sha256'],sha(BASE/'model.safetensors'))
        self.assertEqual(set(cfg['trainable_keys']),BOX_KEYS)
        self.assertEqual(cfg['limits']['max_forwards'],4+5*59+4+19)
        self.assertEqual(cfg['limits']['max_optimizer_updates'],75)
        self.assertEqual(cfg['limits']['max_GPU_seconds_including_smoke_and_diagnostics'],600)
        self.assertEqual(cfg['training']['category_counts'],{'sofa':40,'chair':114,'bed':28})
        self.assertFalse(cfg['training']['new_negative_manifest_used'])
        self.assertFalse((ROOT/cfg['destination']).exists())

    def test_rights_candidates_exact64_new_not_previously_paused(self):
        d=read(DATA21/'additional-rights-candidates-v1.json');records=d['records']
        self.assertEqual(len(records),64);self.assertTrue(d['no_requests_performed'])
        self.assertEqual(len({r['image_id'] for r in records}),64)
        ids={r['image_id'] for r in records}
        self.assertFalse(ids&set(registry21()['image_ids']))
        for rel in ('data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json','data/lvis/rights-audits/phase7-v1/license-and-attribution-ledger.json','data/lvis/rights-audits/phase8a-v1/license-attribution-ledger.json'):
            old=read(ROOT/rel)['records'];oldids={r.get('image_id') or 'coco2017:'+str(r.get('coco_image_id',r.get('lvis_image_id'))).zfill(12) for r in old}
            self.assertFalse(ids&oldids)
        for r in records:
            self.assertEqual(r['rights_status'],'individually_unverified')
            self.assertFalse(r['photo_download_authorized']);self.assertIsNone(r['creator'])
            self.assertEqual(r['metadata_licence_id'],4)

    def test_no_gpu_or_downloader_entrypoints_in_preparation(self):
        for p in HERE21.glob('*.py'):
            if p.name.startswith('test_'):continue
            tree=ast.parse(p.read_text())
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertFalse(set(calls)&{'from_pretrained','urlopen','get_connection','set_device','backward','step'})
        self.assertFalse(torch.cuda.is_initialized())

    def test_write_once_and_no_outside_root(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.json'
            self.assertRaises(PermissionError,atomic21,p,b'{}')
        self.assertRaises(FileExistsError,emit21,REPORT21/'preflight.json',dict(overwrite=True))


if __name__=='__main__':unittest.main()
