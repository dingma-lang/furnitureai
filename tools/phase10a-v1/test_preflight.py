import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from common import ROOT, OUT, READINESS, CLASSES, read, sha, save, safe
from policy import CORE, positive_allowed, normalized_box, explicit_negatives, inner_split, Union
from expanded import detection_loss, supervision, to_square, from_square, xyxy_to_cxcywh, configure_heads, load_delta, load_training_row, HEAD_PARAMETERS, HEAD_KEYS
from protocol import check_bounds, require_training_authorization, update_count, FIT_IDS
from rights import replay
from recovery import Budget
from evaluation import metric_row

def synthetic():
    return dict(width=640,height=320,explicit_native_negative={c:False for c in CLASSES},
                annotations=[dict(label='chair',box=[.1,.1,.5,.9],ignore=False,has_mask=False)])

class PolicyTests(unittest.TestCase):
    def test_exact_counts_and_dispositions(self):
        m=read(OUT/'prepared-manifest.json'); d=read(OUT/'annotation-dispositions.json')
        self.assertEqual((m['original_train_images'],m['original_train_boxes']),(120,373))
        self.assertEqual((m['eligible_images'],m['eligible_groups'],m['eligible_boxes']),(78,75,243))
        self.assertEqual(len(d['records']),373)
        self.assertEqual(sum(not r['prepared_positive'] for r in d['records']),130)
        self.assertTrue(d['previous33_LVIS_overlays_inactive'])
        self.assertTrue(all(not r['activated_for_training'] for r in d['records']))

    def test_native_mapping_not_alias_substitution(self):
        self.assertNotIn(361,CORE['COCO']); self.assertEqual(CORE['LVIS'][361],'desk')
        self.assertEqual(CORE['COCO'][67],'table')
        self.assertNotIn(1051,CORE['LVIS']); self.assertNotIn(628,CORE['LVIS'])
        for row in read(READINESS)['source_rows']:
            for t in row['resolved_targets']:
                if t['comparison_group'] in ('table','desk','cabinet','lamp'):
                    self.assertFalse(positive_allowed(t))

    def test_inside_group_depiction_and_unresolved_inactive(self):
        t=next(t for r in read(READINESS)['source_rows'] if r['split']=='train' for t in r['resolved_targets'] if positive_allowed(t))
        for field in ('group_of',):
            q=copy.deepcopy(t);q[field]=True;self.assertFalse(positive_allowed(q))
        for flag in ('IsInside','IsDepiction'):
            q=copy.deepcopy(t);q['original_annotation']['flags']={flag:1};self.assertFalse(positive_allowed(q))
        q=copy.deepcopy(t);q['mapping_scope']='native_only_unit_policy_required';self.assertFalse(positive_allowed(q))

    def test_no_dataset_duplicate_or_known_partition_leak(self):
        m=read(OUT/'prepared-manifest.json');rows=m['rows']
        for k in ('image_id','source_photo_id','sha256'):
            self.assertEqual(len({r[k] for r in rows}),len(rows))
        a=[r for r in rows if r['internal_partition']=='optimization'];b=[r for r in rows if r['internal_partition']=='inner_development']
        self.assertFalse({r['identity_group'] for r in a}&{r['identity_group'] for r in b})
        official={r['sha256'] for r in read(READINESS)['source_rows'] if r['split']!='train'}
        frozen={r['sha256'] for r in read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')['frozen_references']}
        self.assertFalse({r['sha256'] for r in rows}&(official|frozen))
        self.assertFalse(m['source_group_issues'])

    def test_deterministic_source_group_split(self):
        rows=read(OUT/'prepared-manifest.json')['rows']
        strip=lambda r:{k:v for k,v in r.items() if k!='internal_partition'}
        self.assertEqual(inner_split([strip(r) for r in rows]),inner_split([strip(r) for r in reversed(rows)]))

    def test_transitive_creator_group(self):
        u=Union();u.join('photoA','creatorAlias');u.join('creatorAlias','creatorID');u.join('photoB','creatorID')
        self.assertEqual(u.find('photoA'),u.find('photoB'))

    def test_training_image_decode_checks(self):
        r=read(OUT/'train-photo-decode-verification.json')
        self.assertEqual(r['training_photos_decoded'],120)
        self.assertEqual(r['development_or_frozen_photos_decoded'],0)
        self.assertTrue(all(x['exif_orientation']==1 for x in r['records']))

    def test_no_missing_negative_inference(self):
        r=dict(source_native_scope={'not_exhaustive_category_ids':[232]})
        self.assertFalse(any(explicit_negatives(r).values()))
        r['source_native_scope']={'neg_category_ids':[181,626,367]}
        self.assertFalse(any(explicit_negatives(r).values()))
        r['source_native_scope']={'neg_category_ids':[232]}
        self.assertEqual([c for c,v in explicit_negatives(r).items() if v],['chair'])

    def test_real_hashes_rights_and_native_coverage_pinned(self):
        m=read(OUT/'prepared-manifest.json')
        self.assertEqual(m['source_readiness_sha256'],sha(READINESS))
        rights=read(OUT/'rights-replay-ledger.json')['records'];self.assertEqual(len(rights),120)
        for r in rights:
            self.assertEqual(sha(safe(r['evidence_file'])),r['evidence_sha256'])
            self.assertEqual(sha(safe(r['rights_record'])),r['rights_record_sha256'])

    def test_no_external_partition_or_forged_row_access(self):
        r=read(OUT/'prepared-manifest.json')['rows'][0]
        with patch('expanded.sha',side_effect=AssertionError('must deny before bytes')):
            for p in ('development','final','frozen_evaluation'):
                with self.assertRaises(PermissionError):load_training_row(r,p)
            q=copy.deepcopy(r);q['path']='data/unapproved.jpg'
            with self.assertRaises(PermissionError):load_training_row(q,q['internal_partition'])

    def test_internal_development_cannot_enter_optimization_loader(self):
        r=next(r for r in read(OUT/'prepared-manifest.json')['rows'] if r['internal_partition']=='inner_development')
        with patch('expanded.sha',side_effect=AssertionError('deny before opening')):
            with self.assertRaises(PermissionError):load_training_row(r,'optimization')

    def test_native_evaluation_partial_scopes(self):
        rows=read(OUT/'prepared-manifest.json')['rows']
        for r in rows:
            m=metric_row(r)
            for c in ('table','desk','cabinet','lamp'):self.assertFalse(m['fp_coverage'][c])
            if r['source']=='phase6d_coco':self.assertFalse(any(m['fp_coverage'].values()))
        r=copy.deepcopy(next(r for r in rows if r['source']=='phase7_LVIS'))
        a=next(a for a in r['annotations'] if not a['ignore'])
        r['source_native_scope']['not_exhaustive_category_ids']=[a['native_id']]
        self.assertFalse(metric_row(r)['fp_coverage'][a['label']])
        r['historical_partition']='frozen_evaluation'
        with self.assertRaises(PermissionError):metric_row(r)

    def test_safe_paths(self):
        for name in ('../../etc/passwd','/etc/passwd'):
            with self.assertRaises(ValueError):safe(name)

class LossTests(unittest.TestCase):
    def test_variable_size_coordinate_roundtrip(self):
        for w,h in ((640,320),(320,640),(517,313),(400,400)):
            b=torch.tensor([[.05,.1,.8,.9]])
            torch.testing.assert_close(from_square(to_square(b,w,h),w,h),b)
            native=[.05*w,.1*h,.75*w,.8*h]
            self.assertEqual(len(normalized_box('COCO',{'bbox':native},w,h)),4)
        with self.assertRaises(ValueError):normalized_box('COCO',{'bbox':[10,10,600,100]},500,400)

    def test_actual_local_processor_padding(self):
        # Processor only, synthetic pixels only; no encoder/detector is constructed.
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ImageProcessor
        directory=ROOT/'models/owlv2-original-baseline-20261103'
        processor=Owlv2ImageProcessor.from_pretrained(str(directory),local_files_only=True)
        for w,h in ((640,320),(320,640),(517,313)):
            image=Image.fromarray(np.full((h,w,3),255,dtype=np.uint8))
            batch=processor(images=image,return_tensors='pt')['pixel_values']
            self.assertEqual(tuple(batch.shape),(1,3,960,960))
            # Inverse normalization: real content is white; bottom/right padding is gray(.5).
            mean=torch.tensor(processor.image_mean)[:,None,None];std=torch.tensor(processor.image_std)[:,None,None]
            pixels=batch[0]*std+mean
            torch.testing.assert_close(pixels[:,20,20],torch.ones(3),atol=.015,rtol=0)
            y,x=(900,20) if w>h else (20,900)
            torch.testing.assert_close(pixels[:,y,x],torch.full((3,),.5),atol=.015,rtol=0)

    def test_unknown_zero_class_and_box_gradients(self):
        row=synthetic();logits=torch.zeros(3,7,requires_grad=True)
        boxes=torch.tensor([[.3,.25,.4,.4],[.8,.2,.1,.1],[.7,.4,.1,.1]],requires_grad=True)
        loss,p=detection_loss(logits,boxes,row);loss.backward()
        self.assertEqual(p['matched'],1);self.assertFalse(p['objectness_supervised'])
        self.assertEqual(int((logits.grad!=0).sum()),1)
        self.assertTrue(torch.all(logits.grad[:,[0,2,3,4,5,6]]==0))
        self.assertTrue(torch.all(boxes.grad[1:]==0))

    def test_explicit_negative_ignores_unknown_and_padding(self):
        row=synthetic();row['annotations']=[dict(label='unknown',box=[.1,.1,.5,.9],ignore=True)]
        row['explicit_native_negative']['sofa']=True
        boxes=torch.tensor([[.3,.2,.1,.1],[.8,.2,.1,.1],[.5,.8,.1,.1]])
        _,valid=supervision(torch.zeros(3,7),boxes,row)
        self.assertFalse(valid[0].any());self.assertTrue(valid[1,0]);self.assertFalse(valid[2].any())
        self.assertEqual(int(valid.sum()),1)

    def test_zero_valid_loss_and_gradients(self):
        row=synthetic();row['annotations']=[]
        l=torch.zeros(3,7,requires_grad=True);b=torch.full((3,4),.3,requires_grad=True)
        loss,_=detection_loss(l,b,row);loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertTrue(torch.all(l.grad==0));self.assertTrue(torch.all(b.grad==0))

    def test_unresolved_and_nonfinite_loss_rejected(self):
        row=synthetic();row['annotations'][0]['label']='cabinet'
        with self.assertRaises(ValueError):detection_loss(torch.zeros(3,7),torch.full((3,4),.3),row)
        row=synthetic();l=torch.zeros(3,7);l[0,0]=float('nan')
        with self.assertRaises(ValueError):detection_loss(l,torch.full((3,4),.3),row)

    def test_matched_positive_overlaps_unresolved_region(self):
        row=synthetic();row['annotations'].append(dict(label='unknown',box=[0,0,1,1],ignore=True))
        l=torch.zeros(2,7,requires_grad=True);b=torch.tensor([[.3,.25,.4,.4],[.8,.2,.1,.1]],requires_grad=True)
        loss,p=detection_loss(l,b,row);loss.backward();self.assertEqual(p['valid_class_terms'],1)
        self.assertLess(float(l.grad[0,1]),0)

    def test_positive_negative_conflict_rejected(self):
        row=synthetic();row['explicit_native_negative']['chair']=True
        with self.assertRaises(ValueError):detection_loss(torch.zeros(2,7),torch.full((2,4),.3),row)

    def test_nonexhaustive_positive_does_not_add_negatives(self):
        row=synthetic();row['source_native_scope']={'not_exhaustive_category_ids':[232]}
        row['explicit_native_negative']=explicit_negatives(row)
        logits=torch.zeros(3,7,requires_grad=True);boxes=torch.full((3,4),.3,requires_grad=True)
        loss,parts=detection_loss(logits,boxes,row);loss.backward()
        self.assertEqual(parts['valid_class_terms'],1)
        self.assertEqual(int((logits.grad!=0).sum()),1)

class CheckpointAndControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reg=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json')
        p=Path(reg['checkpoint']);assert sha(p)==reg['sha256']
        cls.payload=torch.load(p,map_location='cpu',weights_only=True)

    def test_actual_delta_head_keys_shapes_and_freezing(self):
        class Toy(torch.nn.Module):
            def __init__(self,weights):
                super().__init__();self.backbone=torch.nn.Linear(2,2);self.objectness_head=torch.nn.Linear(2,1)
                for n,t in weights.items():
                    head,layer,param=n.split('.')
                    if not hasattr(self,head):self.add_module(head,torch.nn.Module())
                    module=getattr(self,head)
                    if not hasattr(module,layer):module.add_module(layer,torch.nn.Module())
                    getattr(module,layer).register_parameter(param,torch.nn.Parameter(torch.zeros_like(t)))
        model=Toy(self.payload['weights']);before=model.backbone.weight.detach().clone()
        self.assertEqual(set(configure_heads(model)),HEAD_KEYS);load_delta(model,self.payload)
        self.assertFalse(model.backbone.training);self.assertFalse(model.objectness_head.training)
        self.assertFalse(model.backbone.weight.requires_grad)
        torch.testing.assert_close(model.backbone.weight,before)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),HEAD_PARAMETERS)
        for n,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p,self.payload['weights'][n])
        bad=dict(self.payload,weights={})
        with self.assertRaises(ValueError):load_delta(model,bad)

    def test_config_caps_and_no_phase10a_authorization(self):
        for name in FIT_IDS:
            c=read(OUT/'configs'/(name+'.json'));self.assertTrue(check_bounds(c))
            for key,val in [('epochs',6),('max_updates',151),('total_queue_max_updates',301),('mode','lora'),('accumulation',1)]:
                bad=dict(c,**{key:val})
                with self.assertRaises(ValueError):check_bounds(bad)
            with self.assertRaises(PermissionError):check_bounds(dict(c,training_authorized=True))
        with self.assertRaises(PermissionError):require_training_authorization()

    def test_real_updates_less_than_ceiling(self):
        self.assertEqual(update_count(59),75);self.assertEqual(update_count(2000),150)

    def test_non_overwriting_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';save(p,{'a':1})
            with self.assertRaises(FileExistsError):save(p,{'a':2})
            self.assertEqual(read(p),{'a':1})

    def test_no_worker_training_import_or_call(self):
        import ast
        for p in (ROOT/'tools/phase10a-v1').glob('*.py'):
            if p.name.startswith('test_'):continue
            tree=ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Attribute):self.assertNotIn(node.attr,('step','backward','from_pretrained'))

class RightsTests(unittest.TestCase):
    def body(self,lic='https://creativecommons.org/licenses/by/2.0/',photo='123',creator='owner'):
        d={'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/owner/'+photo,
           'license':lic,'author':{'name':'Owner','url':'https://www.flickr.com/photos/'+creator},'name':'Room'}
        return ('<script type="application/ld+json">'+json.dumps(d)+'</script>').encode()
    def test_exact_rights_binding(self):
        r=replay(self.body(),'flickr-photo:123','https://creativecommons.org/licenses/by/2.0/','https://www.flickr.com/photos/owner')
        self.assertEqual(r['creator'],'Owner')
    def test_restricted_wrong_photo_or_creator(self):
        for b in [self.body(lic='https://creativecommons.org/licenses/by-nc/2.0/'),self.body(photo='999'),self.body(creator='different')]:
            with self.assertRaises(ValueError):replay(b,'flickr-photo:123','https://creativecommons.org/licenses/by/2.0/','https://www.flickr.com/photos/owner')

class RecoveryTests(unittest.TestCase):
    def test_uncertain_boundary_blocks_replay(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,{'config':'fixed'});b.create();b.reserve(FIT_IDS[0],0,0)
            with self.assertRaises(RuntimeError):b.resume(FIT_IDS[0],0)
            with self.assertRaises(RuntimeError):b.reserve(FIT_IDS[0],0,1)

    def test_hashed_checkpoint_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,{'config':'fixed'});b.create();b.reserve(FIT_IDS[0],0,0)
            p=Path(d)/'unique-checkpoint.bin';p.write_bytes(b'cpu-test-state')
            b.committed(FIT_IDS[0],1,p,1);b.resume(FIT_IDS[0],1)
            p.write_bytes(b'damaged')
            with self.assertRaises(ValueError):b.resume(FIT_IDS[0],1)
            with self.assertRaises(ValueError):Budget(d,{'config':'different'}).resume(FIT_IDS[0],1)

    def test_150_per_fit_300_total_and_time_bound(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,{'config':'fixed'});b.create()
            for j,name in enumerate(FIT_IDS):
                for i in range(150):
                    t=j*150+i;b.reserve(name,i,t)
                    p=Path(d)/f'{j}-{i}.bin';p.write_bytes(b'no-model-test')
                    b.committed(name,i+1,p,t+.5)
                with self.assertRaises(RuntimeError):b.reserve(name,150,301)
            self.assertEqual(sum(r['applied'] for r in b.checked()['fits'].values()),300)
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,{});b.create()
            with self.assertRaises(RuntimeError):b.reserve(FIT_IDS[0],0,3600)
            with self.assertRaises(FileExistsError):b.create()

if __name__=='__main__':unittest.main()
