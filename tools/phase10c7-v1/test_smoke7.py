"""CPU-only forward-free tests. No backward/optimizer calls, even on toy models."""
import ast,copy,tempfile,unittest
import numpy as np
from common7 import *

class CoordinatesTests(unittest.TestCase):
    def test_landscape_padding_inverse_not_clipped(self):
        p,n=original_boxes(np.array([[.5,.5,.5,.5]],np.float32),800,400);np.testing.assert_array_equal(p,[[200,200,600,600]]);np.testing.assert_array_equal(n,[[.25,.5,.75,1.5]])
    def test_portrait_padding_inverse_not_clipped(self):
        p,n=original_boxes(np.array([[.5,.5,.5,.5]],np.float32),400,800);np.testing.assert_array_equal(n,[[.5,.25,1.5,.75]])
    def test_source_to_square_roundtrip(self):
        import torch
        from masks6 import to_square,xyxy_to_cxcywh
        b=torch.tensor([[.1,.2,.3,.4]]);square=xyxy_to_cxcywh(to_square(b,800,400));_,n=original_boxes(square.numpy(),800,400);np.testing.assert_allclose(n,b.numpy(),atol=1e-7)
    def test_real_region_membership_requires_full_box(self):
        from analyze7 import responsible_regions
        r=[dict(candidate_id='SYNTHETIC',box_pixels=[20,20,40,40])]
        self.assertEqual(responsible_regions([[21,21,39,39],[10,10,35,35]],r),[['SYNTHETIC'],[]])
    def test_pinned_processor_square_pad(self):
        from transformers import Owlv2Processor
        from PIL import Image
        p=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True);x=p(text=[PROMPTS],images=Image.new('RGB',(80,40),'black'),return_tensors='np')['pixel_values'];self.assertEqual(x.shape,(1,3,960,960))
        expected=(.5-np.array(p.image_processor.image_mean))/np.array(p.image_processor.image_std);np.testing.assert_allclose(x[0,:,900,400],expected,atol=1e-5)

def fixture_arrays():
    q=np.tile(np.array([[.5,.5,.2,.2]],np.float32),(3600,1));p,n=original_boxes(q,640,480)
    return dict(logits=np.zeros((3600,7),np.float32),square_cxcywh=q,original_xyxy_pixels=p,original_xyxy_normalized=n)

class ArtifactTests(unittest.TestCase):
    def test_atomic_array_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'raw.npz';a=fixture_arrays();write_arrays(path,a);b=load_arrays(path)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_existing_arrays_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'raw.npz';a=fixture_arrays();write_arrays(p,a);old=sha(p)
            with self.assertRaises(FileExistsError):write_arrays(p,a)
            self.assertEqual(old,sha(p))
    def test_nonfinite_raw_rejected(self):
        a=fixture_arrays();a['logits'][0,0]=np.nan
        with self.assertRaises(ValueError):arrays_valid(a)
    def test_wrong_class_count_rejected(self):
        a=fixture_arrays();a['logits']=np.zeros((3600,6),np.float32)
        with self.assertRaises(ValueError):arrays_valid(a)
    def test_wrong_dtype_rejected(self):
        a=fixture_arrays();a['logits']=a['logits'].astype(np.float64)
        with self.assertRaises(ValueError):arrays_valid(a)
    def test_zero_width_rejected(self):
        a=fixture_arrays();a['square_cxcywh'][0,2]=0
        with self.assertRaises(ValueError):arrays_valid(a)

class LossTests(unittest.TestCase):
    def test_masked_focal_preserves_normalization(self):
        import torch,math
        from analyze7 import classification_loss
        with torch.inference_mode():
            x=torch.zeros((2,7));target=torch.zeros_like(x);mask=torch.zeros_like(x,dtype=torch.bool);mask[:,1]=True
            self.assertAlmostEqual(float(classification_loss(x,target,mask,2)),.75*.25*math.log(2),places=6)
    def test_unknown_categories_have_zero_loss(self):
        import torch
        from analyze7 import classification_loss
        with torch.inference_mode():self.assertEqual(float(classification_loss(torch.zeros((3,7)),torch.zeros((3,7)),torch.zeros((3,7),dtype=torch.bool),1)),0.)
    def test_same_matching_as_legacy_loss(self):
        import torch
        from analyze7 import assigned,classification_loss
        from expanded import supervision,detection_loss,xyxy_to_cxcywh
        row=dict(width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
        with torch.inference_mode():
            x=torch.zeros((2,7));b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.1,.1,.2,.2]]));ii,labels,n=assigned(x,b,row);t,m=supervision(x,b,row,ii,labels);loss,_=detection_loss(x,b,row)
            self.assertEqual(ii.tolist(),[0]);self.assertEqual(labels.tolist(),[1]);self.assertAlmostEqual(float(loss),float(classification_loss(x,t,m,n)),places=5)
    def test_positive_or_ignored_boxes_never_added(self):
        import torch
        from masks6 import target_mask,xyxy_to_cxcywh
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False),dict(label='desk',box=[0,0,.3,.3],ignore=True)],explicit_native_negative={})
        with torch.inference_mode():
            b=xyxy_to_cxcywh(torch.tensor([[.75,.75,.85,.85],[.1,.1,.2,.2],[.4,.4,.5,.5]]));x=torch.zeros((3,7));e=[dict(image_id='SYNTHETIC',category='chair',box=[0,0,1,1])];t,m,added=target_mask(x,b,row,torch.tensor([0]),torch.tensor([1]),e)
            self.assertFalse(added[0].any());self.assertFalse(added[1].any());self.assertTrue(added[2,1]);self.assertEqual(int(added.sum()),1)

class SafetyTests(unittest.TestCase):
    def test_worker_has_no_training_or_weight_load_delta(self):
        tree=ast.parse((HERE/'gpu7.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        bad=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','load_state_dict','copy_')];self.assertFalse(bad)
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='model' for n in calls),1)
    def test_hard_caps(self):self.assertEqual((MAX_FORWARDS,MAX_SECONDS,MAX_VRAM),(24,600,6979321856))
    def test_exact_frozen_24_photo_proposal(self):
        p=read(SOURCE/'proposed-smoke-photos.json')['photos'];self.assertEqual(len(p),24);self.assertTrue(all(r['internal_partition']=='optimization' for r in p));self.assertEqual(len({r['identity_group'] for r in p}),24)
    def test_parent_has_hard_timeout_no_retry_loop(self):
        code=(HERE/'launch7.py').read_text();self.assertIn('timeout=MAX_SECONDS',code);self.assertIn('SIGKILL',code);self.assertNotIn('while ',code)

if __name__=='__main__':unittest.main()
