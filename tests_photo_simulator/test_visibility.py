import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from photo_simulator.visibility import coverage_masks
from photo_simulator.compositor import composite,_warp_asset
from photo_simulator.restoration import restore_background,refine_alpha

class VisibilityTests(unittest.TestCase):
 def fixture(self):
  rgb=np.full((120,160,3),170,np.uint8);m=np.zeros((120,160),bool);m[40:95,45:115]=True;rgb[m]=[240,10,10];p=np.zeros_like(m);p[65:72,40:120]=True
  asset=Image.new('RGBA',(80,45),(20,90,40,255));placement={'x':80,'y':95,'width':95,'height':54,'anchor':'floor','rotation':-.6,'skew':.07,'lock_aspect':True}
  return rgb,m,p,asset,placement
 def test_holes_soft_edges_and_protection(self):
  m=np.ones((20,30),bool);p=np.zeros_like(m);p[5,5]=True;a=np.ones(m.shape,np.float32);a[8:12,12:18]=0;a[2,2]=.9999
  target,exposed,opaque=coverage_masks(m,p,a,0)
  self.assertTrue(target[9,13]);self.assertTrue(target[2,2]);self.assertFalse(target[5,5]);self.assertFalse(opaque[5,5]);self.assertTrue(opaque[1,1])
  widened,_,_=coverage_masks(m,p,a,2);self.assertGreater(widened.sum(),target.sum());self.assertFalse(widened[p].any())
 def test_invalid_coverage(self):
  m=np.zeros((10,10),bool)
  for a in [np.full(m.shape,np.nan),np.full(m.shape,1.1),np.zeros((9,10))]:
   with self.assertRaises(ValueError):coverage_masks(m,m,a)
  for margin in [-1,13,2.5,True]:
   with self.assertRaises(ValueError):coverage_masks(m,m,np.ones(m.shape),margin)
 def test_no_work_for_fully_covered_region(self):
  rgb,m,p,_,_=self.fixture();a=np.ones(m.shape,np.float32)
  bg,unknown,d=restore_background(rgb,m,p,{'restoration_scope':'visible','visibility_margin':0},coverage_alpha=a)
  self.assertTrue(np.array_equal(bg,rgb));self.assertEqual(d['roi']['solver_pixels'],0);self.assertEqual(d['visibility']['restoration_pixels'],0);self.assertTrue(unknown.any())
 def test_hidden_old_sofa_cannot_be_a_fill_donor(self):
  rgb,m,p,_,_=self.fixture();a=np.zeros(m.shape,np.float32);a[40:85,45:115]=1
  bg,unknown,d=restore_background(rgb,m,p,{'restoration_scope':'visible','visibility_margin':0,'repair_resolution':1600},coverage_alpha=a)
  target,_,_=coverage_masks(unknown,p,a,0)
  self.assertLess(np.abs(bg[target,0].astype(int)-bg[target,1].astype(int)).max(),2)
  self.assertTrue(np.array_equal(bg[~target],rgb[~target]));self.assertTrue(np.array_equal(bg[p],rgb[p]))
 def test_transformed_alpha_and_geometry_agree(self):
  rgb,m,p,asset,placement=self.fixture();layers={};opts={'restoration_scope':'visible','visibility_margin':2,'edge_feather':.4}
  out,diag=composite(Image.fromarray(rgb),m,asset,placement,opts,protected_mask=p,layers=layers)
  refined,_=refine_alpha(asset,opts);warped,norm=_warp_asset(refined,placement,(160,120));a=np.clip(warped[:,:,3],0,1);a[p]=0
  target,exposed,opaque=coverage_masks(np.array(layers['unknown'])>0,p,a,2)
  self.assertTrue(np.array_equal(np.array(layers['restoration-mask'])>0,target));self.assertTrue(np.array_equal(np.array(layers['replacement-coverage'])>0,opaque));self.assertEqual(diag['placement'],norm)
  self.assertTrue(np.array_equal(np.array(out)[p],rgb[p]));self.assertGreater(diag['background_repair']['visibility']['partially_transparent_pixels'],0)
 def test_cache_invalidation_on_move_and_reuse_on_return(self):
  rgb,m,p,asset,pos=self.fixture();opts={'restoration_scope':'visible','visibility_margin':0}
  with tempfile.TemporaryDirectory() as cache:
   def run(placement):return composite(Image.fromarray(rgb),m,asset,placement,opts,protected_mask=p,cache_dir=cache)[1]['background_repair']
   first=run(pos);second=run(dict(pos,x=pos['x']+20));third=run(pos)
   self.assertFalse(first['cache_hit']);self.assertFalse(second['cache_hit']);self.assertTrue(third['cache_hit']);self.assertNotEqual(first['cache_key'],second['cache_key']);self.assertEqual(len(list(Path(cache).glob('*.png'))),2)
 def test_manual_override_limited_to_exposed_and_persists_on_move(self):
  rgb,m,p,_,_=self.fixture();a=np.ones(m.shape,np.float32);a[85:95]=0;override=np.full_like(rgb,60);opts={'restoration_scope':'visible','visibility_margin':0}
  bg,u,_=restore_background(rgb,m,p,opts,override=override,coverage_alpha=a);target,_,_=coverage_masks(u,p,a,0)
  self.assertTrue((bg[target]==60).all());self.assertTrue(np.array_equal(bg[~target],rgb[~target]));a[75:95]=0
  moved,u,_=restore_background(rgb,m,p,opts,override=override,coverage_alpha=a);self.assertTrue((moved[80,80]==60).all())
 def test_hidden_old_region_rejected_as_manual_patch_donor(self):
  rgb,m,p,_,_=self.fixture();opts={'restoration_scope':'visible','background_patches':[{'source':[50,45,10,10],'target':[50,85]}]}
  with self.assertRaisesRegex(ValueError,'donor'):restore_background(rgb,m,p,opts,coverage_alpha=np.ones(m.shape))
 def test_partial_background_keeps_unknown_distinct_from_target(self):
  rgb,m,p,asset,pos=self.fixture();layers={};_,d=composite(Image.fromarray(rgb),m,asset,pos,{'restoration_scope':'visible'},protected_mask=p,layers=layers)
  self.assertGreater(np.count_nonzero(layers['unknown']),np.count_nonzero(layers['restoration-mask']));self.assertTrue(d['background_repair']['visibility']['background_is_partial'])

 def test_hidden_seams_are_not_reported_as_visible(self):
  from photo_simulator.visibility import visible_seam_metrics
  rgb=np.zeros((30,30,3),np.uint8);rgb[10:20,10:20]=255;target=np.zeros((30,30),bool);target[10:20,10:20]=True
  a=np.ones((30,30),np.float32);p=np.zeros_like(target)
  self.assertEqual(visible_seam_metrics(rgb,target,a,p)['visible_boundary_pairs'],0)
  a[:]=0;self.assertGreater(visible_seam_metrics(rgb,target,a,p)['mean_contribution_jump_0_255'],0)

 def test_fragmented_target_has_bounded_solver_regions(self):
  from photo_simulator.visibility import _regions
  target=np.zeros((80,80),bool);target[::2,::2]=True
  self.assertEqual(_regions(target,16),[[0,0,80,80]])

 def test_scoped_upload_does_not_reinsert_old_sofa_after_move(self):
  rgb,m,p,_,_=self.fixture();opts={'restoration_scope':'visible','visibility_margin':0};a=np.ones(m.shape,np.float32);a[85:95]=0
  old_scope,_,_=coverage_masks(m,p,a,0);upload=rgb.copy();upload[old_scope]=60;a[75:95]=0
  bg,u,d=restore_background(rgb,m,p,opts,override=upload,override_mask=old_scope,coverage_alpha=a)
  new_scope,_,_=coverage_masks(u,p,a,0);newly_visible=new_scope&~old_scope
  self.assertTrue((bg[old_scope]==60).all());self.assertLess(np.abs(bg[newly_visible,0].astype(int)-bg[newly_visible,1].astype(int)).max(),2)
  self.assertEqual(d['manual_override_applied_pixels'],int(old_scope.sum()))
