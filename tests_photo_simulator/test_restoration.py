import base64,io,tempfile,unittest,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from photo_simulator.restoration import restore_background,refine_alpha,contact_shadow
from photo_simulator.compositor import composite
from photo_simulator.server import Store


def fixture():
 yy,xx=np.mgrid[:120,:160];rgb=np.stack([150+xx//3,160+yy//3,np.full_like(xx,170)],axis=2).astype('uint8')
 mask=np.zeros((120,160),bool);mask[35:90,50:110]=True;protection=np.zeros_like(mask);protection[55:70,40:120]=True
 return rgb,mask,protection

def upload(image):
 b=io.BytesIO();image.save(b,format='PNG');return {'image':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()}

class RestorationTests(unittest.TestCase):
 def test_expansion_respects_protection_and_manual_subtraction(self):
  rgb,m,p=fixture();excluded=np.zeros_like(m);excluded[30:95,48:52]=True
  bg,unknown,info=restore_background(rgb,m,p,{'mask_expand':4,'exclude_foreground_donors':True},excluded)
  self.assertFalse(unknown[p|excluded].any());self.assertTrue(unknown[32,60]);self.assertTrue(np.array_equal(bg[~unknown],rgb[~unknown]))
 def test_protected_foreground_not_used_as_donor(self):
  rgb,m,p=fixture();second=rgb.copy();second[p]=[255,0,255]
  opts={'exclude_foreground_donors':True,'repair_resolution':1600}
  one,unknown,_=restore_background(rgb,m,p,opts);two,_,_=restore_background(second,m,p,opts)
  self.assertTrue(np.array_equal(one[unknown],two[unknown]));self.assertTrue(np.array_equal(two[p],second[p]))
 def test_methods_resolutions_keep_known_pixels(self):
  rgb,m,p=fixture()
  for method in ['ns','telea']:
   for res in [256,1600]:
    bg,u,d=restore_background(rgb,m,p,{'repair_method':method,'repair_resolution':res})
    self.assertEqual(bg.shape,rgb.shape);self.assertTrue(np.array_equal(bg[~u],rgb[~u]))
 def test_patch_mapping_and_unknown_provenance(self):
  rgb,m,p=fixture();opts={'background_patches':[{'source':[5,5,10,10],'target':[60,40],'feather':0}]}
  bg,u,d=restore_background(rgb,m,p,opts)
  self.assertTrue(np.array_equal(bg[40:50,60:70],rgb[5:15,5:15]));self.assertTrue(u[40:50,60:70].all());self.assertIn('unobserved',d['geometry'])
 def test_patch_rejects_unobserved_or_out_of_bounds_donors(self):
  rgb,m,p=fixture()
  for patch in [{'source':[60,40,10,10],'target':[80,80]},{'source':[150,0,20,20],'target':[80,80]},{'source':[0,0,10,10],'target':[155,115]}]:
   with self.assertRaises(ValueError):restore_background(rgb,m,p,{'background_patches':[patch]})
 def test_manual_override_applies_only_inside_unknown_mask(self):
  rgb,m,p=fixture();manual=np.full_like(rgb,30);bg,u,d=restore_background(rgb,m,p,{},override=manual)
  self.assertTrue((bg[u]==30).all());self.assertTrue(np.array_equal(bg[~u],rgb[~u]));self.assertTrue(d['manual_override'])
 def test_cache_ignores_placement_edge_shadow_and_invalidates_mask(self):
  rgb,m,p=fixture()
  with tempfile.TemporaryDirectory() as folder:
   a,u,d=restore_background(rgb,m,p,{},cache_dir=folder);b,_,e=restore_background(rgb,m,p,{'edge_feather':1,'shadow_opacity':.4},cache_dir=folder)
   self.assertFalse(d['cache_hit']);self.assertTrue(e['cache_hit']);self.assertTrue(np.array_equal(a,b))
   m[20,20]=True;_,_,f=restore_background(rgb,m,p,{},cache_dir=folder);self.assertFalse(f['cache_hit'])
 def test_edges_preserve_core_geometry_and_no_distant_halo(self):
  a=Image.new('RGBA',(100,60));d=ImageDraw.Draw(a);d.rectangle((10,10,90,50),fill=(0,100,0,255));d.rectangle((10,10,11,50),fill=(240,240,240,128))
  refined,info=refine_alpha(a,{'edge_feather':.55,'edge_contract':.35,'edge_decontam':.65});x=np.array(refined)
  self.assertEqual(refined.size,a.size);self.assertTrue(np.array_equal(x[15:45,15:85],np.array(a)[15:45,15:85]));self.assertEqual(x[:5,:,3].sum(),0);self.assertLess(x[25,10,0],240);self.assertTrue(info['opaque_core_unchanged'])
 def test_contact_shadow_controls_and_protection(self):
  alpha=np.zeros((120,160),np.float32);alpha[50:95,40:120]=1;p=np.zeros_like(alpha,bool);p[95:105,30:50]=True
  zero,_=contact_shadow(alpha,{'width':80},p,{'shadow_opacity':0});self.assertEqual(zero.sum(),0)
  a,d=contact_shadow(alpha,{'width':80},p,{'shadow_opacity':.4,'shadow_offset_y':5,'shadow_softness':2});self.assertGreater(a.sum(),0);self.assertFalse(a[p].any());self.assertEqual(d['contacts'][0][1],99)
 def test_invalid_controls_rejected(self):
  rgb,m,p=fixture()
  for opts in [{'mask_expand':float('nan')},{'repair_method':'fake'},{'mask_expand':13},{'repair_resolution':10}]:
   with self.assertRaises(ValueError):restore_background(rgb,m,p,opts)
  with self.assertRaises(ValueError):refine_alpha(Image.new('RGBA',(40,40)),{'edge_feather':float('inf')})
  with self.assertRaises(ValueError):contact_shadow(m.astype(float),{'width':80},p,{'shadow_opacity':2})
 def test_layers_geometry_and_original_immutable_without_models(self):
  rgb,m,p=fixture();room=Image.fromarray(rgb);asset=Image.new('RGBA',(80,40),(20,90,50,255));placement=dict(x=80,y=100,width=80,height=40,anchor='floor',rotation=-.6,skew=0,lock_aspect=True)
  layers={};_,a=composite(room,m,asset,placement,{'depth_assist':True},protected_mask=p,layers=layers)
  layers2={};_,b=composite(room,m,asset,placement,{'edge_feather':.5,'shadow':True,'shadow_mode':'contact_v2','depth_assist':True},protected_mask=p,layers=layers2)
  self.assertEqual(a['placement'],b['placement']);self.assertEqual(b['depth_status'],'unavailable');self.assertTrue(np.array_equal(layers['background'],layers2['background']));self.assertTrue(np.array_equal(room,rgb));self.assertEqual(layers2['unknown'].size,room.size)

class RestorationServiceTests(unittest.TestCase):
 def test_upload_history_reopen_and_invalid_dimensions(self):
  with tempfile.TemporaryDirectory() as folder:
   store=Store(folder,'cpu')
   try:
    p=store.create(upload(Image.new('RGB',(160,120),'beige')));id=p['id'];store.set_asset(id,upload(Image.new('RGBA',(80,40),(0,100,0,255))))
    override=store.set_background(id,upload(Image.new('RGB',(160,120),'gray')))['background_override'];saved=store.read(id);self.assertEqual(saved['background_override'],override)
    with self.assertRaises(ValueError):store.set_background(id,upload(Image.new('RGB',(80,60),'gray')))
    settings={'box':[30,30,120,100],'mask_edits':[{'mode':'add','points':[[40,40],[110,40],[110,90],[40,90]]}],'protected_edits':[],'placement':{'x':80,'y':80,'width':80,'height':40},'options':{}}
    for i in range(2):
     store.jobs[str(i)]={};settings['options']['mask_expand']=i;store.execute(str(i),id,'render',copy.deepcopy(settings));self.assertEqual(store.jobs[str(i)]['status'],'complete')
    latest=store.read(id);previous=store.history(id,{'direction':'undo'});self.assertNotEqual(latest['latest_result_url'],previous['latest_result_url']);self.assertEqual(previous['settings']['options']['mask_expand'],0)
    reopened=Store(folder,'cpu')
    try:
     restored=reopened.history(id,{'direction':'redo'});self.assertEqual(restored['latest_result_url'],latest['latest_result_url']);self.assertEqual(restored['background_override'],override)
    finally:reopened.pool.shutdown()
    self.assertTrue((Path(folder)/id/override['file']).exists());self.assertFalse(restored['physical_fit_verified'])
   finally:store.pool.shutdown()
