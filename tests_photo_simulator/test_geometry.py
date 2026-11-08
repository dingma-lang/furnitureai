import numpy as np
import unittest
from PIL import Image
from photo_simulator.compositor import placement_geometry, validate_quad, _warp_asset, composite

def test_locked_proportions_and_floor_anchor():
 p=dict(x=400,y=500,width=400,height=300,rotation=3,skew=.1,lock_aspect=True,anchor='floor')
 g=placement_geometry(p,(801,301),(1000,700))
 assert g['height']==150 and g['vertical_to_horizontal_scale']==1
 assert np.allclose(np.mean(g['corners'][2:],axis=0),[400,500])
 assert p['height']==300

BAD_QUADS = [
 [[0,0],[10,10],[10,0],[0,10]],
 [[0,0],[10,0],[10,0],[0,10]],
 [[0,0],[10,0],[5,2],[0,10]],
 [[0,0],[float('nan'),0],[10,10],[0,10]],
 [[0,0],[1e100,0],[1e100,10],[0,10]],
]
def test_reject_bad_quad(bad):
 with unittest.TestCase().assertRaises(ValueError):validate_quad(bad)

BAD_PLACEMENTS = [dict(skew=.46),dict(x=float('inf')),dict(width=0),dict(height=900),dict(anchor='physical'),dict(rotation=float('nan'))]
def test_reject_unsafe_transform(extra):
 with unittest.TestCase().assertRaises(ValueError):placement_geometry(dict(x=300,y=250,width=400,height=150,**{})|extra,(801,301),(1000,700))

def test_alpha_and_rgb_use_identical_warp_and_do_not_accumulate():
 a=np.zeros((100,200,4),np.uint8);a[20:80,30:170]=[255,0,0,255];asset=Image.fromarray(a)
 p=dict(x=300,y=250,width=350,height=180,rotation=-5,skew=.1)
 w,g=_warp_asset(asset,p,(600,500));w2,g2=_warp_asset(asset,p,(600,500))
 assert np.array_equal(w,w2) and g==g2
 assert np.allclose(w[:,:,0],w[:,:,3]) and not w[:,:,1:3].any()

def test_legacy_proportions_explicitly_preserved_and_warn():
 g=placement_geometry(dict(x=800,y=710,width=1120,height=550,rotation=2,skew=-.05),(1270,486),(1600,1067))
 assert g['height']==550 and not g['lock_aspect'] and g['warnings']

def test_contact_shadow_protected_pixels_and_original_immutable():
 room=Image.new('RGB',(200,150),'white');before=np.array(room).copy();mask=Image.new('L',room.size);mask.paste(255,(60,60,130,130));protect=Image.new('L',room.size);protect.paste(255,(60,60,80,140))
 result,diag=composite(room,mask,Image.new('RGBA',(100,60),(20,80,40,255)),dict(x=100,y=130,width=100,height=60,anchor='floor'),dict(shadow=True,shadow_mode='contact'),protected_mask=protect)
 assert np.array_equal(np.array(room),before)
 assert np.array_equal(np.array(result)[np.array(protect)>0],before[np.array(protect)>0])
 assert diag['shadow_bbox'] and diag['geometry']['physical_dimensions_verified'] is False

class GeometryTests(unittest.TestCase):
    pass

for name, function in list(globals().items()):
    if name.startswith("test_") and name not in ("test_reject_bad_quad", "test_reject_unsafe_transform"):
        setattr(GeometryTests,name,lambda self, f=function:f())
for i, q in enumerate(BAD_QUADS):
    setattr(GeometryTests,f"test_bad_quad_{i}",lambda self, q=q:test_reject_bad_quad(q))
for i, p in enumerate(BAD_PLACEMENTS):
    setattr(GeometryTests,f"test_bad_placement_{i}",lambda self, p=p:test_reject_unsafe_transform(p))
# Do not expose the loop's class alias to unittest discovery.
del function
