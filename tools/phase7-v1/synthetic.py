"""Offline CC0 bundle specification, seeded scene plans and visible-mask export."""
import argparse
import math
from pathlib import PurePosixPath
import random
from urllib.parse import urlparse
from contracts import CATEGORIES, OUT, read, sha, write_new

ASSETS = {'sofa_02':'sofa','modern_arm_chair_01':'chair','dining_table':'table',
          'GothicBed_01':'bed','metal_office_desk':'desk','modern_wooden_cabinet':'cabinet'}

def asset_plan():
    catalogue=read(OUT/'sources/polyhaven-all-models.json');assets=[];objects={}
    for aid,category in ASSETS.items():
        source=OUT/'sources'/('files-'+aid+'.json');files=read(source)
        node=files['gltf']['1k']['gltf']
        dependencies=[('asset.gltf',node)]+list(node.get('include',{}).items())
        entries=[]
        for relative,info in dependencies:
            path=PurePosixPath(relative);url=urlparse(info['url'])
            if path.is_absolute() or '..' in path.parts or '\\' in relative or url.scheme != 'https' or url.netloc != 'dl.polyhaven.org' or not url.path.startswith('/file/ph-assets/Models/') or url.query or url.fragment:
                raise ValueError('Unsafe asset reference')
            if not isinstance(info['size'],int) or info['size'] <= 0 or len(info['md5']) != 32:
                raise ValueError('Invalid published asset integrity')
            entries.append(dict(relative_path=relative,**{k:info[k] for k in ('url','size','md5')}))
            previous=objects.setdefault(info['url'],dict(url=info['url'],bytes=info['size'],published_md5=info['md5']))
            if previous['bytes'] != info['size'] or previous['published_md5'] != info['md5']:
                raise ValueError('Contradictory asset object')
        assets.append(dict(asset_id=aid,category=category,source_page='https://polyhaven.com/a/'+aid,
            creator=catalogue[aid]['authors'],license_url='https://creativecommons.org/publicdomain/zero/1.0/',
            license_evidence='sources/polyhaven-license.html',license_evidence_sha256=sha(OUT/'sources/polyhaven-license.html'),
            catalogue_dimensions=catalogue[aid].get('dimensions'),dimension_interpretation='Millimetre-like catalogue values; verify against imported geometry before accepting renders',
            metadata_sha256=sha(source),resolution='1k',files=entries,
            credit='Poly Haven; '+', '.join(catalogue[aid]['authors']),approved_for_download=False))
    return dict(schema='phase7-cc0-bundle-proposal-v1',assets=assets,unique_objects=list(objects.values()),
        expected_asset_body_bytes=sum(x['bytes'] for x in objects.values()),asset_count=len(assets),
        photograph_downloads=0,approved=False,installation_approved=False,rendering_approved=False,
        note='1k glTF plus every listed dependency; main glTF is stored as asset.gltf. Later inspect actual glTF URIs against this exact allowlist, forbid network resolution. Published MD5 is transport evidence, not SHA-256. Compute SHA-256 after acquisition.',
        lamp='Catalogue contains desk_lamp_arm_01, but its file API returned HTTP403; do not bypass. Pilot instead uses a clearly identified original procedural lamp; no claim of a downloaded CC0 lamp asset.',
        required_api_credit='Powered by Poly Haven',
        destination='data/synthetic/preparation/phase7-assets-v1/')

def intersects(a,b,margin=0.12):
    return abs(a['x']-b['x'])*2 < a['sx']+b['sx']+2*margin and abs(a['y']-b['y'])*2 < a['sy']+b['sy']+2*margin

def scenes(seed=20261106, count=100):
    if not 1 <= count <= 100:
        raise ValueError('Preparation pilot is bounded to 100 views')
    shapes={'sofa':(1.85,.85,.85),'chair':(.7,.75,1.05),'table':(1.5,.95,.78),'bed':(2.1,1.65,1.3),
            'desk':(1.4,.75,.8),'cabinet':(1.1,.6,1.65),'lamp':(.45,.45,1.5)}
    result=[]
    for room_id in range((count+3)//4):
        rng=random.Random(seed+room_id);width=rng.uniform(6,8);length=rng.uniform(6,8);objects=[]
        templates=[('living_room',{'sofa','chair','table','cabinet','lamp'}),
                   ('bedroom',{'bed','chair','cabinet','lamp'}),
                   ('office',{'desk','chair','cabinet','lamp'}),
                   ('studio',set(CATEGORIES))]
        room_type,room_categories=templates[room_id%len(templates)]
        ordered=[(a,c) for a,c in list(ASSETS.items())+[('procedural_floor_lamp_v1','lamp')] if c in room_categories]
        for index,(aid,category) in enumerate(ordered,1):
            sx,sy,sz=shapes[category];rotation=rng.choice([0,math.pi/2,math.pi,3*math.pi/2])
            if round(rotation/(math.pi/2))%2:sx,sy=sy,sx
            for attempt in range(1000):
                obj=dict(instance_id=index,asset_id=aid,category=category,x=rng.uniform(sx/2+.3,width-sx/2-.3),
                         y=rng.uniform(sy/2+.3,length-sy/2-.3),sx=sx,sy=sy,sz=sz,rotation_z=rotation)
                if not any(intersects(obj,old) for old in objects):objects.append(obj);break
            else:raise ValueError('Cannot fit approved room objects without intersection')
        for view in range(4):
            if len(result)==count:break
            vrng=random.Random(seed*1000+room_id*4+view)
            # Four perimeter views, keeping the camera above the floor and inside the walls.
            corners=[(.25,.25),(.75,.25),(.75,.75),(.25,.75)]
            cx,cy=corners[view]
            result.append(dict(schema='phase7-synthetic-scene-plan-v1',seed=seed+room_id,
                render_seed=seed*1000+room_id*4+view,scene_id=f'phase7-seed-{seed}-room-{room_id:03d}',view_id=view,
                domain='synthetic',partition='train',width=640,height=480,
                room_type=room_type,room_metres=[width,length,vrng.uniform(2.5,3.2)],objects=objects,
                camera_position=[cx*width,cy*length,vrng.uniform(1.35,1.8)],
                camera_target=[width/2,length/2,.9],fov_degrees=vrng.uniform(60,85),
                light_energy_watts=vrng.uniform(350,1400),light_colour=[1,vrng.uniform(.75,1),vrng.uniform(.65,1)],
                samples=32,visible_instances_verified=False,
                caveat='Layouts and occlusion opportunities only; visibility, realism and segmentation must be verified after rendering'))
    return result

def visible_annotations(instance_map, instance_lookup):
    """Exact raster-visible xywh and COCO uncompressed RLE. Zero visibility yields no box."""
    import numpy as np
    arr=np.asarray(instance_map)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype,np.integer) or np.any(arr<0):
        raise ValueError('Expected nonnegative integer instance raster')
    if set(int(v) for v in np.unique(arr)) - {0} - set(instance_lookup):
        raise ValueError('Unknown rendered instance identity')
    result=[]
    for iid,meta in sorted(instance_lookup.items()):
        mask=arr==iid;y,x=np.where(mask)
        if not len(x):continue
        flat=mask.ravel(order='F');counts=[];previous=False;run=0
        for value in flat:
            if bool(value)==previous:run+=1
            else:counts.append(run);run=1;previous=bool(value)
        counts.append(run)
        result.append(dict(instance_id=iid,category=meta['category'],asset_id=meta['asset_id'],
            bbox_xywh=[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],
            segmentation={'size':list(arr.shape),'counts':counts},segmentation_format='coco_uncompressed_rle',
            area=int(mask.sum()),box_definition='visible_mask_extent_not_amodal',iscrowd=0))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    plan=asset_plan();views=scenes();plan['scene_plans']=views
    plan['rendered_images']=0;plan['planned_views']=len(views);plan['independent_layouts']=len({r['scene_id'] for r in views})
    write_new(args.output,plan)

if __name__=='__main__':main()
