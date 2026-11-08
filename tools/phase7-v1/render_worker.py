"""Prepared BlenderProc adapter, NOT executed during Phase 7 preparation.

Run only in a separately approved Blender environment after exact asset acquisition.
BlenderProc's launcher may install dependencies: do not launch it before approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

def approved_inputs(bundle, approval, assets):
    """Pure validation: pinned user approval and every local asset object, no network."""
    from contracts import sha
    data=json.loads(Path(bundle).read_text());auth=json.loads(Path(approval).read_text())
    if auth.get('synthetic_rendering_approved') is not True or auth.get('bundle_sha256') != sha(bundle):
        raise ValueError('Separate approval bound to this exact bundle is required')
    if not 1 <= auth.get('max_images',0) <= 100:
        raise ValueError('Invalid approved image bound')
    base=Path(assets).resolve()
    for a in data['assets']:
        for f in a['files']:
            unresolved=base/a['asset_id']/f['relative_path'];path=unresolved.resolve()
            if not path.is_relative_to(base) or any(p.is_symlink() for p in (unresolved,*unresolved.parents)) or path.stat().st_size != f['size']:
                raise ValueError('Unsafe/missing asset object')
            if hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest() != f['md5']:
                raise ValueError('Published asset hash mismatch')
        gltf=json.loads((base/a['asset_id']/'asset.gltf').read_text())
        allowed={r['relative_path'] for r in a['files']}
        for item in gltf.get('buffers',[])+gltf.get('images',[]):
            if 'uri' in item and item['uri'] not in allowed:
                raise ValueError('Unapproved glTF URI; no external resolution permitted')
        if gltf.get('extensionsRequired'):
            raise ValueError('glTF extensions require separate loader review')
    return data,auth

def run(bundle, approval, assets, output):
    from contracts import sha,write_new
    from synthetic import visible_annotations
    data,auth=approved_inputs(bundle,approval,assets)
    out=Path(output).resolve()
    if not out.is_relative_to(Path(__file__).resolve().parents[2]/'data/synthetic'):
        raise ValueError('Synthetic outputs must stay under project data/synthetic')
    if out.exists():raise FileExistsError(out)
    # Imports deliberately occur after local safety checks. This code is an unbenchmarked adapter.
    import blenderproc as bproc
    import bpy
    import numpy as np
    from PIL import Image
    out.mkdir(parents=True)
    for index,plan in enumerate(data['scene_plans'][:auth['max_images']]):
        if (out/'STOP').exists():break
        if sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file()) > 900*1024**2:
            raise ValueError('Synthetic output safety reserve exhausted')
        bproc.clean_up();bproc.init();np.random.seed(plan['render_seed'] % 2**32)
        bpy.context.scene.render.engine='CYCLES';bpy.context.scene.cycles.seed=plan['render_seed'] % 2**31
        bproc.renderer.set_max_amount_of_samples(plan['samples'])
        w,l,h=plan['room_metres']
        room=[([w/2,l/2,-.05],[w/2,l/2,.05]),([-.05,l/2,h/2],[.05,l/2,h/2]),
              ([w+.05,l/2,h/2],[.05,l/2,h/2]),([w/2,-.05,h/2],[w/2,.05,h/2]),([w/2,l+.05,h/2],[w/2,.05,h/2])]
        for loc,scale in room:
            obj=bproc.object.create_primitive('CUBE',location=loc,scale=scale);obj.set_cp('category_id',0);obj.set_cp('object_id',0)
        lookup={}
        for obj in plan['objects']:
            iid=obj['instance_id'];category=obj['category'];lookup[iid]=obj
            if obj['asset_id']=='procedural_floor_lamp_v1':
                parts=[]
                for radius,depth,z in [(.2,.04,.02),(.025,1.2,.64),(.23,.30,1.38)]:
                    bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=radius,depth=depth,location=(0,0,z));parts.append(bpy.context.object)
            else:
                before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(Path(assets)/obj['asset_id']/'asset.gltf'))
                parts=[o for o in set(bpy.data.objects)-before if o.type=='MESH']
            if not parts:raise ValueError('Asset has no mesh')
            bpy.ops.object.select_all(action='DESELECT')
            for part in parts:part.select_set(True)
            bpy.context.view_layer.objects.active=parts[0];bpy.ops.object.join();joined=bpy.context.object
            # Bake parent transforms before a common origin/scale. glTF axes have been converted by Blender.
            world=joined.matrix_world.copy();joined.parent=None;joined.matrix_world=world
            bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
            if min(joined.dimensions) <= 0:raise ValueError('Degenerate furniture geometry')
            # Floor-plan extents already account for right-angle rotation. Fit local dimensions accordingly.
            localx,localy=obj['sx'],obj['sy']
            if round(obj['rotation_z']/(np.pi/2))%2:localx,localy=localy,localx
            joined.dimensions=(localx,localy,obj['sz']);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
            joined.rotation_euler[2]=obj['rotation_z'];joined.location=(obj['x'],obj['y'],obj['sz']/2)
            mesh=bproc.types.MeshObject(joined);mesh.set_cp('category_id',1+list(__import__('contracts').CATEGORIES).index(category));mesh.set_cp('object_id',iid)
        light=bproc.types.Light();light.set_type('AREA');light.set_location([w/2,l/2,h-.2]);light.set_energy(plan['light_energy_watts']);light.set_color(plan['light_colour'])
        position=np.array(plan['camera_position']);direction=np.array(plan['camera_target'])-position
        rotation=bproc.camera.rotation_from_forward_vec(direction)
        bproc.camera.set_resolution(plan['width'],plan['height'])
        bpy.context.scene.camera.data.type='PERSP';bpy.context.scene.camera.data.angle=np.deg2rad(plan['fov_degrees'])
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(position,rotation))
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(map_by=['instance','object_id'],default_values={'object_id':0})
        rendered=bproc.renderer.render(output_dir=str(out/f'tmp-{index:03d}'))
        raw=rendered['instance_segmaps'][0];mapping=rendered['instance_attribute_maps'][0]
        instances=np.zeros(raw.shape,dtype=np.uint16)
        for item in mapping:instances[raw==item['idx']]=int(item['object_id'])
        annotations=visible_annotations(instances,lookup)
        stem=f'{index:03d}';Image.fromarray(np.asarray(rendered['colors'][0],dtype=np.uint8)).save(out/(stem+'.png'))
        np.save(out/(stem+'-instances.npy'),instances,allow_pickle=False)
        depth=np.asarray(rendered['depth'][0],dtype=np.float32);np.save(out/(stem+'-depth.npy'),depth,allow_pickle=False)
        write_new(out/(stem+'.json'),dict(schema='phase7-render-candidate-v1',plan=plan,domain='synthetic',partition='train',
            annotations=annotations,depth_units='metres',depth_definition='camera_z_non_antialiased',
            image_sha256=sha(out/(stem+'.png')),instance_sha256=sha(out/(stem+'-instances.npy')),depth_sha256=sha(out/(stem+'-depth.npy')),
            bundle_sha256=sha(bundle),status='pending_post_render_integrity_visibility_and_realism_checks',human_verified=False,
            synthetic_ground_truth_only=True,renderer_versions={'blender':bpy.app.version_string,'blenderproc':getattr(bproc,'__version__','unknown')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.bundle,args.approval,args.assets,args.output)
