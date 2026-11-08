"""Review saved CC0 proposal and deterministic layouts without assets, Blender or network."""
import collections as C
import importlib.util
import shutil
from prep_core import *

def run():
 path=ROOT/'reports/phase7-data-strategy-v1/synthetic-pilot-v3.json';proposal=read(path)
 catalogue=read(ROOT/'reports/phase7-data-strategy-v1/sources/polyhaven-furniture.json');all_models=read(ROOT/'reports/phase7-data-strategy-v1/sources/polyhaven-all-models.json')
 ideas={'sofa':['sofa_03','chinese_sofa'],'chair':['dining_chair_02','plastic_monobloc_chair_01'],
  'table':['coffee_table_round_01','wooden_table_02'],'bed':['vintage_day_bed'],
  'desk':['SchoolDesk_01'],'cabinet':['chinese_cabinet','painted_wooden_cabinet_02'],
  'lamp':['industrial_pipe_lamp','industrial_wall_lamp','modern_ceiling_lamp_01']}
 suggestions={}
 for group,ids in ideas.items():
  suggestions[group]=[dict(asset_id=aid,name=all_models[aid]['name'],creator=all_models[aid]['authors'],source_url='https://polyhaven.com/a/'+aid,
   rights='Saved Poly Haven CC0 policy/catalogue; file availability/asset body not verified',download_bytes=None,scope='Subtype/complete-unit and multi-object inspection required before labelling') for aid in ids if aid in all_models]
 views=proposal['scene_plans'];presence=C.Counter();layouts={}
 for view in views:
  presence.update({o['category'] for o in view['objects']});layouts[view['scene_id']]=view['room_type']
 spec=importlib.util.spec_from_file_location('phase8a_saved_synthetic',ROOT/'tools/phase7-v1/synthetic.py');module=importlib.util.module_from_spec(spec)
 # This module imports sealed historical contracts; prep_core deliberately uses a different name.
 import sys
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'));spec.loader.exec_module(module)
 if module.scenes()!=views:raise ValueError('Saved100-view seeds/layouts are not reproducible')
 resources=dict(planned_views=100,independent_layouts=25,planned_object_appearances=sum(len(v['objects']) for v in views),actually_rendered_images=0,
  resolution=[640,480],cycles_samples=32,view_presence_counts=dict(presence),unique_planned_external_asset_models_by_class={c:sum(a['category']==c for a in proposal['assets']) for c in CLASSES},
  actual_asset_models_downloaded=0,procedural_lamp_geometries=1,
  room_type_layouts=dict(C.Counter(layouts.values())),file_objects=len(proposal['unique_objects']),expected_asset_body_bytes=proposal['expected_asset_body_bytes'],
  installed_Blender=shutil.which('blender'),installed_BlenderProc=bool(importlib.util.find_spec('blenderproc')),
  frame_time_projection_seconds=[5,60],time_projection_not_benchmark=True,render_wall_clock_budget_proposal_seconds=7200,
  soft_VRAM_monitor_target_GiB=6,hard_VRAM_reservation_implemented=False,render_output_budget_bytes=1073741824,
  final_100_image_storage_projection_bytes=[268435456,644245094],temporary_EXR_cleanup_and_depth_validation_require_smoke_test=True,
  runtime_install_download_bytes=None,runtime_installed_storage_bytes=None,current_Python_environment_changed=False)
 commit(OUT/'synthetic-pilot-assessment.json',dict(source_proposal_sha256=sha(path),resources=resources,suggested_additional_assets_from_saved_catalogue=suggestions,
  seven_labels_representable_in_plans=True,adequate_product_or_style_diversity=False,real_world_development_replacement=False,
  caveats=['One external asset/model per six categories and one procedural lamp is a technical pilot, not adequate class diversity','100 views share25 layouts; do not call them100 independent rooms/products',
   'No known dedicated floor-lamp mesh verified in cached collection; wall/ceiling/street/oil lamps must not be silently mapped to floor lamps',
   'old_bed_frame is not a verified complete bed; add a mattress only as explicitly authored geometry or a verified separate asset',
   'School desk may be a specialised/multi-object unit; file/body inspection required before automatic instance labels',
   'Current BlenderProc adapter has not been run in Blender; APIs, multi-mesh instance IDs, glTF units, camera axes, alpha handling and depth registration remain unverified',
   'Instance-raster visible boxes/RLE export can be tested offline but actual renderer correctness requires an approved smoke subset','BlenderProc launcher may auto-install/download before script approval checks; never launch it now',
   'Project Python3.14 is not the Blender bundled Python runtime; use separately approved isolated installation without changing .venv']))
 commit(OUT/'approval-C1-exact-CC0-asset-bundle.json',dict(approved=False,scope='Acquire only saved six-asset33-object bundle and verify files; no installation or rendering',
  source_manifest=str(path.relative_to(ROOT)),source_manifest_sha256=sha(path),assets=proposal['assets'],unique_objects=proposal['unique_objects'],
  expected_response_body_bytes=11026401,max_cumulative_response_body_bytes=25000000,max_attempts_per_object=2,max_new_allocated_storage_bytes=134217728,
  destination='data/synthetic/assets/phase8a-v1/',models_or_textures_source='Only exact pinned https://dl.polyhaven.org/file/ph-assets/Models/ URLs; no alternate collections',
  licence='Saved per-asset catalogue creators and Poly Haven CC0 policy; reverify current official metadata before body acquisition',api_credit='Powered by Poly Haven',
  verify_published_sizes_and_MD5=True,compute_SHA256=True,stop_on_source_change=True,Blender_installation=False,rendering=False,training=False))
 commit(OUT/'approval-C2-runtime-metadata-preflight.json',dict(approved=False,scope='Only runtime documentation/installer-metadata planning; no binary or package acquisition or installation',
  exact_urls=['https://pypi.org/pypi/blenderproc/json','https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/blenderproc/python/utility/InstallUtility.py','https://download.blender.org/release/'],
  max_cumulative_response_body_bytes=5000000,max_attempts_per_object=2,max_new_allocated_storage_bytes=20971520,destination='data/synthetic/runtime-preflight/phase8a-v1/',
  deliverable='Exact compatible Blender/BlenderProc/Python/dependency versions, official installer/wheel URLs, sizes, hashes and separate installation budget',
  binary_or_model_downloads=False,installations=False,rendering=False,training=False))
 commit(OUT/'approval-C3-conditional-render-pilot.json',dict(approved=False,execution_ready=False,scope='Conditional100-view technical synthetic pilot only, never real evaluation',
  exact_scene_plan=str(path.relative_to(ROOT)),plan_sha256=sha(path),views_max=100,layouts=25,seed=20261106,smoke_views_included_in_100=5,
  resolution=[640,480],samples=32,wall_seconds_max=7200,one_render_process=True,max_new_allocated_render_storage_bytes=1073741824,
  destination='data/synthetic/image-pilots/phase8a-v1/',synthetic_partition='train_only',no_real_dev_or_frozen_membership=True,
  prerequisites=['C1 asset verification approval/completion','C2 metadata preflight and separately approved exact runtime installation','Source/native semantic policy for procedural lamp and each model','Approved five-frame RGB/mask/depth API-registration smoke checks'],
  runtime_installation_approval_request_ready=False,soft_VRAM_target_GiB=6,stop_on_OOM=True,extra_asset_downloads=False,training=False))
 commit(OUT/'approval-D-training-status.json',dict(training_approval_requested_now=False,ready_for_controlled_seven_class_training=False,
  reasons=['Cabinet has0 fresh-development photos','Lamp has1 and Bed2 fresh-development photos','Shared category/unit policy unresolved','Prepared loss-mask reference is not a tested production training adapter'],
  next_config_requirements=['Versioned source-native manifest and tested partial-label adapter','Adequate grouped fresh-development coverage for every scored class','Frozen baseline/checkpoint comparison protocol without frozen optimisation','Separately approved exact head-only update/epoch/configuration budget'],
  GPU='RTX4070 Laptop8GiB; future head-only batch1/accumulation is plausible from prior experiments, not a new memory benchmark',
  fine_tuning_or_encoder_updates=False))
 print('SYNTHETIC REVIEW:100 plans/25 layouts;6 single-model classes+procedural lamp; no actual rendering',flush=True)

if __name__=='__main__':run()
