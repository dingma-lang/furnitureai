"""Record actual assistant sheet inspection and conservative source-geometry gaps."""
import math
from collections import Counter
import numpy as np
from common7 import *

FINDINGS=[
('mixed_appliance_reflection','The crop contains a glass/plant-reflection/laptop-screen boundary;native laptop context does not establish full-region Chair absence.'),
('small_dark_unknown','A small dark glossy patch with highlights is visible among tabletop objects;its precise identity is unresolved.'),
('small_dark_unknown','The dark tabletop/equipment-like patch is not a whole identifiable Chair;source-mask containment is not established.'),
('small_kitchen_detail','A tiny kitchen/refrigerator-area detail is visible,but scale and incomplete source-region support prevent a trustworthy negative.'),
('kitchen_clutter','The crop contains sink/dish-rack or kitchen clutter;the native sink bbox is not a fully certified visible negative region.'),
('body_table_boundary','A small hand/body/wood-table boundary is visible near a seated person;surrounding seating remains protected.'),
('dark_clothing_boundary','A very narrow dark person/clothing boundary is visible;the exact background and furniture relation are uncertain.'),
('person_background_boundary','The crop contains a person/clothing/table-background boundary in a room with seating;no absence label is approved.'),
('dark_equipment_clutter','Dark glossy tabletop/equipment-like clutter is visible;native person bbox context is only an automatic spatial cue.'),
('counter_background_strip','A narrow counter/background strip appears behind a person;the query does not tightly identify the native person.'),
('dark_object_background','A dark object/background boundary under the window is visible;possible furniture/reflection content remains unresolved.'),
('mixed_glossy_objects','The crop spans several dark glossy objects and highlights;it cannot be assigned a confirmed negative from the person bbox.'),
('toilet_fragment','The visible crop includes a toilet-tank/bathroom-object fragment,consistent with native toilet context,but full source-polygon containment still failed.'),
('tiny_kitchen_background','The crop is a tiny kitchen/counter-area patch;there is insufficient resolution and visible-region evidence for approval.'),
('dark_glass_boundary','A narrow dark/glass/screen boundary is visible;the native laptop bbox includes mixed background content.'),
('bright_glare_or_display','A bright blue/glare/display-like background patch is visible above people;native person context does not identify the crop content.'),
('decoration_or_reflection','The small crop is around mantel decorations or reflection;its complete contents and source-mask support remain uncertain.'),
('plain_background_patch','A small green/dark background patch appears by the window;automatic person-bbox context is not visible object identity.'),
('body_brick_boundary','The tiny crop lies along a person/brick-background edge;seating context and full mask coverage remain unresolved.'),
('plant_detail','The crop contains plant/flower-like detail,consistent with potted-plant context,but the whole model box is not source-certified.'),
('person_clothing_fragment','The crop contains a standing person clothing/jeans fragment;source bbox overlap alone is insufficient for the full query region.'),
('real_Chair_mixed_with_Table','The crop clearly contains a real dining Chair and table;this high-confidence prediction must not be labelled Chair-negative.'),
('real_Chair_fragment','The crop shows an upholstered/wood dining Chair back and frame;it remains protected despite a broad native House context.'),
('real_Chair','The crop clearly shows a dining Chair in a bedroom;native House context does not make it background or non-Chair.')]

def run():
    samples=read(RUN/'analysis/inspection-samples.json');rows=samples['records'];pages=samples['pages']
    if len(rows)!=24 or len(FINDINGS)!=24:raise ValueError('Exact actually viewed sample set required')
    reviews=[]
    for row,(finding,evidence) in zip(rows,FINDINGS):
        page=next(p for p in pages if row['sample_id'] in p['sample_ids'])
        if sha(ROOT/page['path'])!=page['sha256']:raise ValueError('Actually viewed sheet changed')
        reviews.append(dict(sample_id=row['sample_id'],image_id=row['image_id'],query_index=row['query_index'],candidate_id=row['candidate_id'],
             source_sha256=row['source_sha256'],page=page['path'],page_sha256=page['sha256'],finding=finding,evidence=evidence,
             assistant_actual_visual_inspection=True,human_reviewed=False,negative_label_approved=False,training_activated=False))
    emit(RUN/'analysis/visual-observations.json',dict(records=reviews,human_reviews_created=0,new_training_labels=0))
    sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
    from geometry6 import polygons,certify
    photos={p['image_id']:p for p in read(ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json')['photos']}
    obs=[json.loads(s) for s in (RUN/'analysis/high-confidence-Chair.jsonl').read_text().splitlines()]
    B=[r for r in obs if r['evidence_tier']=='B_automatic_unverified'];diagnostics=[]
    for r in B:
        a=next(a for a in photos[r['image_id']]['annotations'] if a['annotation_key']==r['native_annotation_key'])
        b=r['box'];px=[math.floor(b[0]*r['width']),math.floor(b[1]*r['height']),math.ceil(b[2]*r['width']),math.ceil(b[3]*r['height'])]
        p=polygons(a['original_annotation'],r['width'],r['height'])
        diagnostics.append(dict(candidate_id=r['candidate_id'],image_id=r['image_id'],native_label=a['native_label'],
             native_has_mask=a['has_mask'],valid_simple_source_polygons=len(p),rectangle_pixels=px,minimum_integer_side=min(px[2]-px[0],px[3]-px[1]),
             passes_source_polygon_2pixel_margin=any(certify(px,poly) for poly in p),training_eligible=False))
    emit(RUN/'analysis/source-geometry-gaps.json',dict(records=diagnostics,E6_policy_unchanged=True))
    side=np.asarray([[(r['box'][2]-r['box'][0])*r['width'],(r['box'][3]-r['box'][1])*r['height']] for r in B])
    summary=dict(automatic_unverified_candidates=len(B),photos=len({r['image_id'] for r in B}),source_groups=len({r['source_group'] for r in B}),
         contexts=dict(Counter(r['source_context'] for r in B)),native_labels=dict(Counter(r['native_label'] for r in B)),
         ge08=sum(r['student_Chair_score']>=.8 for r in B),minimum_Chair_score=min(r['student_Chair_score'] for r in B),maximum_Chair_score=max(r['student_Chair_score'] for r in B),
         minimum_side_below32_pixels=int((side.min(1)<32).sum()),median_width_height_pixels=np.median(side,axis=0).tolist(),
         native_source_mask_available=sum(r['native_has_mask'] for r in diagnostics),has_supported_simple_source_polygons=sum(r['valid_simple_source_polygons']>0 for r in diagnostics),
         selected_source_polygon_full_containment_2px_passes=sum(r['passes_source_polygon_2pixel_margin'] for r in diagnostics),
         actual_assistant_visual_samples=24,human_reviews_created=0,new_verified_hard_negatives=0,
         independent_seven_category_acceptance_photographs=0,
         training_data_acquisition='No newly trusted high-confidence coverage was established. Additional source-complete diverse room data is likely needed,but necessity is not proven while21unverified query footprints and42inactive local photos remain to qualify;no downloads authorized.',
         next_action='CPU-only source-evidence qualification of the21actual E5query candidates and local native masks,including exact box/mask correspondence and small-region exclusions;preserve real seating and keep all candidates inactive until supported. No training or another loss variant.')
    emit(REPORT/'candidate-review-summary.json',summary)
    print('Recorded24actual assistant inspections and21source-geometry gaps;zero human reviews or activated labels')

if __name__=='__main__':run()
