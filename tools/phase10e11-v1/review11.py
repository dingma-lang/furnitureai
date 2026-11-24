"""Assistant visual observations, not human reviews, source annotations or labels."""
import html
from common11 import *
OBS={
 'efqgz854':dict(scene='Historical institutional kitchen; two people preparing food',
  definite=[],possible=['Table','Cabinet'],chairs='No definite chair established; clutter and occlusion prevent absence certification',
  observations='Freestanding preparation surface behind workers, long foreground counter/island, drawers, ovens, pots, pipes, clock and ceiling globes. Prep tables/counters and cabinetry need whole-object taxonomy decisions.',
  distractors='People, cookware, appliance fronts, pipes, sinks, repeated handles and pendant fixtures',
  quality='Sharp room layout with grain, deep shadows and some clipped highlights. Foreground counter is truncated; distant/occluded work surfaces are harder to box.',
  annotation='Protect people and any ambiguous seating; review prep table vs counter and island vs whole Cabinet. Ceiling globes are not standalone Lamp positives.',
  primary_medical_subject=False,people_visible=True),
 'bpxzpgg7':dict(scene='Historical laboratory, close view of a cluttered work surface',
  definite=['Chair'],possible=['Table','Cabinet'],chairs='A curved chair back behind the work surface is visible; foreground curved seating fragment also needs protection',
  observations='Scientific glassware, shelves, cabinets/book storage and a worktable fill the frame; genuine seating is partly hidden. Do not mark the bench area wholesale Chair-negative.',
  distractors='Bottles, tubing, scientific apparatus, shelf edges, papers and chair-like curved forms',
  quality='Fine grain and usable detail, but strong shadows, bright windows, cut-off foreground furniture and partial chairs.',
  annotation='Review each visible chair fragment; distinguish furniture from scientific apparatus. Full object extent and Table/Desk/bench/Cabinet semantics remain unresolved.',
  primary_medical_subject=False,people_visible=False),
 'x6v3kwb7':dict(scene='Historical laboratory, longer room view with glass-front storage',
  definite=[],possible=['Chair','Table','Cabinet'],chairs='Small dark stool-like form near the back-left bench and curved fittings are ambiguous; protect pending closer evidence',
  observations='Long laboratory bench, drawers, glass-front storage, bottles, pipes and instruments. Could be another view of the laboratory in the other two photographs; room identity not proven.',
  distractors='Plumbing, glassware, drawer fronts, scientific apparatus and repeated vertical supports',
  quality='Cabinet structures clear but left shadows and window highlights obscure small seating/objects; right edge clips storage.',
  annotation='Do not convert ambiguous stool-like forms to negatives. Separate built-in storage/fronts and workbench from whole Cabinet/Table/Desk labels.',
  primary_medical_subject=False,people_visible=False),
 'y4n8bmat':dict(scene='Historical laboratory, opposite-side view with windows and work island',
  definite=['Chair'],possible=['Table','Cabinet'],chairs='Bentwood seating behind the central workbench is visible; left-midframe seating/curved object needs review',
  observations='Real seating, central drawer workbench, glass-front cases, hanging fittings and many instruments. Likely related room/capture context to the other laboratory views, not an independent room claim.',
  distractors='Scientific glassware, tubing, hanging fittings, cabinet divisions, drawers and supports',
  quality='Good architectural detail with glare, shadowed cabinets, foreground truncation and partial seating.',
  annotation='Protect visible/possible seating before any hard-negative proposal. Review complete cabinet extent, workbench mapping and fixture semantics.',
  primary_medical_subject=False,people_visible=False),
 'vugqdsr7':dict(scene='Historical library/reading or museum room; one person reading',
  definite=['Chair','Table'],possible=[],chairs='Several upholstered/wood-frame chairs are clearly visible, including occupied/partly occluded seating',
  observations='Large reading table, chairs, wall bookcases, glass-topped display case, suspended lights and telephone/guide equipment. Closest of the seven to a generic office/reading room, but still historical monochrome.',
  distractors='Book rows, display case legs, telephone cords, light fittings and seated person',
  quality='Best general furniture visibility in the pilot; large clear table/chairs. Some occlusion and frame-edge cuts remain.',
  annotation='Table and Chair boxes are feasible after actual annotation review. Bookcases/display case are not automatically Cabinet; hanging fixtures not standalone Lamp. Address identifiable-person reuse conditions.',
  primary_medical_subject=False,people_visible=True),
 'mhnzfe6b':dict(scene='Historical single-patient hospital room, apparently unoccupied',
  definite=['Chair','Bed'],possible=['Table','Cabinet'],chairs='Central upholstered armchair and cropped right-side wooden chair are visible',
  observations='One hospital bed, armchair, partial chair, bedside phone cabinet, partial tray/worktop, sink, flowers and wall fixtures. A room scene rather than a medical-body subject.',
  distractors='Telephone, sink, grab/bed fittings, flowers, curtains and wall fixtures',
  quality='Large useful Bed/Chair shapes with monochrome grain. Bed/tray/chair ends are clipped or overlap; full boxes need an explicit visible-vs-full extent policy.',
  annotation='Protect both clear chairs and possible left-edge seating. Review bedside furniture taxonomy and hospital-bed scope. Wall fixtures not automatically standalone Lamps.',
  primary_medical_subject=False,people_visible=False),
 'mqxh737p':dict(scene='Historical two-patient hospital room, apparently unoccupied',
  definite=['Chair','Bed'],possible=['Table','Cabinet'],chairs='Wooden side chair and upholstered armchair are clearly visible',
  observations='Two hospital beds, two clear chairs, bedside furniture/tray, telephone, plants, sink and a folding privacy screen. Foreground bed is strongly cropped.',
  distractors='Folding screen/frame, bed rails, telephone, plants, basin and bedside frames',
  quality='Furniture shapes reasonably distinct, but foreground clipping, occlusion, grain and vintage layout limit transfer to modern rooms.',
  annotation='Never label the folding privacy screen or bed-rail frames Chair solely by shape. Protect actual chairs; annotate cropped beds and uncertain bedside furniture explicitly.',
  primary_medical_subject=False,people_visible=False)}
def run():
    closed=read(OUT/'acquisition-closed.json')
    if len(closed['records'])!=7 or closed['stop']:raise ValueError('Review only the successfully acquired exact seven')
    result=[];source={r['source_image_id']:r for r in rows()}
    for r in closed['records']:
        ident=r['image_id'];note=OBS[ident];result.append(dict(image_id=ident,source_native_image=source[ident]['original_image_id'],
          image_sha256=r['sha256'],source_url=r['source_url'],source_work=source[ident]['canonical_source_url'],credit=source[ident]['credit'],
          licence_URL='https://creativecommons.org/licenses/by/4.0/',rights_status='source_supported_conditional_CC_BY4',
          suitability='B_limited_use_historical_or_atypical',modern_room_qualified=False,review_type='assistant_visual_inspection_of_acquired_photograph',
          human_reviewed=False,source_native_detection_boxes=0,source_native_instance_masks=0,
          definite_visual_category_observations=note['definite'],possible_or_taxonomy_uncertain_categories=note['possible'],
          **note,all_other_classes_status='unknown_or_not_observed; not verified absent',visual_observations_are_not_ground_truth=True,
          hard_negative_status='potential_distractors_only; no trusted negatives or actual model-query coverage established',
          annotation_active=False,verified_negative_regions=0,final_partition=None,source_group_id=source[ident]['source_group_id']))
    summary=dict(suitability_counts=dict(A_modern=0,B_historical_or_atypical=7,C_unsuitable=0),
       assistant_visual_category_image_counts={c:sum(c in r['definite_visual_category_observations'] for r in result) for c in ('Sofa','Chair','Table','Bed','Desk','Cabinet','Lamp')},
       counts_are_not_source_labels_or_complete_instance_counts=True,
       possible_or_taxonomy_uncertain_image_counts={c:sum(c in r['possible_or_taxonomy_uncertain_categories'] for r in result) for c in ('Sofa','Chair','Table','Bed','Desk','Cabinet','Lamp')},
       actual_room_scenes=7,primarily_medical_subject_photos=0,people_visible_photos=2,
       credible_modern_room_training_improvement=False,activated_annotations=0,verified_hard_negatives=0,
       tentative_related_capture_cluster=['bpxzpgg7','x6v3kwb7','y4n8bmat'],related_cluster_not_asserted_pixel_duplicate=True,
       independent_source_groups_verified=0,conservative_source_group_buckets=1,
       recommendation='Close further Wellcome acquisition for modern-room expansion. Retain these immutable7 as limited-use inspection assets; prioritize explicitly consented modern rooms. No training justified.')
    emit(OUT/'visual-review.json',dict(records=result,summary=summary));emit(REPORT/'visual-review.json',dict(records=result,summary=summary))
    cards=[]
    for r in result:
        esc=lambda v:html.escape(str(v),quote=True)
        props=[('Scene',r['scene']),('Suitability','B — limited-use historical/atypical'),('Visual furniture observations',', '.join(r['definite']) or 'No definite target-category observation'),
          ('Taxonomy/visibility uncertainty',', '.join(r['possible']) or 'No additional candidate labels assigned'),('Chair protection',r['chairs']),('Context',r['observations']),('Potential distractors, not verified negatives',r['distractors']),
          ('Quality',r['quality']),('Annotation requirements',r['annotation']),('Evidence limit','Assistant observations only; no human-reviewed boxes, negative labels, model confidence or category-absence certification')]
        cards.append('<article><h2>'+esc(r['image_id'])+' · '+esc(r['source_native_image'])+'</h2><a href="images/'+esc(r['image_id'])+'.jpg"><img loading="lazy" src="images/'+esc(r['image_id'])+'.jpg" alt="'+esc(r['scene'])+'"></a><dl>'+''.join('<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>' for k,v in props)+'</dl><p>Credit: '+esc(', '.join(r['credit']))+' · <a href="'+esc(r['source_work'])+'" rel="noreferrer">source work</a> · <a href="https://creativecommons.org/licenses/by/4.0/" rel="noreferrer">CC BY4.0</a>. Resized derivative; actual photographer remains unknown.</p><p>SHA256: <code>'+esc(r['image_sha256'])+'</code></p></article>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Furniture AI — seven-photo suitability review</title><style>body{font:17px system-ui;max-width:1250px;margin:auto;padding:1rem;line-height:1.55;background:#f3f5f8;color:#17212b}article{background:white;border:1px solid #ccd4df;margin:2rem 0;padding:1.2rem;border-radius:8px}img{width:100%;height:auto}dt{font-weight:bold;margin-top:.7rem}dd{margin-left:0}code{word-break:break-all}aside{background:#fff2cc;padding:1rem}</style><h1>Seven-photo Wellcome suitability pilot</h1><aside>Read-only assistant visual assessment. 0 modern / 7 limited-use historical / 0 unsuitable. No trained-model inference, active annotations or certified negative regions. Click a photograph for its full acquired1600px derivative. Rights and usefulness are separate.</aside><p>All7are grouped conservatively under unknown creator. Laboratory views may be related. No independent evaluation membership. <a href="creator-intake.html">Blank future creator-consent/intake workflow</a> · <a href="visual-review.json">Review metadata</a></p>'+''.join(cards)+'</html>'
    atomic(OUT/'gallery.html',page.encode());print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
