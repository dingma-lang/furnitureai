"""Append diagnostic assistant observations of actually viewed C9 contact plates.

This is not a human-review submission or a training-supervision exporter.
"""
from collections import Counter
from datetime import datetime,timezone
from common9 import *

# Recorded after viewing all nine plates with the local image-view tool.
DEVELOPMENT={
 'S001':('C','Whole red-topped seating stool is visible on the left, outside retained Chair boxes; native context labels it only Furniture. Project Chair includes seating stools, so this is a likely annotation/taxonomy gap, not a trustworthy background negative.'),
 'S002':('E','Box covers an annotated dining-chair back and part of its seat, with legs excluded; source geometry also flags a Chair fragment.'),
 'S003':('F','Tiny dark vertical feature behind the table is not sufficiently identifiable; could be architecture or an object edge. No definitive non-Chair label.'),
 'S004':('E','Mixed rectangle includes a seated person, table and part of the annotated Chair. It is not an isolated background object.'),
 'S005':('E','Partial side/seat/leg region of the annotated dining Chair; localization rather than a clearly unrelated object.'),
 'S006':('E','Lower frame/legs of the apparently unannotated red-topped seating stool. Also implicated in the Chair/stool completeness gap.'),
 'S007':('E','Thin horizontal crop of that stool seat; a fragment of seating, not evidence of absent Chairs.'),
 'S008':('E','Small metal-leg/base fragment below an annotated Chair.'),
 'S009':('F','Distant person/seating cluster is too small and mixed to distinguish a missing stool from background confusion.'),
 'S010':('C','Distant metal seating-stool/chair frame appears below a person, outside active Chair boxes. Apparent unannotated seating requires confirmation, not negative labeling.'),
 'S011':('F','Distant person and possible stool frame overlap. Cannot isolate a background object confidently.'),
 'S012':('F','Small distant dark patch at a table/plant area; insufficient detail for a definitive object identity.'),
 'S013':('F','Tiny green/brown feature behind a plant may be seating or room clutter. Remains unresolved.'),
 'S014':('E','Thin back/side edge of a known dining Chair is boxed rather than the complete Chair.'),
 'S015':('A','Small window/reflection/architectural patch is boxed; no complete seating object is apparent in this crop. Assistant visual judgment only.'),
 'S016':('E','Tiny connection/frame detail of the annotated dining Chair.'),
 'S017':('E','Partial office-chair back/seat/base, with a native Chair box available. IoU miss is a localization issue.'),
 'S018':('A','Curved wooden musical-instrument body beside the bed is boxed as Chair. Assistant visual judgment; source overlap does not independently certify the instrument.'),
 'S019':('D','Another box around the same annotated office Chair; preserved metric matching independently marks this as a duplicate.'),
 'S020':('B','Wooden bed-frame/post structure is boxed as Chair. Whole-object Bed/Cabinet mapping is not changed by this diagnosis.'),
 'S021':('F','Tight wooden-post/instrument-edge crop near the bed is mixed and not a confidently identified whole object.'),
 'S022':('E','Annotated office-chair backrest and support, with most of the seat/base absent.'),
 'S023':('A','Transparent container/trash-can region beside a bedpost is predicted Chair; native trash_can box has approximately0.98 IoU. Source-object support plus actual visual judgment, not human confirmation.'),
 'S024':('B','Desk drawer/storage unit is boxed as Chair. Native drawer context supports this; no automatic Cabinet-versus-Desk relabeling.'),
 'S025':('F','Small white feature just above the tabletop may be a chair-back fragment or another object; insufficient context to decide.'),
 'S026':('F','Thin white strip along the tabletop/possible chair-back boundary; do not certify a non-Chair negative.'),
 'S027':('E','One large rectangle spans the conference table and several genuine Chairs. Multi-object/poor-localization mechanism, not isolated background.'),
 'S028':('E','Rectangle spans table support and two Chair regions; mixed furniture and poor instance localization.'),
 'S029':('A','Small tabletop stationery/container object is predicted Chair. Assistant visual judgment; no new source-negative certificate.'),
 'S030':('E','Seat/base portion of an annotated conference Chair; source IoU is just below0.5.'),
 'S031':('A','Small tabletop container/object detail is boxed as Chair; assistant judgment only, at limited resolution.'),
 'S032':('E','Thin Chair leg/base fragment at the left edge of the scene.'),
 'S033':('C','Wicker Chair is clearly visible behind the bed; additional native chair annotation overlaps at approximately0.93 IoU, but it is absent from active project Chair GT. Already unscored by the primary protocol.'),
 'S034':('C','Complete folding Chair is visible by the window; native chair overlap approximately0.85, absent from active project Chair GT. Already unscored; not a confirmed mistake.'),
 'S035':('F','Foreground upholstered shape could be a Chair back or a loose cushion. Native Furniture is too broad to resolve it.'),
 'S036':('A','Small brown plant-pot/container region on a table is boxed as Chair. Unscored example and assistant judgment only.'),
 'S037':('B','Wooden bed-frame/rail structure is boxed as Chair; native Bed overlap supports the other-furniture explanation. This remains unscored in the primary Chair protocol.'),
 'S038':('E','Partial red seating-stool back and legs, with adjacent stool boxes present; a localization fragment in an incomplete scope.'),
 'S039':('E','Distant group of metal dining-chair frames is captured in a small mixed box; neither a complete single instance nor safe background.'),
 'S040':('F','Narrow dark foreground frame/rail could belong to seating or another furnishing. Insufficient visual certainty.')}
TRAINING={
 'T001':('F','Folding luggage-rack-like structure, possibly seating-related; no certified native mask, so not a training negative.'),
 'T002':('A','Wicker basket/container draped with fabric appears non-Chair, but no entire-query source-polygon certificate exists.'),
 'T003':('F','Dark foreground person/clothing/possible furniture area; too ambiguous for negative supervision.'),
 'T004':('A','Tall drinking/champagne glass is visible. Full query is not certified within a native mask, so source-negative eligibility remains false.'),
 'T005':('F','Dark furniture/background boundary at the far table is not confidently identifiable.'),
 'T006':('F','Mixed television/cabinet/luggage-rack area; category and spatial extent are unresolved.'),
 'T007':('F','Thin dark vertical frame adjacent to a person; could be furniture or clothing. Remains ignored.'),
 'T008':('A','Small forearm/hand region near the tabletop appears non-Chair; no complete source-mask certificate exists.'),
 'T009':('C','Visible wooden Chair seat near the foreground person appears to be real seating outside protected boxes. This demonstrates why unmatched training queries cannot automatically become negatives.'),
 'T010':('F','Narrow patterned hanging textile/object beside a table; insufficient source/spatial evidence.'),
 'T011':('F','Small brown furniture/frame patch could be part of a Chair. Do not treat as background.'),
 'T012':('A','Plant/pot and surrounding floor are visible. Mixed exterior pixels prevent full source-mask certification; not eligible for training.')}

def run():
    samples=read(OUT/'review-sample.json')['records']+read(OUT/'training-review-sample.json')['records'];pages=read(OUT/'visual-plates.json')['pages'];decisions=dict(DEVELOPMENT,**TRAINING);records=[]
    viewed=read(HERE/'viewed-evidence.json')
    if viewed['plates']!=pages or viewed['sample_digests']!={p['sample_id']:digest(p) for p in samples}:raise ValueError('These are not the exact previously viewed plates/queries; new visual inspection is required')
    if set(decisions)!={p['sample_id'] for p in samples}:raise ValueError('Every supplied visual observation must match a viewed sample')
    for p in samples:
        category,note=decisions[p['sample_id']];page=next(q for q in pages if p['sample_id'] in q['sample_ids'])
        if sha(ROOT/page['path'])!=page['sha256']:raise ValueError('Viewed plate changed')
        record=dict(sample_id=p['sample_id'],image_id=p['image_id'],query_index=p['query_index'],prediction_id=p.get('prediction_id'),scope='development_diagnostic_only' if p['sample_id'].startswith('S') else 'optimization_training_unverified_diagnostic',
            taxonomy_primary=category,observation=note,method='assistant directly viewed source-frame plus saved-query crop using local image viewer',evidence_tier='assistant_visual_judgment',
            native_support=[q for q in p.get('native_overlap_context',[]) if q['iou']>=.5],source_plate=page['path'],source_plate_sha256=page['sha256'],
            human_reviewed=False,human_confirmed=False,training_enabled=False,model_inference_used=False,observed_utc=datetime.now(timezone.utc).isoformat())
        if p['sample_id']=='S019':record['evidence_tier']='source_metric_duplicate_and_assistant_visual'
        elif p['sample_id'] in ('S023','S024','S033','S034','S037'):record['evidence_tier']='native_object_overlap_and_assistant_visual'
        records.append(record)
    save(OUT/'assistant-observations.json',dict(records=records,human_confirmations_created=0,training_labels_created=0))
    save(REPORT/'visual-findings.json',dict(targeted_not_random=True,development_samples=40,training_samples=12,
        scored_FP_sample_taxonomy=dict(Counter(r['taxonomy_primary'] for r in records if r['sample_id'] in {p['sample_id'] for p in samples if p.get('status')=='fp'})),
        development_unknown_sample_taxonomy=dict(Counter(r['taxonomy_primary'] for r in records if r['sample_id'] in {p['sample_id'] for p in samples if p.get('status')=='unknown_unmatched'})),
        training_unverified_sample_taxonomy=dict(Counter(r['taxonomy_primary'] for r in records if r['sample_id'].startswith('T'))),
        observations_are_not_human_ground_truth=True,full_636_FP_visually_inspected=False,new_training_eligible_hard_negatives=0,
        annotation_gap='S001 red-topped seating stool is only native Furniture, not Chair, yet project Chair includes seating stools. Native Chair completeness cannot guarantee completeness of the broader project concept.',
        source_supported_missing_active_Chair_examples=['S033','S034'],source_and_visual_nonchair_example='S023',training_missing_seating_risk_example='T009'))

if __name__=='__main__':run()
