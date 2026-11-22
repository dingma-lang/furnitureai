"""Record actual assistant contact-sheet inspection;never fabricate human review."""
from common14 import *
from g5_common import overlap


def run():
    candidates=read(RAW14/'new-hard-negative-candidates.json')['candidates'];old=read(SOURCE/'source-negative-evidence.json')['records'];index=read(RAW14/'visual-index.json')['records'];records=[]
    notes={
        '09629ce8':('source_corroborated','Query shows visible television screen/face depiction and frame;no discernible seating in this local source polygon. Same rectangle also has a COCO TV lineage.'),
        '203132b7':('source_corroborated','Local crop shows dark handbag fabric/hardware;native visible mask supports localized non-Chair context. Same rectangle has another source lineage.'),
        '23984bbd':('source_corroborated','Query crop shows foreground person hair and clothing;no discernible seating within the certified local mask.'),
        '445a715c':('source_corroborated','Query crop shows foreground person chin/neck,hair and clothing. Source person polygon supports this local context;nearby seating is outside this certified rectangle.'),
        '862f34f1':('source_corroborated','Same visible TV-screen rectangle as09629ce8,not an independent object or additional physical photograph.'),
        'b9729523':('source_corroborated','Local crop shows hair/neck/clothing of the foreground person. It overlaps other proposed person rectangles and existing evidence.'),
        'bf2fd19f':('uncertain','Dark reflective television region contains room reflections;local seating absence is not sufficiently clear at this inspection resolution. Keep unresolved.'),
        'c85ca213':('source_corroborated','Near-identical person hair/neck rectangle to b9729523 from another model state;not independent evidence.'),
        'ff659c9d':('source_corroborated','Same handbag rectangle as203132b7 with another native source lineage;do not count it as a new independent object.')}
    for c in candidates:
        key=c['candidate_key'];status,note=notes[key[:8]];plate=ROOT/next(x['plate'] for x in index if x['item_id']==key);overlaps=[e['candidate_id'] for e in old if e['image_id']==c['image_id'] and overlap(c['box'],e['box'])]
        records.append(dict(candidate_key=key,image_id=c['image_id'],source_sha256=c['source_sha256'],candidate_sha256=digest(c),assistant_visual_inspected=True,human_reviewed=False,human_confirmed=False,visual_outcome=status,observation=note,plate=str(plate.relative_to(ROOT)),plate_sha256=sha(plate),overlapping_existing_regions=overlaps,
            current_policy_training_eligible=False,reason='Overlaps existing frozen source rectangle;new geometry requires explicit versioned replacement/union policy and provenance validation' if overlaps else 'No automatic target approval in this audit',training_activated=False,Chair_taxonomy_includes_seating_and_bar_stools=True))
    context_notes={
        'T01':'Native person bbox overlaps a visible upholstered booth/chair-like seating structure. Seating identity/coverage is uncertain;this is not a reliable non-Chair example.',
        'T02':'Query covers a vertical furniture/seating structure beside the foreground person. Person-bbox context does not prove Chair absence.',
        'T03':'Small mixed crop at a known seating/person/plant boundary;potential chair fragment. Do not mine as background.',
        'T04':'Mostly Christmas-tree branches and ornaments in the inspected crop. This is a possible confusion,not certified negative evidence for the whole query.',
        'T05':'Laptop/display and nearby support structure;native bbox overlap alone is insufficient for exact query-negative geometry.',
        'T06':'TV/display region with border/context. Plausible confusion,but the whole model box is not certified by this bbox context.',
        'T07':'Visible partial seating by the window;not trustworthy curtain-negative evidence.',
        'T08':'Visible wooden Chair back/slats under the curtain bounding box. Must not be labelled background.',
        'T09':'Visible open drawer/tray and small objects. Furniture-part semantics and source geometry require care;no negative target created.',
        'T10':'Visible pink seating chair/stool. Project Chair includes seating/bar stools,so this ignored query must not become a negative merely because it is unmatched.',
        'T11':'Furniture rail/back structure at the bed/table boundary;Chair-versus-bed-part identity remains uncertain. Keep ignored.'}
    context=[]
    for r in read(RAW14/'training-inspection-sample.json')['records']:
        plate=ROOT/next(x['plate'] for x in index if x['item_id']==r['sample_id']);context.append(dict(sample_id=r['sample_id'],image_id=r['image_id'],state=r['state'],query_index=r['query_index'],assistant_visual_inspected=True,human_reviewed=False,observation=context_notes[r['sample_id']],Chair_negative_approved=False,plate=str(plate.relative_to(ROOT)),plate_sha256=sha(plate)))
    save(RAW14/'assistant-visual-observations.json',dict(candidate_records=records,context_records=context,actual_contact_sheets_inspected=4,human_reviews_created=0,source_annotations_modified=False))
    distinct={(c['image_id'],tuple(c['box_pixels'])) for c in candidates};corroborated={ (c['image_id'],tuple(c['box_pixels'])) for c in candidates if notes[c['candidate_key'][:8]][0]=='source_corroborated'}
    save(REPORT14/'visual-findings.json',dict(source_geometry_lineage_records=len(candidates),distinct_proposed_rectangles=len(distinct),source_photo_count=len({c['image_id'] for c in candidates}),known_source_groups=len({c['identity_group'] for c in candidates}),visually_corroborated_lineage_records=sum(r['visual_outcome']=='source_corroborated' for r in records),visually_corroborated_distinct_rectangles=len(corroborated),uncertain_lineage_records=sum(r['visual_outcome']=='uncertain' for r in records),all_candidates_overlap_existing_negative_regions=all(r['overlapping_existing_regions'] for r in records),new_additive_training_eligible_hard_negatives=0,all_new_geometry_inactive=True,context_examples_inspected=len(context),human_reviews_created=0,
        finding='Some highest-score person/curtain bbox overlaps actually depict seating or mixed furniture;unmatched and source-bbox overlap must never imply background',context_observations=context))
    print('Assistant visual audit:9 lineage records/7rectangles;0new additive eligible negatives;no human flags changed',flush=True)


if __name__=='__main__':run()
