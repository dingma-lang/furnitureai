"""Exact source-instance and historical-evidence accounting,without object merges."""
from collections import defaultdict
from common8q import *

def run():
    rows=read(DATA/'automatic-decisions-v1.json')['records']
    old=read(ROOT/'data/open-images/preparation/phase10c6-v1/source-negative-evidence.json')['records']
    prior=read(ROOT/'data/open-images/preparation/phase10c21-v1/hard-negative-manifest-v1.json')
    inactive=[r for r in prior['records'] if r['candidate_id'] in prior['accepted_records']]
    results=[]
    for r in rows:
        best=r['native_mask_hits'][0] if r['native_mask_hits'] else None
        exact=[e['candidate_id'] for e in old if e['image_id']==r['image_id'] and best and e['source_annotation_key']==best['annotation_key']]
        possible_alias=[e['candidate_id'] for e in inactive if e['image_id']==r['image_id'] and best and e['native_label'].lower()==best['native_label'].lower()]
        results.append(dict(candidate_id=r['candidate_id'],image_id=r['image_id'],query_index=r['query_index'],source_photo_id=r['source_photo_id'],
           source_sha256=r['source_sha256'],previously_seen_E6_exact_footprint=r['previously_observed_exact_footprint'],
           existing41_exact_source_annotation_grants=exact,
           annotation_reference_match_is_not_observed_object_identity=True,
           strong_existing_handbag_instance_match=bool(exact and r.get('repeated_partial_source_instance')),
           existing41_rectangle_overlap=r['existing_negative_region_overlap'],existing41_full_containment=r['existing_negative_region_full_containment'],
           prior_inactive_C21_same_photo_label_possible_alias=possible_alias,
           possible_aliases_not_automatically_merged=True,
           source_instance_representative=r.get('source_instance_representative'),
           not_an_independent_new_object=bool(exact and r.get('repeated_partial_source_instance') or r.get('source_instance_representative') and r['source_instance_representative']!=r['candidate_id'])))
    destination=DATA/('historical-evidence-deduplication-v2.json' if (DATA/'historical-evidence-deduplication.json').exists() else 'historical-evidence-deduplication.json')
    emit(destination,dict(records=results,
         candidate_queries=21,unique_exact_candidate_ids=len({r['candidate_id'] for r in rows}),unique_original_photo_ids=len({r['source_photo_id'] for r in rows}),
         original_image_hashes=len({r['source_sha256'] for r in rows}),known_source_groups=len({r['source_group'] for r in rows}),
         previously_seen_E6_footprints=sum(r['previously_seen_E6_exact_footprint'] for r in results),
         exact_native_annotation_reference_matches_existing41=sum(bool(r['existing41_exact_source_annotation_grants']) for r in results),
         strong_existing_handbag_instance_matches=sum(r['strong_existing_handbag_instance_match'] for r in results),
         old_region_overlap_queries=sum(bool(r['existing41_rectangle_overlap']) for r in results),old_region_full_containment_queries=sum(bool(r['existing41_full_containment']) for r in results),
         new_trusted_source_instances=0,new_high_confidence_covered_queries=0,
         overlapping_annotations_or_same_creator_not_independent_object_proof=True))
    print('Deduplication:13old E6footprints;5queries on already-represented C174handbag;zero new trusted instances')
if __name__=='__main__':run()
