"""Deterministic metadata-only content prioritization;no acquisition eligibility."""
import csv
from common9 import *
def run():
    original=read(OUT/'local-candidate-audit.json')['records'];ranked=[]
    confusers={'person','potted plant','vase','handbag','laptop','tv','refrigerator','oven','microwave','sink'}
    outdoor={'airplane','zebra','skateboard','car','truck','bicycle'}
    for r in original:
        native=r['native_category_counts'];contexts=sorted(set(native)&confusers);outside=sorted(set(native)&outdoor)
        score=3*('person' in native)+2*len(contexts)+2*('bed' in native)+2*('couch' in native)-3*len(outside)
        ranked.append(dict(image_id=r['image_id'],source_photo_id=r['source_photo_id'],score=score,confuser_metadata=contexts,
             potential_outdoor_or_mixed_context=outside,furniture_counts=r['native_furniture_counts'],
             native_annotation_IDs=r['native_annotation_ids'],rights_status=STATUS_B,
             training_eligible=False,photo_acquisition_eligible=False,visual_suitability_verified=False,Chair_absence_inferred=False))
    ranked.sort(key=lambda r:(-r['score'],r['image_id']))
    for i,r in enumerate(ranked,1):r['rank']=i
    emit(OUT/'metadata-priority-ranking.json',dict(records=ranked,deterministic=True,random_seed=None,
          rule='3*person_presence+2*count(confuser classes)+2*bed_presence+2*sofa_presence-3*count(outdoor/mixed-context classes);tie by exact image ID',
          rank_is_not_rights_clearance_or_visual_evidence=True,current_acquisition_eligible_count=0,
          no_inference_or_holdout_labels_used=True))
    stream=io.StringIO();w=csv.writer(stream,lineterminator='\n');w.writerow(['rank','image_id','source_photo_id','metadata_priority_score','confuser_metadata','potential_outdoor_or_mixed_context','rights_status','acquisition_eligible'])
    for r in ranked:w.writerow([r['rank'],r['image_id'],r['source_photo_id'],r['score'],';'.join(r['confuser_metadata']),';'.join(r['potential_outdoor_or_mixed_context']),r['rights_status'],False])
    atomic(REPORT/'metadata-priority.csv',stream.getvalue().encode())
    print('Ranked64metadata candidates;all remain rights-unresolved and acquisition-ineligible')
if __name__=='__main__':run()
