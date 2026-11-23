"""Compose immutable E8 evidence components and a small decision CSV."""
import csv
from common8q import *
def run():
    components={k:DATA/v for k,v in dict(hard_negative_manifest='hard-negative-dataset-v1.json',source_audit='candidate-source-audit.json',decisions='automatic-decisions-v1.json',
        actual_query_coverage='actual-query-coverage.json',deduplication='historical-evidence-deduplication-v2.json',proposed_rights_candidates='licensed-training-rights-candidates-v1.json').items()}
    emit(DATA/'dataset-index-v1.json',dict(schema='phase10e8-composed-qualification-v1',components={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in components.items()},
         classification='0trusted/5unverified/16protected-or-unsuitable',new_training_activation=False,
         authoritative_deduplication_version=2,older_deduplication_reference_counts_not_object_identity=True,independent_evaluation_excluded=True))
    stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['sample_id','image_id','query_index','E5_Chair_score','original_Chair_score','best_visible_native_label','best_single_polygon_fraction','decision','object_assessment','human_reviewed','training_activated'])
    for r in read(DATA/'automatic-decisions-v1.json')['records']:
        a=r['native_mask_hits'][0]
        writer.writerow([r['sample_id'],r['image_id'],r['query_index'],r['student_Chair_score'],r['original_same_query_Chair_score'],a['native_label'],a['maximum_single_polygon_query_fraction'],r['decision'],r['object_assessment'],False,False])
    atomic(REPORT/'candidate-decisions.csv',stream.getvalue().encode())
    print('Composed E8dataset index and21decision CSV;no labels activated')
if __name__=='__main__':run()
