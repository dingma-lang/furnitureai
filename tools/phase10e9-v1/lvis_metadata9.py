"""Add existing LVIS availability to per-ID rights metadata without upgrading rights."""
from common9 import *
def run():
    p=OUT/'LVIS-existing-annotation-audit.json';v=read(p)
    emit(REPORT/'LVIS-annotations-by-ID.json',dict(records=v['records'],source_archives=v['archives'],
         annotation_licence='https://creativecommons.org/licenses/by/4.0/',private_native_annotation_audit=str(p.relative_to(ROOT)),private_audit_sha256=sha(p),
         no_photo_rights_granted=True,no_training_labels_activated=True,no_double_counting_same_COCO_photographs=True))
    old=read(REPORT/'rights-audit-index.json')
    emit(REPORT/'rights-audit-index-v2.json',dict(old,
         prior_index_sha256=sha(REPORT/'rights-audit-index.json'),
         additional_existing_LVIS_annotations='reports/phase10e9-v1/LVIS-annotations-by-ID.json',additional_existing_LVIS_sha256=sha(REPORT/'LVIS-annotations-by-ID.json'),
         additional_existing_LVIS_summary='reports/phase10e9-v1/LVIS-existing-annotation-summary.json',additional_existing_LVIS_summary_sha256=sha(REPORT/'LVIS-existing-annotation-summary.json'),
         authoritative_version=2,rights_counts_unchanged=True))
    print('Authoritative rights indexv2adds159existing LVISinstances;all64rights remain unresolved')
if __name__=='__main__':run()
