"""Training-derived native metric scopes only; no predictions or scoring are run."""
from common import CLASSES
from policy import CORE, SUPPORTED

def metric_row(row):
    if row['historical_partition']!='train' or row['internal_partition'] not in ('optimization','inner_development'):
        raise PermissionError('Only the prepared training pool can enter internal evaluation')
    coverage={c:False for c in CLASSES};fp={c:False for c in CLASSES}
    for c in SUPPORTED:
        positive=[a for a in row['annotations'] if a['label']==c and not a['ignore']]
        negative=row['explicit_native_negative'].get(c,False)
        if positive and negative:raise ValueError('Contradictory native scope')
        coverage[c]=bool(positive or negative);fp[c]=negative
        if not positive:continue
        dataset=positive[0]['native_dataset'];native=positive[0]['native_id']
        if CORE[dataset].get(native)!=c:raise ValueError('Native metric class mismatch')
        all_retained=all(not a['ignore'] for a in row['annotations'] if a['native_dataset']==dataset and a['native_id']==native)
        if dataset=='LVIS':fp[c]=all_retained and native not in row['source_native_scope'].get('not_exhaustive_category_ids',[])
        elif dataset=='OpenImages':fp[c]=all_retained and row['source_native_scope'].get('native_image_level_labels',{}).get(native)==1
        # COCO's project-wide completeness was never independently established.
    return dict(row,coverage=coverage,fp_coverage=fp,
                metric_scope='Source-native partial annotation agreement; not exhaustive or human-verified accuracy')
