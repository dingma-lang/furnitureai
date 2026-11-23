"""Read-only E6 dataset verification. No candidate promotion or model execution."""
from collections import Counter
from commonE6 import *

def check():
    c,train,dev=context()
    index=read(DATA/'query-index-v3.json')
    if index['evaluation_cache_bodies_mined'] or index['invalid_layout_records']:
        raise ValueError('Unsafe query cache membership/layout')
    if {r['snapshot_id'] for r in index['records']}!={r['snapshot_id'] for r in read(DATA/'query-index-v2.json')['records']}:
        raise ValueError('An unmined numerical snapshot was added')
    dev_ids={r['image_id'] for r in dev}
    remaining=index['unindexed_non_evaluation_NPZ']
    if len(remaining)!=19 or {Path(r['path']).stem for r in remaining}!=dev_ids:
        raise ValueError('Unindexed raw files need additional source investigation')
    for r in index['records']:
        require_train(r['image_id'],train)
        for a in r['aliases']:
            if a['image_id']!=r['image_id'] or a['source_sha256']!=train[r['image_id']]['sha256'] or sha(ROOT/a['path'])!=a['sha256']:
                raise ValueError('Query identity/hash mismatch')
    dataset=read(DATA/'dataset-index-v1.json')
    for k,hk in [('manifest','manifest_sha256'),('lineage','lineage_sha256'),('inactive_pool','inactive_pool_sha256'),('native_furniture_candidates','native_furniture_sha256')]:
        if sha(ROOT/dataset[k])!=dataset[hk]:raise ValueError('Dataset component hash mismatch')
    manifest=read(ROOT/dataset['manifest'])
    if sha(ROOT/manifest['base_training_context'])!=manifest['base_training_context_sha256'] or manifest['training_authorized'] is not False:
        raise ValueError('Base manifest or training gate changed')
    for key in ('candidates','query_observations'):
        if sha(ROOT/manifest[key])!=manifest[key+'_sha256']:raise ValueError('Candidate evidence changed')
    candidates=read(ROOT/manifest['candidates'])['records'];sets=manifest['sets']
    identifiers=[r['candidate_id'] for r in candidates]
    assigned=[i for v in sets.values() for i in v]
    if len(set(identifiers))!=len(identifiers) or len(set(assigned))!=len(assigned) or set(assigned)!=set(identifiers):
        raise ValueError('Candidate evidence tiers overlap or omit a record')
    if sets['A_trusted_new_training_eligible_hard_negatives']:
        raise ValueError('New trusted evidence requires a new versioned audit;E6 established none')
    for r in candidates:
        row=require_train(r['image_id'],train)
        if r['source_sha256']!=row['sha256'] or r['source_group']!=row['identity_group']:
            raise ValueError('Source identity changed')
        if r['training_eligible'] or r['training_activated'] or r['human_reviewed']:
            raise ValueError('Unverified evidence cannot be activated or marked human reviewed')
        if r['evidence_tier'] not in ('B','C') or r['candidate_id'] not in sets['B_automatic_unverified' if r['evidence_tier']=='B' else 'C_uncertain_or_rejected']:
            raise ValueError('Incorrect tier assignment')
    for r in read(DATA/'source-lineage-v1.json')['photos']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Source image changed')
        for k,hk in [('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')]:
            if sha(ROOT/r['rights'][k])!=r['rights'][hk]:raise ValueError('Rights evidence changed')
    return dict(status='PASS',optimization_images=len(train),development_images_excluded=len(dev),
                raw_cache_files=index['raw_source_bound_files'],unique_snapshots=len(index['records']),
                tiers=dict(Counter(r['evidence_tier'] for r in candidates)),new_training_labels=0,
                remaining_internal_validation_files_excluded=19,GPU_calls=0)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
