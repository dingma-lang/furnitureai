"""Offline expanded-dataset audit and inactive, versioned training manifest."""
import collections
from common import *
from policy import CORE, Union, positive_allowed, normalized_box, explicit_negatives, summarize, inner_split
from rights import proof

SOURCES = {
 'legacy_openimages': ('data/open-images/image-pilots/phase5-v1', 'data/open-images/experiments/phase5b-v1/dataset-lock.json','rows'),
 'phase6a_openimages': ('data/open-images/image-pilots/phase6a-v1', 'data/open-images/image-pilots/phase6a-v1/screening-v3.json','images'),
 'phase6d_coco': ('data/coco/image-pilots/phase6d-v1', 'data/coco/image-pilots/phase6d-v1/screening-v2.json','images'),
 'phase7_LVIS': ('data/lvis/image-pilots/phase7-v1', 'data/lvis/image-pilots/phase7-v1/screening-v2.json','images'),
 'phase8c_LVIS': ('data/lvis/image-pilots/phase8b-v1', 'data/lvis/image-pilots/phase8b-v1/screening-v2.json','images')}

def run():
    d = read(READINESS); source_rows = d['source_rows']
    train = [r for r in source_rows if r['split']=='train']
    if (len(train),sum(len(r['resolved_targets']) for r in train)) != (120,373):
        raise ValueError('Prospective training inventory changed')
    cache = {k:{r['image_id']:r for r in read(ROOT/path)[key]} for k,(_,path,key) in SOURCES.items()}
    refs = read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']
    uf = Union(); protected = set(); seen = collections.defaultdict(list)
    for r in source_rows:
        for node in [r['known_source_component'],r.get('original_source_group'),r.get('decoded_pixel_sha256') and 'pixel-sha:'+r['decoded_pixel_sha256'],'image-sha:'+r['sha256']]:
            if node: uf.join(r['source_photo_id'],node)
        if r['split']!='train': protected.add(r['source_photo_id'])
    # All proposed historical development/frozen IDs remain protected, including quarantine slots.
    for r in refs:
        uf.join(r['source_identity'],r['identity_group'])
        if r['proposed_split']!='train': protected.add(r['source_identity'])
    rights = []; rows = []; decisions = []; integrity = []; issues = []
    for s in train:
        base = ROOT/SOURCES[s['source']][0]; original = cache[s['source']][s['image_id']]
        p = (base/original['file']).resolve()
        if not p.is_relative_to(base/'photos') or sha(p)!=s['sha256'] or original['sha256']!=s['sha256']:
            raise ValueError('Source image hash/path mismatch')
        width,height = original['width'],original['height']
        grant = proof(s,original,base)
        uf.join(s['source_photo_id'],grant['creator_group'])
        rights.append(dict(image_id=s['image_id'],source_photo_id=s['source_photo_id'],**grant))
        native_rows = original.get('original_annotations',original.get('annotations',[]))
        for t in s['resolved_targets']:
            if t['original_annotation'] not in native_rows:
                raise ValueError('Derived target not identical to preserved native annotation')
        allowed = [t for t in s['resolved_targets'] if positive_allowed(t)]
        positives = [t['original_annotation'] for t in allowed]
        annotations = []
        for a in native_rows:
            dataset = 'OpenImages' if 'openimages' in s['source'] else 'COCO' if s['source']=='phase6d_coco' else 'LVIS'
            native = a.get('source_mid',a.get('category_id'))
            label = CORE[dataset].get(native,'unknown')
            box = normalized_box(dataset,a,width,height)
            active = a in positives
            annotations.append(dict(label=label,box=box,ignore=not active,
                                    native_dataset=dataset,native_id=native,
                                    original_annotation=a,has_mask=bool(a.get('segmentation'))))
        for t in s['resolved_targets']:
            active = positive_allowed(t)
            decisions.append(dict(image_id=s['image_id'],annotation_id=t['stable_audit_annotation_id'],category=t['comparison_group'],
              prepared_positive=active,activated_for_training=False,
              reason='source-native provisional positive' if active else 'Unresolved native-to-project unit/semantics or ignored target; inactive',
              border_or_truncation=t['border_or_truncation'],native_scope=t['native_scope']))
        neg = explicit_negatives(s)
        for t in s['resolved_targets']:
            if neg.get(t['comparison_group']): raise ValueError('Positive/explicit-negative source contradiction')
        row = dict(image_id=s['image_id'],historical_partition='train',source=s['source'],source_photo_id=s['source_photo_id'],
                   path=str(p.relative_to(ROOT)),sha256=s['sha256'],width=width,height=height,
                   annotations=annotations,explicit_native_negative=neg,source_native_scope=s['source_native_scope'],
                   photograph_rights_evidence_sha256=grant['evidence_sha256'],human_verified=False,
                   project_semantics_unverified=True,eligible_positive_count=sum(not a['ignore'] for a in annotations))
        rows.append(row); seen[s['sha256']].append(s['image_id'])
        integrity.append(dict(image_id=s['image_id'],sha256=s['sha256'],width=width,height=height,
                              photo_sha_verified=True,native_targets_identical=True,photo_decoded=False))
    protected = {uf.find(p) for p in protected}
    for r in rows:
        r['identity_group'] = uf.find(r['source_photo_id'])
        if r['identity_group'] in protected: issues.append(dict(image_id=r['image_id'],reason='Historical development/frozen source-group overlap'))
    duplicates = [ids for ids in seen.values() if len(ids)>1]
    if duplicates: issues.append(dict(reason='Duplicate acquired photograph identities',groups=duplicates))
    rejected_ids = {x['image_id'] for x in issues if 'image_id' in x}
    eligible = [r for r in rows if r['eligible_positive_count'] and r['image_id'] not in rejected_ids]
    if duplicates: raise ValueError('Duplicate photos require conservative reconciliation before preparation')
    prepared = inner_split(eligible)
    partitions = {}
    for part in ('optimization','inner_development'):
        subset = [r for r in prepared if r['internal_partition']==part]
        partitions[part] = dict(images=len(subset),groups=len({r['identity_group'] for r in subset}),
                                boxes=sum(r['eligible_positive_count'] for r in subset),per_category=summarize(subset))
    if {r['identity_group'] for r in prepared if r['internal_partition']=='optimization'} & {r['identity_group'] for r in prepared if r['internal_partition']=='inner_development'}:
        raise ValueError('Internal group leakage')
    save(OUT/'rights-replay-ledger.json',dict(records=rights,network_requests=0,photo_specific_replayed=len(rights),
        ownership_or_release_clearance=False,live_provider_reverification=False))
    save(OUT/'annotation-dispositions.json',dict(records=decisions,unresolved_annotations_never_activated=True,
        previous33_LVIS_overlays_inactive=True,source_annotations_modified=False))
    save(OUT/'image-integrity.json',dict(records=integrity,model_inference=0,photo_decoding=0))
    save(OUT/'prepared-manifest.json',dict(schema='phase10a-inactive-head-preparation-v1',training_authorized=False,
        source_readiness_sha256=sha(READINESS),classes=list(CLASSES),supported_positive_categories=['sofa','chair','bed'],
        rows=prepared,original_train_images=120,original_train_boxes=373,original_train_groups=114,
        eligible_images=len(eligible),eligible_groups=len({r['identity_group'] for r in eligible}),
        eligible_boxes=sum(r['eligible_positive_count'] for r in eligible),per_category=summarize(eligible),
        internal_partitions=partitions,seed='phase10a-inner-20261118-v1',
        project_background_never_inferred=True,source_group_issues=issues,
        excluded_no_supported_positive=[r['image_id'] for r in rows if not r['eligible_positive_count']],
        broader_seven_class_readiness=False,source_or_creator_group_not_physical_identity_proof=True))
    print('AUDIT', len(eligible), 'images',sum(r['eligible_positive_count'] for r in eligible),'boxes; inner partitions',
          {p:(s['images'],s['boxes']) for p,s in partitions.items()},flush=True)

if __name__=='__main__': run()
