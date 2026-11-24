"""Read-only project metadata/photograph hashing; no network, decoding or models."""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/phase10e15-v1'


def sha(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def emit(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    paths={key:ROOT/path for key,path in dict(
        E14='reports/phase10e14-v1/summary-final.json',
        eligibility='data/open-images/preparation/phase10b-v1/eligibility-manifest.json',
        lineage='data/open-images/preparation/phase10e6-v1/source-lineage-v1.json',
        inactive='data/open-images/preparation/phase10e6-v1/inactive-local-candidates-v1.json',
        wellcome='reports/phase10e11-v1/image-and-rights-manifest.json',
        wellcome_summary='reports/phase10e11-v1/summary.json',
        room_integration='reports/phase4-room-comparison-v1/input-lock.json',
        exclusion='data/open-images/preparation/phase10e6-v1/independent-exclusion-registry.json',
        original_context='data/open-images/preparation/phase10e1-v1/training-context.json',
        E14_seal='reports/phase10e14-v1/artifact-seal.json',
        E11_seal='reports/phase10e11-v1/artifact-seal.json').items()}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()}
    emit('reliability-baseline.json',dict(parent='f9a06a6103428bee4915acbde03862c498b66c92',
        baseline_branch='experiment/phase10e14-synthetic-pipeline',initial_worktree_clean=True,
        CPU_passed=425,documented_skips=33,test_errors=0,new_synthetic_passed=34,
        historical_data_model_report_files_preserved=28342,tests_rerun=False,
        source=str(paths['E14'].relative_to(ROOT)),source_sha256=sha(paths['E14'])))
    eligible=docs['eligibility'];lineage=docs['lineage']['photos']
    # Use actual metadata rows, rather than summing overlapping COCO/LVIS labels.
    rows={r['image_id']:r for r in lineage}
    for r in eligible['rows']:rows.setdefault(r['image_id'],r)
    checked=[]
    for r in rows.values():
        p=ROOT/r['path']
        actual=sha(p)
        if actual != r['sha256']:raise ValueError('Local source photo changed: '+r['image_id'])
        checked.append(dict(image_id=r['image_id'],sha256=actual,sha1=sha(p,'sha1')))
    if len(rows)!=120 or len({r['sha256'] for r in checked})!=120:
        raise ValueError('Historical source pool requires reconciliation')
    # E11 and Phase4 records are reported separately, not activated or relabelled.
    wellcome=docs['wellcome'];records=wellcome.get('records',wellcome.get('images',[]))
    if not records:raise ValueError('Unexpected Wellcome manifest schema')
    room=docs['room_integration']['rooms']
    for r in room:
        if sha(r['file'])!=r['sha256']:raise ValueError('Historical integration photo changed')
    inventory=dict(source_detection_pool_images=len(rows),source_detection_unique_hashes=120,
        active_training=eligible['internal_partitions']['optimization'],
        exposed_internal_development=eligible['internal_partitions']['inner_development'],
        inactive_unassigned_images=len(docs['inactive']['records']),
        inactive_known_groups=len({r['source_group'] for r in docs['inactive']['records']}),
        per_category_counts_are_overlapping=True,wellcome=dict(acquired=7,modern_qualified=0,native_boxes=0,native_masks=0,
            source_rights_supported=7,category_visual_observations=docs['wellcome_summary']['visual_summary']['assistant_visual_category_image_counts']),
        historical_integration=dict(photos=len(room),exposed=True,exhaustive_labels=False,
            new_ML_rights_status='Unsplash purpose-specific ML permission not established; do not count as cleared new training data'),
        modern_fully_annotated_independent_acceptance_images_verified_for_this_pilot=0,
        new_usable_training_images_added=0,annotations_activated=0,
        scope='Existing declared source eligibility, not proof every source photograph is a representative modern room')
    emit('existing-room-inventory.json',inventory)
    # Official Commons imageinfo metadata fetched separately; never image bodies.
    receipt=json.loads((OUT/'commons-metadata-access.json').read_text())
    candidates=[];exclude=docs['exclusion']
    for p in receipt['document']['query']['pages'].values():
        i=p['imageinfo'][0];m=i['extmetadata']
        artist=re.sub('<[^>]+>','',m['Artist']['value'])
        group='commons-user:'+artist
        direct_group_hits=[g for g in exclude['source_groups'] if g.casefold()==group.casefold()]
        c=dict(source_id='commons-page:'+str(p['pageid']),native_file_title=p['title'],
            source_page=i['descriptionurl'],source_object_url=i['url'],expected_bytes=i['size'],
            source_sha1=i['sha1'],width=i['width'],height=i['height'],creator_credit=artist,
            license=m['LicenseShortName']['value'],license_url=m['LicenseUrl']['value'],
            rights_evidence='Official file-specific imageinfo and own-work declaration; not a blanket Commons metadata licence',
            attribution_required=m.get('AttributionRequired',{}).get('value')=='true',
            source_group_id=group,rights_status='file_specific_declaration_supported_pending_execution_reverification',
            proposed_purpose='training-side suitability inspection only',
            known_group_matches=direct_group_hits,exact_local_sha1_matches=[r['image_id'] for r in checked if r['sha1']==i['sha1']],
            source_id_known=p['title'] in exclude['image_ids'] or 'commons-page:'+str(p['pageid']) in exclude['source_photo_ids'],
            visual_suitability='unverified; no preview or photograph acquired',native_boxes=0,native_masks=0,
            photographer_person_identity='uploader account declaration; cross-provider aliases not independently verified',
            physical_room_capture_groups='unverified; never infer independent evaluation eligibility',
            privacy_property_people_and_authenticity='pending visual screening; quarantine unresolved cases',
            annotations_active=False,acquisition_authorized=False,
            checked_at_utc=datetime.now(timezone.utc).isoformat(),metadata_receipt_sha256=sha(OUT/'commons-metadata-access.json'))
        if c['known_group_matches'] or c['exact_local_sha1_matches'] or c['source_id_known']:
            raise ValueError('Known candidate overlap; do not silently replace')
        candidates.append(c)
    candidates.sort(key=lambda c:c['source_id'])
    emit('proposed-photo-acquisition.json',dict(schema='modern-room-suitability-acquisition-v1',
        status='REQUEST_SEPARATE_APPROVAL',records=candidates,exact_photo_count=len(candidates),
        expected_image_bytes=sum(c['expected_bytes'] for c in candidates),
        cumulative_response_body_cap=60000000,per_image_cap=12582912,max_attempts_per_object=2,
        new_storage_cap=134217728,destination='data/licensed-rooms/image-pilots/modern-room-screening-v1/',
        metadata_budget_included=True,no_substitutions=True,no_creator_outreach=True,
        independent_evaluation_allocation=0,training_activation=False,
        required_before_each_GET=['Reverify exact file-specific rights, source size, object URL and source SHA1',
            'Stop on changed sources, provider restrictions or budget failures; no alternate IDs'],
        after_acquisition=['SHA256 and source SHA1 verification','Decode and orientation checks',
            'Visual authenticity, modern-room suitability and privacy screening',
            'Exact/perceptual duplicate and source-group checks','Quarantine uncertain examples; no automatic labels']))
    emit('input-preservation.json',dict(records=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths.values()],
        source_images_verified=120,real_images_downloaded=0,model_operations=0,
        main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()))
    print('Existing120 source photos verified; proposed',len(candidates),'photos;',sum(c['expected_bytes'] for c in candidates),'bytes; no images acquired')


if __name__=='__main__':main()
