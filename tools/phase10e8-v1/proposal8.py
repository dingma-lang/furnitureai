"""Exact offline licensed-data proposal;no provider requests or downloads."""
from collections import Counter
from common8q import *

def run():
    original=ROOT/'data/open-images/preparation/phase10e6-v1/conditional-rights-candidates-v1.json';prior=read(original)
    records=prior['records'];ids={r['image_id'] for r in records}
    if len(ids)!=64 or len(records)!=64:raise ValueError('Exact existing64IDshortlist required')
    registry=read(ROOT/'data/open-images/preparation/phase10c21-v1/historical-source-registry-v2.json')
    if ids&set(registry['image_ids']) or {r['source_photo_id'] for r in records}&set(registry['source_photo_ids']):raise ValueError('Known historical source/photo relationship')
    attempted=set();ledgers={}
    for rel in ('data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json','data/lvis/rights-audits/phase7-v1/license-and-attribution-ledger.json','data/lvis/rights-audits/phase8a-v1/license-attribution-ledger.json'):
        ledgers[rel]=sha(ROOT/rel)
        for r in read(ROOT/rel)['records']:
            attempted.add(r.get('image_id') or 'coco2017:'+str(r.get('coco_image_id',r.get('lvis_image_id'))).zfill(12))
    if ids&attempted:raise ValueError('Prior attempted/restricted/paused IDs must not be resumed')
    inventory=ROOT/'data/coco/metadata/phase6b-v1/candidate-inventory.jsonl';selected={}
    for line in inventory.open():
        r=json.loads(line)
        if r['image_id'] in ids:selected[r['image_id']]=r
    if set(selected)!=ids:raise ValueError('Shortlist source metadata IDs unavailable locally')
    output=[]
    for r in sorted(records,key=lambda r:r['image_id']):
        native=selected[r['image_id']]
        if native['source_identity']!=r['source_photo_id'] or native['native_category_box_counts']!=r['native_furniture_counts'] or native['proposed_secure_image_url']!=r['official_image_url']:
            raise ValueError('Original source/image/category provenance changed')
        output.append(dict(r,native_annotation_ids=native['native_annotation_ids'],native_source_partition=native['source_partition'],width=native['width'],height=native['height'],
            license_verified_this_phase=False,current_creator_or_capture_group_verified=False,photo_acquisition_authorized=False,
            permitted_next_stage='Separate approval for individual-photo rights evidence only;not bodies or model inference',
            annotation_completeness='Native positives only;project Chair includes stools,soCOCOChairabsence cannot certify all project seating absence'))
    manifest=DATA/'licensed-training-rights-candidates-v1.json'
    emit(manifest,dict(records=output,source_manifest=str(original.relative_to(ROOT)),source_manifest_sha256=sha(original),
         inventory=str(inventory.relative_to(ROOT)),inventory_sha256=sha(inventory),historical_registry_sha256=digest(registry),
         prior_rights_ledger_hashes=ledgers,prior_attempted_or_paused_IDs_excluded=len(attempted),new_requests=0,
         training_only_candidate_roles_preserved=True,no_final_partitions_assigned=True,automatic_replacements=False))
    classes=('sofa','chair','table','bed','desk','cabinet','lamp')
    counts=sum((Counter(r['native_furniture_counts']) for r in records),Counter())
    categories=read(ROOT/'data/coco/metadata/phase6b-v1/category-and-license-definitions.json')['categories'];native_names={r['name']:r['id'] for r in categories}
    proposal=dict(approved=False,stage='RIGHTS_EVIDENCE_APPROVAL_REQUIRED',exact_source_ids=64,
         exact_manifest=str(manifest.relative_to(ROOT)),exact_manifest_sha256=sha(manifest),verified_photo_rights=0,
         verified_eligible_new_hard_negatives=0,photo_acquisition_ready=False,source_availability_rechecked_online=False,
         sources='Already-local officialCOCO2017metadata;exact HTTPSFlickrphoto-page resolvers and officialHTTPSimages.cocodataset S3objects are in the manifest',
         categories={c:dict(metadata_candidate_images=sum(r['native_furniture_counts'].get(c,0)>0 for r in records),native_boxes=counts[c]) for c in classes},
         native_COCO_class_ids={k:native_names[k] for k in ('chair','couch','dining table','bed')},total_native_furniture_boxes=sum(counts.values()),
         masks='COCOinstance annotations are available locally;selected native annotation IDs preserved. Exact usable visible-mask counts and annotation completeness must be checked before photo acquisition;no new labels or complete supervision are promised.',
         unsupported_COCO_categories=['desk','cabinet','lamp'],
         desired_content=['People/body/clothing with clearly annotated seating nearby','Handbags and household object hardware','Screens,reflective surfaces and kitchen appliances','Plant/decoration versus real seating frames','Multiple fully boxed real Chairs and seating/bar stools,including occluded and partial cases'],
         stage_A=dict(scope='Exactly64frozen IDs:photo-specific rights,creator/source/capture grouping,evidence hashes/timestamps;no photograph bodies',
             maximum_response_body_bytes=32000000,maximum_allocated_storage_bytes=134217728,maximum_evidence_attempts_per_ID=1,
             minimum_provider_request_interval_seconds=3,one_worker=True,stop_provider_on_429_or_restriction=True,respect_Retry_After=True,
             do_not_resume_previously_paused296=True,no_access_bypass=True,TLS_verification_required=True,
             optional_HEAD_only_after_rights_checks='At most one HEAD for each exact official image URL;HEAD is only size/availability evidence,not copyright proof',
             destination='data/coco/rights-audits/phase10e8-v1/',separate_approval_required=True),
         stage_B=dict(scope='Only exact rights/identity-cleared survivors;request another frozen photograph manifest approval',
             maximum_images=64,guaranteed_verified_count=0,estimated_image_transfer_bytes=[9600000,25600000],estimates_not_HEAD_verified=True,
             maximum_transfer_bytes=100000000,per_image_maximum_bytes=3145728,maximum_attempts_per_image=2,maximum_new_storage_bytes=536870912,
             destination='data/coco/image-pilots/phase10e8-v1/',no_substitutions=True,separate_approval_required=True),
         licence_policy=dict(preferred=['CC0','CC BY'],photo_specific_evidence_required=True,metadata_license4_not_rights_clearance=True,
             require=['Exactphotoidentity','Currentacceptablephoto-specificlicenseandversion','Creatorcreditandoriginalsource','EvidenceURL/hash/timestamp','Attribution text and license URL'],
             quarantine=['Unverified or conflicting ownership/license','NC/ND/restricted/noncommercial terms','Inaccessible evidence','Creator/source/capture relationship with evaluation or historical groups']),
         required_annotations=['Preserve original native IDs,boxes,visible masks and image dimensions','Explicit project Chair plus seating/bar-stool taxonomy;footstools/ottomans/step stools separated','Source-backed visible negative regions with actual scene/crop inspection;no whole-photo absence inferred from missing labels','Whole-object extents and ignored/ambiguous furniture retained','Separate additional annotation approval if existing native evidence is incomplete'],
         source_group_isolation='Creator/capture groups currently unknown remain quarantined. Preserve existing partitions;new shortlisted IDs remain training-only candidates. No frozen or independent-acceptance sources may be used.',
         future_GPU_or_training='No inference/training approval implied. After approved acquisition and annotation audit,request separate bounded actual-query coverage verification before any loss/training change.',
         reason_for_expansion='Closed21candidate audit:zero new trusted high-confidence queries. Seek more independently licensed,source-complete room supervision rather than repeating training on unchanged easy negatives.',
         independent_evaluation='Separate newfullyannotatedseven-classsource-disjointcollection;none of these64trainingcandidates becomes acceptance material')
    emit(REPORT/'licensed-training-acquisition-proposal.json',proposal)
    print('Prepared64exact sourceIDs,268native furniture boxes;0individually rights-cleared;separate rights/photo approvals required')

if __name__=='__main__':run()
