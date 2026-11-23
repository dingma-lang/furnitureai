"""Local-only64photo identity/annotation/isolation audit,streaming existing COCO."""
import collections,importlib.util,re
from common9 import *
MAP={62:'chair',63:'sofa',65:'bed',67:'table'}
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only phase')
    if OUT.exists() or REPORT.exists():raise FileExistsError('Approved fresh rights destination required')
    sys.path.insert(0,str(ROOT/'tools/phase10e8-v1'));import verify8
    verify8.check()
    m=manifest();rows=m['records'];wanted={int(r['image_id'].split(':')[1]):r for r in rows}
    registry_path=ROOT/'data/open-images/preparation/phase10c21-v1/historical-source-registry-v2.json';registry=read(registry_path)
    old_ids=set(registry['image_ids']);old_sources=set(registry['source_photo_ids']);old_groups=set(registry['source_groups']);old_attempted=set();alias_evidence={};prior_dates=[]
    for rel,h in m['prior_rights_ledger_hashes'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Historical rights ledger changed')
        for r in read(ROOT/rel)['records']:
            ident=r.get('image_id') or 'coco2017:'+str(r.get('coco_image_id',r.get('lvis_image_id'))).zfill(12);old_attempted.add(ident)
            for alias in r.get('creator_aliases',[]):old_groups.add(alias.lower());alias_evidence.setdefault(alias.lower(),[]).append(ident)
    if len(old_attempted)!=844:raise ValueError('Prior844attempted/paused exclusion inventory changed')
    source=ROOT/'data/coco/metadata/phase6b-v1/extracted/annotations/instances_train2017.json'
    if sha(source)!='610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d':raise ValueError('Official COCO metadata changed')
    spec=importlib.util.spec_from_file_location('_E9_bounded_coco_stream',ROOT/'tools/phase6b-v1/streaming.py');stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
    categories={};images={};annotations=collections.defaultdict(list);licenses={}
    for kind,a in stream.entries(source):
        if kind=='categories':categories[a['id']]=a
        elif kind=='licenses':licenses[a['id']]=a
        elif kind=='images' and a['id'] in wanted:images[a['id']]=a
        elif kind=='annotations' and a['image_id'] in wanted:annotations[a['image_id']].append(a)
    if set(images)!=set(wanted):raise ValueError('Approved COCO images missing')
    local_names=collections.defaultdict(list)
    for root in (ROOT/'data',ROOT/'assets',ROOT/'samples'):
        if root.exists():
            for p in root.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):local_names[p.name].append(str(p.relative_to(ROOT)))
    adapted=[]
    for cid,row in wanted.items():
        im=images[cid];fid=row['source_photo_id'].split(':')[1]
        if row['image_id'] in old_ids or row['source_photo_id'] in old_sources or row['image_id'] in old_attempted:raise ValueError('Historical source or prior paused ID overlap')
        if (im['width'],im['height'],im['license'])!=(row['width'],row['height'],row['metadata_licence_id']):raise ValueError('Native image dimensions/licence mismatch')
        if not re.search(r'/'+re.escape(fid)+r'_[^/]+\.(?:jpg|png|jpeg)$',im.get('flickr_url',''),re.I):raise ValueError('Official Flickr asset ID mismatch')
        furniture=[a for a in annotations[cid] if a['category_id'] in MAP]
        if {a['id'] for a in furniture}!=set(row['native_annotation_ids']) or dict(collections.Counter(MAP[a['category_id']] for a in furniture))!=row['native_furniture_counts']:
            raise ValueError('Furniture annotation counts/IDs changed')
        actual=[]
        for a in annotations[cid]:
            b=a['bbox'];valid=len(b)==4 and b[2]>0 and b[3]>0 and min(b[:2])>=0 and b[0]+b[2]<=im['width']+1e-6 and b[1]+b[3]<=im['height']+1e-6
            seg=a.get('segmentation');kind='polygon' if isinstance(seg,list) and seg else 'RLE' if isinstance(seg,dict) else 'missing'
            actual.append(dict(original_annotation=a,sha256=digest(a),native_label=categories[a['category_id']]['name'],box_valid=valid,segmentation_kind=kind))
        adapted.append(dict(row,original_dataset='COCO2017',native_image=im,native_image_sha256=digest(im),
            original_host='Flickr',canonical_source_url=None,documented_creator=None,source_group=None,capture_session_group=None,
            local_filename_matches=local_names.get(im['file_name'],[]),known_local_availability='no known local filename/source match' if not local_names.get(im['file_name']) else 'unresolved filename match',
            native_annotations=actual,native_category_counts=dict(collections.Counter(a['native_label'] for a in actual)),
            photograph_SHA256=None,perceptual_duplicate_screening_performed=False,visual_content_verified=False,
            annotation_completeness_verified=False,Chair_absence_inferred=False,final_dataset_partition_assigned=False))
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    emit(OUT/'authorization.json',dict(approved=True,phase='10E.9',exact_manifest_sha256=PIN,candidates=64,network_transfer_cap=CAP,new_storage_cap=DISK,
          metadata_rights_only=True,photograph_bodies=False,GPU_inference=False,training=False,max_evidence_attempts_per_ID=1,min_request_interval_seconds=3,
          destination=str(OUT.relative_to(ROOT)),annotation_files_are_preexisting=True))
    emit(OUT/'local-candidate-audit.json',dict(records=adapted,source=str(source.relative_to(ROOT)),source_sha256=sha(source),categories=categories,licences=licenses,
          annotation_licence='https://creativecommons.org/licenses/by/4.0/',annotation_creator='COCO Consortium',missing_labels_are_not_negative=True))
    emit(OUT/'historical-exclusion-registry.json',dict(image_ids=sorted(old_ids),source_photo_ids=sorted(old_sources),source_groups=sorted(old_groups),
          previous_attempted_or_paused_ids=sorted(old_attempted),source_group_evidence=alias_evidence,source_registry_sha256=sha(registry_path)))
    emit(REPORT/'preflight.json',dict(status='PASS',candidates=64,source_IDs_unique=64,known_historical_photo_overlap=0,previous_attempted_or_paused_ID_overlap=0,
          prior_attempted_or_paused_IDs_excluded=844,source_annotations=sum(len(r['native_annotations']) for r in adapted),furniture_boxes=sum(len(r['native_annotation_ids']) for r in adapted),
          source_images_decoded=0,proposed_training_only_roles_preserved=True,partitions_assigned=0,current_creator_capture_identity_unverified=64,
          photo_licence_claims_only_not_verified=True,original_licence_definitions_preserved=True,exact_manifest_sha256=PIN))
    print('Local preflight PASS64exact photos,268furniture boxes;all source evidence/creator rights remain unverified',flush=True)
if __name__=='__main__':run()
