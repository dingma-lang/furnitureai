"""Private modern-room intake metadata validation; never acquires or activates data."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/phase10e11-v1')]
from intake11 import validate as validate_permission
from furniture_bench.data import CATEGORIES, load_image_with_digest

SCHEMA='modern-room-private-intake-v1'
TAXONOMY='modern-room-seven-v1-chair-includes-seating-stools'
FRAME='exif-oriented-xyxy-pixels'


def text(value):return isinstance(value,str) and bool(value.strip())
def sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def box_valid(box,width,height):
    return (isinstance(box,list) and len(box)==4 and all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
            and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)


def validate_source(document,photo):
    """Require exact identity/frame; never guess raw-EXIF or amodal transformations."""
    if not isinstance(document,dict) or document.get('schema')!='modern-room-source-annotations-v1':
        raise ValueError('Unsupported source annotation envelope')
    if any(document.get(k)!=photo.get(k) for k in ('image_sha256','width','height','box_frame')):
        raise ValueError('Source annotations do not match exact oriented photograph')
    if not text(document.get('source_id')) or not sha(document.get('source_file_sha256')) or not text(document.get('annotation_license')):
        raise ValueError('Native annotation provenance and licence required')
    records=document.get('annotations')
    if not isinstance(records,list) or len(records)>1000:raise ValueError('Bounded annotation list required')
    seen=set()
    for r in records:
        if not isinstance(r,dict) or not text(r.get('id')) or r['id'] in seen:
            raise ValueError('Unique source annotation IDs required')
        seen.add(r['id'])
        if not box_valid(r.get('box'),photo['width'],photo['height']):raise ValueError('Invalid source box')
        if not text(r.get('native_label')) or not isinstance(r.get('native_annotation'),dict):
            raise ValueError('Preserve native label and original annotation')
        if r.get('suggested_category') not in (None,*CATEGORIES):raise ValueError('Unknown proposed mapping')
    return records  # Proposals only; source origin does not imply human confirmation.


def validate(record,existing=None,exclusions=None,image_path=None):
    existing=existing or [];exclusions=exclusions or {}
    errors=[]
    if not isinstance(record,dict):raise ValueError('Intake record must be an object')
    for key in ('photo_ids','room_group_ids'):
        if not isinstance(record.get(key),list) or not record[key] or any(not text(v) for v in record[key]):
            errors.append('Nonempty '+key+' list required')
    if errors:return dict(status='invalid',errors=errors,annotations_active=False)
    if any(not text(record.get(k)) for k in ('creator_id','source_group_id','capture_group_id')):
        return dict(status='invalid',errors=['Creator/source/capture IDs must be nonempty strings'],annotations_active=False)
    base=validate_permission(record,existing)
    errors+=base['errors']
    if len(record['photo_ids'])!=1:errors.append('One photograph per intake record required')
    if record.get('schema')!=SCHEMA or record.get('taxonomy_version')!=TAXONOMY:errors.append('Schema/taxonomy mismatch')
    for key in ('creator_id','source_group_id','capture_group_id','permission_evidence','permission_date','credit'):
        if not text(record.get(key)):errors.append('Missing '+key)
    if not sha(record.get('permission_evidence_sha256')):errors.append('Permission evidence file SHA256 required')
    photo=record.get('photo',{})
    if not isinstance(photo,dict):photo={};errors.append('Photograph metadata object required')
    valid_dimensions=all(type(photo.get(k)) is int and 0<photo[k]<=100000 for k in ('width','height'))
    if (not valid_dimensions or not sha(photo.get('image_sha256')) or photo.get('box_frame')!=FRAME
            or type(photo.get('bytes')) is not int or not 0<photo['bytes']<=20*1024*1024):
        errors.append('Invalid photograph identity, dimensions, frame or size')
    if record.get('orientation_explicitly_checked') is not True:errors.append('Orientation confirmation required')
    for key,target in (('photo_ids','image_ids'),('photo_ids','source_photo_ids')):
        if set(record[key])&set(exclusions.get(target,[])):errors.append('Historical/exposed image ID conflict')
    if photo.get('image_sha256') in exclusions.get('sha256',[]):errors.append('Historical/exposed image hash conflict')
    grouping={record.get('creator_id'),record.get('source_group_id'),record.get('capture_group_id'),*record['room_group_ids']}
    if grouping&set(exclusions.get('source_groups',[])):errors.append('Historical/exposed group conflict')
    for old in existing:
        prior={old.get('creator_id'),old.get('source_group_id'),old.get('capture_group_id'),*old.get('room_group_ids',[])}-{None}
        if photo.get('image_sha256')==old.get('photo',{}).get('image_sha256'):errors.append('Duplicate image bytes')
        if grouping&prior and old.get('purpose')!=record.get('purpose'):errors.append('Capture/creator/room group crosses purposes')
    if record.get('purpose')=='independent_evaluation' and record.get('exposure_status')!='not_model_exposed_attested':
        errors.append('Independent evaluation needs explicit unexposed attestation')
    annotations=record.get('annotations',[])
    source=record.get('source_annotations')
    if source is not None:
        try:validate_source(source,photo)
        except ValueError as e:errors.append(str(e))
    if not isinstance(annotations,list) or len(annotations)>1000:annotations=[];errors.append('Bounded annotation list required')
    confirmed=0;seen=set()
    for a in annotations:
        if not isinstance(a,dict):errors.append('Annotation object required');continue
        if not text(a.get('id')) or a['id'] in seen:errors.append('Duplicate/missing annotation ID')
        seen.add(a.get('id'))
        if a.get('decision') not in ('proposal','confirmed','ignored','rejected'):errors.append('Invalid annotation decision')
        if valid_dimensions and not box_valid(a.get('box'),photo['width'],photo['height']):errors.append('Invalid annotation box')
        if a.get('decision')=='confirmed':
            if a.get('category') not in CATEGORIES:errors.append('Confirmed object category required')
            if a.get('explicit_human_confirmation') is not True or not all(text(a.get(k)) for k in ('reviewer','reviewed_at','evidence')):
                errors.append('Confirmed object requires explicit human evidence')
            else:confirmed+=1
        if a.get('decision')=='ignored' and not text(a.get('evidence')):errors.append('Ignored region needs a reason')
        if a.get('origin')=='source_proposal' and (not isinstance(a.get('native_annotation'),dict) or not sha(a.get('source_file_sha256'))):
            errors.append('Imported annotation provenance missing')
    completeness=record.get('class_completeness',{})
    complete=True
    for c in CATEGORIES:
        d=completeness.get(c,{})
        if not isinstance(d,dict) or d.get('status') not in ('unknown','partial','complete','reviewed_absent'):
            errors.append('Completeness decision required: '+c);complete=False;continue
        if d['status'] not in ('complete','reviewed_absent'):complete=False
        if d['status'] in ('complete','reviewed_absent') and not all(text(d.get(k)) for k in ('reviewer','reviewed_at','evidence')):
            errors.append('Completeness evidence missing: '+c);complete=False
        if d['status']=='reviewed_absent' and any(a.get('decision')=='confirmed' and a.get('category')==c for a in annotations):
            errors.append('Absent category contradicts confirmed positive: '+c)
    if record.get('purpose')=='independent_evaluation' and (not complete or any(a.get('decision')=='proposal' for a in annotations)):
        errors.append('Independent evaluation annotations incomplete/unresolved')
    if image_path is not None:
        image,actual=load_image_with_digest(image_path,photo.get('image_sha256'))
        if image.size!=(photo.get('width'),photo.get('height')):errors.append('Actual oriented dimensions differ')
    # This validator cannot establish actual authority, privacy or cross-provider aliases.
    return dict(status='invalid_or_incomplete' if errors else 'metadata_ready_for_rights_identity_and_annotation_qualification',
        errors=sorted(set(errors)),confirmed_object_records=confirmed,complete_seven_class_review=complete,
        image_content_verified=image_path is not None,rights_verified=False,group_independence_verified=False,
        annotations_active=False,eligible_for_training=False,eligible_for_independent_evaluation=False,
        reason='Activation requires a separately approved qualified dataset manifest; exported checkboxes are not automatic rights clearance')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('record',type=Path);parser.add_argument('--image',type=Path)
    parser.add_argument('--exclusions',type=Path,default=ROOT/'data/open-images/preparation/phase10e6-v1/independent-exclusion-registry.json')
    parser.add_argument('--existing',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    result=validate(json.loads(args.record.read_text()),[json.loads(p.read_text()) for p in args.existing],
                    json.loads(args.exclusions.read_text()),args.image)
    print(json.dumps(result,indent=2));sys.exit(1 if result['errors'] else 0)
