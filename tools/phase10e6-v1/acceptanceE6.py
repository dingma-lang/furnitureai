"""Independent all-seven acceptance intake. No photographs/reviews prefilled."""
import argparse,datetime,math,re
from urllib.parse import urlsplit
from commonE6 import *

def validate(record,registry,verify_files=True):
    for key in ('image_id','source_photo_id','source_group','capture_session_group','path','rights_evidence_path','license_url','creator','source_url','license_version','rights_verified_at'):
        if not isinstance(record.get(key),str) or not record[key].strip():raise ValueError('Required independent identity/rights field:'+key)
    for key in ('sha256','rights_evidence_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',str(record.get(key,''))):raise ValueError('Exact identity/evidence checksum required')
    for field,registry_key in (('image_id','image_ids'),('sha256','sha256'),('source_photo_id','source_photo_ids'),('source_group','source_groups'),('capture_session_group','source_groups')):
        if record[field].lower() in {str(v).lower() for v in registry[registry_key]}:raise PermissionError('Historical/exposed/frozen identity cannot enter independent acceptance')
    if record.get('role')!='independent_acceptance' or record.get('source_group_independence_verified') is not True or not record.get('source_group_evidence') or record.get('rights_verified') is not True:
        raise ValueError('Independence and individual rights evidence required')
    if not record['source_url'].startswith('https://') or not record['license_url'].startswith('https://'):
        raise ValueError('Certificate-verified source and licence URLs required')
    verified_at=datetime.datetime.fromisoformat(record['rights_verified_at'].replace('Z','+00:00'))
    if verified_at.utcoffset() is None:raise ValueError('Rights verification timestamp requires timezone')
    licence=urlsplit(record['license_url'])
    allowed={'/publicdomain/zero/1.0/','/licenses/by/2.0/','/licenses/by/2.5/','/licenses/by/3.0/','/licenses/by/4.0/'}
    if licence.netloc.lower() not in ('creativecommons.org','www.creativecommons.org') or licence.path not in allowed:
        raise ValueError('Individually evidenced CC0/CC BY commercial policy required')
    review=record.get('review',{})
    if review.get('human_reviewed') is not True or review.get('blind_to_model_predictions') is not True or not all(review.get(k) for k in ('reviewer','date','evidence')):
        raise ValueError('Real blind human completeness review required')
    datetime.date.fromisoformat(review['date'])
    if record.get('category_complete')!={c:True for c in CLASSES} or record.get('Chair_includes_seating_and_bar_stools') is not True or record.get('whole_object_taxonomy_confirmed') is not True or record.get('unresolved_objects')!=[]:
        raise ValueError('All7 completeness,taxonomy and unresolved-object gates required')
    anns=record.get('annotations')
    if not isinstance(anns,list):raise ValueError('Explicit annotation list required')
    for a in anns:
        b=a.get('box',[])
        if a.get('label') not in CLASSES or len(b)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in b) or not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1) or type(a.get('ignore')) is not bool:
            raise ValueError('Invalid native-independent annotation')
    counts={c:sum(a['label']==c and not a['ignore'] for a in anns) for c in CLASSES}
    declared=record.get('verified_visible_instance_counts',{})
    if declared!=counts or any(type(v) is not int for v in declared.values()):raise ValueError('Completeness counts must agree with annotations,including explicit zero categories')
    if verify_files:
        for key,hk in (('path','sha256'),('rights_evidence_path','rights_evidence_sha256')):
            p=(ROOT/record[key]).resolve()
            if not p.is_relative_to(ROOT/'data') or sha(p)!=record[hk]:raise ValueError('Local independent source/evidence mismatch')
        from PIL import Image
        with Image.open(ROOT/record['path']) as im:
            if im.size!=(record.get('width'),record.get('height')) or im.getexif().get(274,1)!=1:
                raise ValueError('Original dimensions/orientation must be verified before independent annotation')
            im.verify()
    return dict(eligible=True,counts=counts,synthetic_test_only=not verify_files)

def validate_manifest(records,registry,verify_files=True):
    seen={k:set() for k in ('image_id','sha256','source_photo_id','source_group','capture_session_group')}
    for r in records:
        validate(r,registry,verify_files)
        for key,values in seen.items():
            v=r[key].lower()
            if v in values:raise ValueError('Repeated acceptance identity/group:'+key)
            values.add(v)
    return dict(eligible_photographs=len(records),source_groups=len(seen['source_group']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('submission',type=Path)
    p.add_argument('--registry',type=Path,default=DATA/'independent-exclusion-registry.json')
    a=p.parse_args()
    # CLI has no synthetic-test bypass and never changes a submitted review.
    submission=read(a.submission)
    print(json.dumps(validate_manifest(submission['records'],read(a.registry)),sort_keys=True))
