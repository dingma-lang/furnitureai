"""Prepare a metadata registry, never prefill a human rectangle."""
from core import *

def run():
    if RAW.exists() or REPORT.exists():raise FileExistsError('New versioned destination required')
    RAW.mkdir(parents=True);REPORT.mkdir(parents=True)
    from PIL import Image
    old=read(RAW3/'region-evidence-template.json');priority={r['image_id'] for r in old['records']}
    rows=[r for r in manifest()['rows'] if optimizer_member(r)];rows.sort(key=lambda r:(r['image_id'] not in priority,r['image_id']))
    photos=[]
    for i,row in enumerate(rows):
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT/'data') or sha(path)!=row['sha256']:raise ValueError('Source changed')
        with Image.open(path) as im:
            if im.size!=(row['width'],row['height']):raise ValueError('Raw annotation dimensions mismatch')
            o=im.getexif().get(274,1)
            if o not in range(1,9):raise ValueError('Unsupported EXIF orientation')
        w,h=row['width'],row['height'];restricted=[];annotations=[]
        for a in row['annotations']:
            box=[v*s for v,s in zip(a['box'],(w,h,w,h))]
            seating=a['label']=='chair' or (a['native_dataset']=='OpenImages' and a['native_id']=='/m/0fqt361') or (a['native_dataset']=='LVIS' and a['native_id'] in (1018,558,710,19,358,464,889))
            reason='known Chair/seating-positive region' if seating else 'ignored or restricted native region'
            annotations.append(dict(label=a['label'],ignore=a['ignore'],box_pixels=box,native_dataset=a['native_dataset'],native_id=a['native_id']))
            if seating or a['ignore']:restricted.append(dict(box_pixels=box,reason=reason))
        dw,dh=display_size(w,h,o)
        photos.append(dict(key=f'photo-{i:03d}',image_id=row['image_id'],source_photo_id=row['source_photo_id'],source=row['source'],sha256=row['sha256'],identity_group=row['identity_group'],
            width=w,height=h,orientation=o,display_width=dw,display_height=dh,native_annotation_digest=digest(row['annotations']),
            historical_partition='train',internal_partition='optimization',priority_scope_review=row['image_id'] in priority,
            annotations=annotations,restricted=restricted,whole_image_complete=False))
    reviews=read(RAW3/'review-validation.json')['records'];clar=[r for r in reviews if r['clarification_flags']]
    if len(photos)!=59 or len(clar)!=24:raise ValueError('Registry/review inputs changed')
    reg=dict(schema='phase10c4-registry-v1',source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        taxonomy=TAXONOMY,taxonomy_sha256=digest(TAXONOMY),photos=photos,clarifications=clar,
        previous_review_sha256=sha(RAW3/'review-validation.json'),source_template_sha256=sha(RAW3/'region-evidence-template.json'),
        coordinate_frame='Original stored-image pixel EDGES XYXY, integer half-open bounds; EXIF display transform is reversible',
        confirmed_regions_initial=0,source_groups_initial=0)
    save(RAW/'registry.json',reg)
    initial=dict(schema=SCHEMA,registry_sha256=digest(reg),taxonomy_sha256=digest(TAXONOMY),regions=[],clarifications=[])
    save(RAW/'empty-session.json',initial)
    save(REPORT/'summary.json',dict(eligible_training_photos=59,known_training_groups=len({p['identity_group'] for p in photos}),priority_photos=9,
        clarification_items=24,training_scope_clarifications=3,diagnostic_only_clarifications=21,confirmed_negative_rectangles=0,confirmed_source_groups=0,
        human_rectangles_prefilled=False,registry_sha256=digest(reg),training_enabled=False,orientation_counts={str(o):sum(p['orientation']==o for p in photos) for o in range(1,9)}))
    print('Prepared59 optimization photos;24 separate clarification items;0 rectangles/0 confirmations')

if __name__=='__main__':run()
