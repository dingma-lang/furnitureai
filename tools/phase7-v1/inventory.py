"""Read-only known-photo/group protection inventory. No frozen pixels or scoring."""
import csv
import re
from contracts import ROOT,OUT,read,write_new

def flickr_id(url):
    patterns=[r'/photos/[^/]+/(\d+)',r'/(\d+)_[^/]+\.(?:jpg|png)',r'[?&]id=(\d+)']
    for pattern in patterns:
        match=re.search(pattern,url or '',re.I)
        if match:return 'flickr-photo:'+match.group(1)
    return None

def protected_inventory():
    paths=[('openimages','data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json'),
           ('openimages','data/open-images/preparation/phase6a-v1/acquisition-manifest-v2.json'),
           ('coco2017','data/coco/image-pilots/phase6d-v1/screening-v2.json')]
    result=[]
    for source,filename in paths:
        for row in read(ROOT/filename)['images']:
            ident=row['image_id'];canonical=ident if ident.startswith(source+':') else source+':'+ident
            aliases={canonical};groups=set()
            for key in ('source_identity','original_url','original_landing_url','verified_source_page','original_flickr_asset_url'):
                value=row.get(key)
                if value:
                    if value.startswith('flickr-photo:'):aliases.add(value)
                    else:
                        fid=flickr_id(value)
                        if fid:aliases.add(fid)
            for key in ('identity_group','known_source_group','author_profile_url','creator_url'):
                if row.get(key):groups.add(row[key])
            result.append(dict(source=source,source_image_id=ident,canonical_id=canonical,
                coco_image_id=row.get('coco_image_id'),source_aliases=sorted(aliases),source_groups=sorted(groups),
                partition=row['proposed_split'],historical_status=row.get('status','preserve_original_membership'),
                sha256=row.get('sha256'),source_manifest=filename,novel_training_photo=False))
    index={r['source_image_id']:r for r in result if r['source']=='openimages'}
    for part in ('validation','test'):
        with (ROOT/'data/open-images/metadata/phase5-v1'/f'{part}-images-with-rotation.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                if row['ImageID'] not in index:continue
                dest=index[row['ImageID']];aliases=set(dest['source_aliases'])
                for key in ('OriginalURL','OriginalLandingURL'):
                    fid=flickr_id(row[key])
                    if fid:aliases.add(fid)
                dest['source_aliases']=sorted(aliases)
                if row.get('AuthorProfileURL'):dest['source_groups']=sorted(set(dest['source_groups'])|{row['AuthorProfileURL']})
    return dict(schema='phase7-known-photo-protection-v1',records=result,
        historical_photograph_slots=len(result),new_training_images=0,
        rule='Any identity/hash/source-family hit must preserve its old partition/status. Adding LVIS labels is an annotation overlay, never a new independent photograph. Unknown identities stay provisional.',
        abo_protection='Official ABO holds and Phase3 frozen panel are sealed by preserve.py. No ABO images are read by this metadata join; future pixel duplicates still need screening.',
        limitation='Source account aliases are not globally canonicalized here; existing known group IDs are retained. New source aliases require graph resolution before acquisition eligibility.')

if __name__=='__main__':write_new(OUT/'protected-source-inventory-v1.json',protected_inventory())
