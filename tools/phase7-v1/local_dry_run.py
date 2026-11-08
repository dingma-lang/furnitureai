"""Count already-local metadata and prepare ID-only LVIS join targets."""
import argparse
from contracts import ROOT,OUT,read,write_new
from audit import oi_counts,stream
from inventory import flickr_id

def oi_source_index(records):
    return {alias:r for r in records if r['source']=='openimages' for alias in r['source_aliases'] if alias.startswith('flickr-photo:')}

def run(version='v1'):
    local={}
    for part in ('validation','test'):
        local[part]=oi_counts(ROOT/'data/open-images/metadata/phase5-v1'/f'{part}-annotations-bbox.csv',part)
    local['training']={'images':None,'boxes':None,'commercial_verified_candidates':None,
        'reason':'2,258,447,590-byte source training box CSV plus image/label metadata require separate substantial-download approval'}
    write_new(OUT/f'openimages-local-counts-{version}.json',local)
    protected=read(OUT/'protected-source-inventory-v1.json')['records']
    source_index=oi_source_index(protected)
    targets={r['coco_image_id']:dict(coco_image_id=r['coco_image_id'],historical_partition=r['partition'],
        historical_status=r['historical_status'],historical_source=r['canonical_id'],
        additional_lvis_annotations=None,lvis_membership=None,rights_inherited_only_for_identical_photo=True)
        for r in protected if r['source']=='coco2017'}
    matches=[]
    for part in ('train','val'):
        path=ROOT/'data/coco/metadata/phase6b-v1/extracted/annotations'/f'instances_{part}2017.json'
        for kind,r in stream(path):
            if kind!='images':continue
            fid=flickr_id(r.get('flickr_url',''))
            if fid in source_index:
                old=source_index[fid]
                match=dict(coco_image_id=r['id'],coco_source_split=part,source_alias=fid,
                    historical_source=old['canonical_id'],historical_partition=old['partition'],
                    historical_status=old['historical_status'],lvis_membership=None,additional_lvis_annotations=None,
                    reuse_authorized=False)
                matches.append(match);targets.setdefault(r['id'],match)
    write_new(OUT/f'lvis-overlap-targets-{version}.json',dict(existing_coco_pilot_images=72,
        coco_matches_to_historical_openimages=matches,unique_target_coco_ids=len(targets),targets=list(targets.values()),
        actual_lvis_overlap_count=None,actual_added_lvis_masks=None,
        why='No LVIS archive acquired. These are exact COCO/source-ID join targets, not confirmed LVIS membership or new training permission.',
        frozen_pixels_opened=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['v1','v2'],default='v1');args=p.parse_args();run(args.version)
