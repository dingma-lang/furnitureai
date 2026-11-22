"""Stream already-local official metadata to audit labels omitted by older name filters."""
import csv
from common21 import *
from g5_inventory import json_entries
from g5_common import category_group,normalized_box


def run():
    cfg,prepared,train,dev,reg,inventory,active=context21();photos={p['image_id']:dict(image_id=p['image_id'],source_group=p['identity_group'],sha256=p['sha256'],width=p['width'],height=p['height'],path=p['path'],cohort=p['cohort'],rights=p['licence'],native_source_scopes=p['native_source_scopes'],annotations=[]) for p in inventory['photos']};coco={int(k.split(':')[1]):k for k in photos if k.startswith('coco2017:')};oi={k for k in photos if not k.startswith('coco2017:')}
    co_defs={c['id']:c['name'] for c in read(ROOT/'data/coco/metadata/phase6b-v1/category-and-license-definitions.json')['categories']};lv_defs={c['id']:c['name'] for c in read(ROOT/'data/lvis/metadata/phase7-v1/native-category-definitions.json')['categories']}
    with (ROOT/'data/open-images/metadata/phase5-v1/oidv7-class-descriptions-boxable.csv').open() as f:oi_defs=dict(csv.reader(f))
    rejected=[];source_counts=[]
    def add(ident,dataset,a,path,h):
        p=photos[ident];native=a['category_id'] if dataset!='OpenImages' else a['LabelName'];label=({'COCO':co_defs,'LVIS':lv_defs,'OpenImages':oi_defs}[dataset])[native]
        try:box=normalized_box(a['bbox'],p['width'],p['height']) if dataset!='OpenImages' else [float(a[k]) for k in ('XMin','YMin','XMax','YMax')]
        except ValueError as exc:rejected.append(dict(image_id=ident,dataset=dataset,native_id=a.get('id'),reason=str(exc)));return
        if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):rejected.append(dict(image_id=ident,dataset=dataset,reason='Out of frame'));return
        bad=bool(a.get('iscrowd') or a.get('ignore')) if dataset!='OpenImages' else any(a[k]=='1' for k in ('IsGroupOf','IsDepiction','IsInside'))
        group=category_group(label);lower=label.lower().replace('_',' ')
        furniture=group=='protected_furniture' or any(token in lower for token in ('chair','stool','seat','ottoman','bench','couch','sofa','bed','table','desk','cabinet','cupboard','bookcase','bookshelf','lamp','furniture'))
        key=dataset+':'+(str(a['id']) if dataset!='OpenImages' else digest(a)[:20]);p['annotations'].append(dict(annotation_key=key,dataset=dataset,native_category_id=native,native_label=label,context=group or 'other_native',box=box,has_mask=bool(a.get('segmentation')),native_ignored=bad,protect_furniture=furniture,part_or_semantics_unresolved=group=='furniture_parts' or 'drawer' in lower,source=str(path.relative_to(ROOT)),source_sha256=h,original_annotation_sha256=digest(a),original_annotation=a,Chair_absence_inferred=False))
    for dataset,folder in [('COCO',ROOT/'data/coco/metadata/phase6b-v1'),('LVIS',ROOT/'data/lvis/metadata/phase7-v1')]:
        specifications=read(folder/'extraction-complete.json')['members'] if dataset=='COCO' else read(folder/'archive-integrity.json')['archives']
        for source in specifications:
            path=folder/source['file'];h=sha(path)
            if h!=source['sha256']:raise ValueError('Official local native source changed')
            count=0
            for kind,a in json_entries(path,None if dataset=='COCO' else source['member']):
                if kind=='images' and a['id'] in coco:
                    p=photos[coco[a['id']]]
                    if (p['width'],p['height'])!=(a['width'],a['height']):raise ValueError('Native image dimensions differ')
                elif kind=='annotations' and a['image_id'] in coco:add(coco[a['image_id']],dataset,a,path,h);count+=1
            source_counts.append(dict(source=str(path.relative_to(ROOT)),sha256=h,local_annotation_records_seen=count));print('STREAMED',dataset,path.name,count,flush=True)
    for filename in ('validation-annotations-bbox.csv','test-annotations-bbox.csv'):
        path=ROOT/'data/open-images/metadata/phase5-v1'/filename;h=sha(path)
        if h!=inventory['metadata_sources'][str(path.relative_to(ROOT))]:raise ValueError('OI metadata changed')
        count=0
        with path.open(newline='') as f:
            for a in csv.DictReader(f):
                if a['ImageID'] in oi:add(a['ImageID'],'OpenImages',a,path,h);count+=1
        source_counts.append(dict(source=str(path.relative_to(ROOT)),sha256=h,local_annotation_records_seen=count));print('STREAMED OpenImages',filename,count,flush=True)
    eligible=[p for p in photos.values() if p['image_id'] in active];old_keys={a['annotation_key'] for p in inventory['photos'] if p['image_id'] in active for a in p['annotations']};new_annotations=[(p,a) for p in eligible for a in p['annotations'] if a['annotation_key'] not in old_keys]
    emit21(DATA21/'full-local-native-inventory.json',dict(photos=list(photos.values()),source_files=source_counts,rejected_geometry_records=rejected,source_annotations_modified=False))
    emit21(REPORT21/'full-native-audit.json',dict(local_inventory_photos=len(photos),optimization_photos=len(eligible),native_annotation_records=sum(len(p['annotations']) for p in photos.values()),optimization_annotation_records=sum(len(p['annotations']) for p in eligible),newly_visible_native_records=len(new_annotations),new_native_labels=dict(Counter(a['native_label'] for p,a in new_annotations)),source_file_counts=source_counts,rejected_geometry_records=len(rejected),inactive42_scanned_metadata_but_not_mined=True,source_labels_not_auto_negatives=True))
    print('Full local native audit complete;no images decoded or downloaded',flush=True)


if __name__=='__main__':run()
