"""Offline audit of approved local Open Images metadata. Never opens a URL."""
import argparse
import collections
import csv
import json
from pathlib import Path
import detection_prep as d


def audit(dictionary,hierarchy,box_files,label_files,image_files,out):
    if not all(len(v)==2 for v in (box_files,label_files,image_files)): raise ValueError('Require paired validation/test sources')
    if out.exists(): raise FileExistsError('Fresh metadata audit output required')
    mapping=d.class_mapping(dictionary);tree=d.read(hierarchy)
    mids=set()
    def visit(node):
        if not isinstance(node,dict) or not isinstance(node.get('LabelName'),str): raise ValueError('Unexpected hierarchy schema')
        mids.add(node['LabelName'])
        for child in node.get('Subcategory',[]):visit(child)
    visit(tree)
    if set(mapping['mapping'])-mids:raise ValueError('Mapped dictionary class missing from hierarchy')
    result=dict(mapping=mapping,partitions={},hierarchy_sha256=d.sha(hierarchy),
                hierarchy_policy='exact native labels only; hierarchy audited, not expanded for official mAP',
                proposed_images=dict(train=60,development=30,frozen_evaluation=60),
                actual_selected_images=None,actual_selected_boxes=None)
    sources={}; partition_ids=set()
    for partition,bp,lp,ip in zip(('validation','test'),box_files,label_files,image_files):
        boxes=d.parse_annotations(bp,mapping['mapping']);labels=d.parse_image_labels(lp,mapping['mapping'])
        image_info={}
        with open(ip,newline='') as stream:
            reader=csv.DictReader(stream)
            required={'ImageID','OriginalURL','OriginalLandingURL','License','Author','AuthorProfileURL','Rotation'}
            if not required<=set(reader.fieldnames or []):raise ValueError('Unexpected image information schema')
            for row in reader:
                ident=row['ImageID']
                if ident in image_info:raise ValueError('Repeated public image ID')
                image_info[ident]=row
        if partition_ids & set(image_info):raise ValueError('Official source partitions share image IDs')
        partition_ids.update(image_info)
        missing={r['image_id'] for r in boxes}-set(image_info)
        if missing:raise ValueError('BBox images lack image-source records')
        contradiction=[r for r in boxes if labels.get((r['image_id'],r['label']))!=1]
        # Do not repair a hierarchy/coverage mismatch silently.
        perclass={c:dict(native_mid=[mid for mid,label in mapping['mapping'].items() if label==c],
                   images=len({r['image_id'] for r in boxes if r['label']==c}),
                   boxes=sum(r['label']==c for r in boxes),
                   ordinary_boxes=sum(r['label']==c and not r['ignore'] for r in boxes),
                   ignored_group_depiction_or_unknown=sum(r['label']==c and r['ignore'] for r in boxes)) for c in d.CLASSES}
        ids={r['image_id'] for r in boxes}
        info=[dict(image_id=i,source_partition=partition,**{k:image_info[i][k] for k in required if k!='ImageID'}) for i in sorted(ids)]
        result['partitions'][partition]=dict(category_counts=perclass,total_furniture_images=len(ids),
             image_label_inconsistencies=len(contradiction),
             missing_license=sum(not image_info[i]['License'] for i in ids),
             license_status='metadata claim only; per-image source verification remains necessary',
             indoor_status='not established by furniture box presence',
             source_sha256=dict(boxes=d.sha(bp),human_labels=d.sha(lp),images=d.sha(ip)))
        sources[partition]=dict(source_records=info,annotations=boxes,
            image_label_coverage=[dict(image_id=i,label=c,value=v) for (i,c),v in sorted(labels.items()) if i in ids])
    out.mkdir(parents=True)
    d.write_new(out/'metadata-audit.json',result)
    d.write_new(out/'furniture-candidates.json',sources)
    if any(r['image_label_inconsistencies'] for r in result['partitions'].values()):
        raise ValueError('Native bbox/image-label mismatch; audit saved, acquisition selection blocked pending schema/hierarchy investigation')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dictionary',required=True,type=Path);ap.add_argument('--hierarchy',required=True,type=Path)
    ap.add_argument('--boxes',required=True,nargs=2,type=Path,metavar=('VALIDATION','TEST'))
    ap.add_argument('--labels',required=True,nargs=2,type=Path,metavar=('VALIDATION','TEST'))
    ap.add_argument('--images',required=True,nargs=2,type=Path,metavar=('VALIDATION','TEST'));ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();audit(a.dictionary,a.hierarchy,a.boxes,a.labels,a.images,a.out)
