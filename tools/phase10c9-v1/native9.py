"""Diagnostic-only native annotation join for the 19 exposed development photos.

This separate inventory can never become a training-negative source.
Streams already-local archives; no network or extraction writes.
"""
import csv
from common9 import *
sys.path.insert(0,str(ROOT/'tools/phase10c5-v1'))
from g5_inventory import json_entries
from g5_common import category_group,normalized_box

def run():
    _,dev=subsets();photos={r['image_id']:dict(image_id=r['image_id'],width=r['width'],height=r['height'],sha256=r['sha256'],annotations=[],role='internal_development_diagnostic_only',training_eligible=False) for r in dev}
    co={int(i.split(':')[1]):i for i in photos if i.startswith('coco2017:')};oi={i for i in photos if not i.startswith('coco2017:')}
    definitions=ROOT/'data/lvis/metadata/phase7-v1';lv={c['id']:c['name'] for c in read(definitions/'native-category-definitions.json')['categories']}
    cr=ROOT/'data/coco/metadata/phase6b-v1';cc={c['id']:c['name'] for c in read(cr/'category-and-license-definitions.json')['categories']}
    ori=ROOT/'data/open-images/metadata/phase5-v1'
    with (ori/'oidv7-class-descriptions-boxable.csv').open() as f:oc=dict(csv.reader(f))
    sources=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json')['metadata_sources']
    def add(ident,dataset,a,label,path):
        p=photos[ident]
        try:b=[float(a[k]) for k in ('XMin','YMin','XMax','YMax')] if dataset=='OpenImages' else normalized_box(a['bbox'],p['width'],p['height'])
        except ValueError:return
        if not 0<=b[0]<b[2]<=1 or not 0<=b[1]<b[3]<=1:return
        p['annotations'].append(dict(annotation_key=dataset+':'+(digest(a)[:20] if dataset=='OpenImages' else str(a['id'])),dataset=dataset,native_label=label,
            native_category_id=a['LabelName'] if dataset=='OpenImages' else a['category_id'],category=category_group(label),box=b,original_annotation=a,
            source=str(path.relative_to(ROOT)),source_sha256=sources[str(path.relative_to(ROOT))],training_eligible=False))
    for info in read(cr/'extraction-complete.json')['members']:
        path=cr/info['file'];seen=set();pending=[]
        if sha(path)!=sources[str(path.relative_to(ROOT))]:raise ValueError('COCO metadata changed')
        for kind,a in json_entries(path):
            if kind=='images' and a['id'] in co:
                p=photos[co[a['id']]]
                if (a['width'],a['height'])!=(p['width'],p['height']):raise ValueError('COCO coordinate registration mismatch')
                seen.add(a['id'])
            elif kind=='annotations' and a['image_id'] in co:pending.append(a)
        for a in pending:
            if a['image_id'] not in seen:raise ValueError('Unregistered COCO annotation')
            add(co[a['image_id']],'COCO',a,cc[a['category_id']],path)
        print('DIAGNOSTIC NATIVE COCO',len(pending),flush=True)
    for info in read(definitions/'archive-integrity.json')['archives']:
        path=definitions/info['file'];seen=set();pending=[]
        if sha(path)!=sources[str(path.relative_to(ROOT))]:raise ValueError('LVIS metadata changed')
        for kind,a in json_entries(path,info['member']):
            if kind=='images' and a['id'] in co:
                p=photos[co[a['id']]]
                if (a['width'],a['height'])!=(p['width'],p['height']):raise ValueError('LVIS coordinate registration mismatch')
                seen.add(a['id'])
            elif kind=='annotations' and a['image_id'] in co:pending.append(a)
        for a in pending:
            if a['image_id'] not in seen:raise ValueError('Unregistered LVIS annotation')
            add(co[a['image_id']],'LVIS',a,lv[a['category_id']],path)
        print('DIAGNOSTIC NATIVE LVIS',len(pending),flush=True)
    for filename in ('validation-annotations-bbox.csv','test-annotations-bbox.csv'):
        path=ori/filename
        if sha(path)!=sources[str(path.relative_to(ROOT))]:raise ValueError('Open Images metadata changed')
        n=0
        with path.open(newline='') as f:
            for a in csv.DictReader(f):
                if a['ImageID'] in oi:add(a['ImageID'],'OpenImages',a,oc[a['LabelName']],path);n+=int(a['ImageID'] in oi)
        print('DIAGNOSTIC NATIVE OpenImages',n,flush=True)
    save(OUT/'development-native-context.json',dict(photos=list(photos.values()),source_hashes=sources,training_eligible=False,missing_annotations_are_not_negatives=True))

if __name__=='__main__':run()
