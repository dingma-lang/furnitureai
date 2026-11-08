"""Offline metadata audits. No network client, model loading or image decoding."""
import argparse
import ast
import collections as C
import csv
import importlib.util
import io
import re
import stat
import zipfile
from pathlib import Path
from contracts import ROOT, OUT, LVIS, OI, box, read, sha, write_new, lvis_scope

def literal(path, variable):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Published table variable unavailable')

def category_summary():
    src = OUT/'sources'
    cats = literal(src/'lvis-categories.py','LVIS_CATEGORIES')
    counts = {r['id']:r['image_count'] for r in literal(src/'lvis-image-counts.py','LVIS_CATEGORY_IMAGE_COUNT')}
    extras = {19:'chair',170:'bed',285:'table',301:'ambiguous_sofa_bed',346:'ambiguous_closet_cupboard',358:'outdoor_chair',367:'table',438:'cabinet_subtype',464:'chair',558:'highchair',610:'table',627:'outdoor_lamppost',628:'excluded_part',731:'oil_lamp',889:'chair',1051:'lamp'}
    result = []
    for c in cats:
        if c['id'] in LVIS or c['id'] in extras:
            result.append(dict(c, proposed_category=LVIS.get(c['id'],extras.get(c['id'])),
                               mapping='core_native' if c['id'] in LVIS else 'separate_subtype_review_required',
                               published_train_images=counts[c['id']],
                               instance_count=None, rights_verified_image_count=None))
    return dict(schema='phase7-lvis-category-audit-v1',source_categories=len(cats),categories=result,
                counts_source='Detectron2 LVIS v1 training category image-count table; generated from official annotation categories',
                counts_are_not_disjoint=True,instance_counts_available=False,
                local_photo_overlap_count=None,additional_instances_for_local_photos=None,
                reason='LVIS annotation archives not downloaded; image-count table has no IDs or instance counts')

def oi_counts(path, expected_split):
    counts = C.Counter(); images = {k:set() for k in OI}; ignored = C.Counter(); row_count=0
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        required={'ImageID','LabelName','XMin','XMax','YMin','YMax','IsGroupOf','IsDepiction','IsInside','IsOccluded','IsTruncated','Confidence'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Unexpected Open Images bbox schema')
        for r in reader:
            row_count += 1
            if not re.fullmatch('[0-9a-f]{16}',r['ImageID']):
                raise ValueError('Unsafe Open Images ID')
            if r['LabelName'] not in OI:
                continue
            x,y,x2,y2 = [float(r[k]) for k in ('XMin','YMin','XMax','YMax')]
            box([x,y,x2-x,y2-y],1,1)
            if float(r['Confidence']) != 1 or any(r[k] not in ('0','1') for k in ('IsGroupOf','IsDepiction','IsInside','IsOccluded','IsTruncated')):
                raise ValueError('Unexpected Open Images furniture confidence/flags')
            mid = r['LabelName']; counts[mid] += 1; images[mid].add(r['ImageID'])
            for k in ('IsGroupOf','IsDepiction','IsInside','IsOccluded','IsTruncated'):
                ignored[mid,k] += int(r[k])
    return dict(source_file=str(path),sha256=sha(path),source_split=expected_split,all_source_bbox_rows=row_count,
                classes=[dict(category=OI[mid],source_mid=mid,images=len(images[mid]),boxes=counts[mid],
                              flags={k:ignored[mid,k] for k in ('IsGroupOf','IsDepiction','IsInside','IsOccluded','IsTruncated')}) for mid in OI],
                rights_verified_image_count=None,counts_are_not_disjoint=True,
                no_missing_categories_are_background=True,source_membership_join_verified=False,
                source_split_is_caller_assertion=True,
                candidate_eligibility='Raw native counts only; join pinned official image-info and human labels, licences and known groups before selection')

def stream(path):
    spec=importlib.util.spec_from_file_location('phase7_pinned_json_stream',ROOT/'tools/phase6b-v1/streaming.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if Path(path).suffix.lower() != '.zip':
        yield from module.entries(path)
        return
    with zipfile.ZipFile(path) as archive:
        member=safe_lvis_member(archive)
        module.open=lambda *_args,**_kwargs:io.TextIOWrapper(archive.open(member),encoding='utf-8')
        yield from module.entries(path)

def safe_lvis_member(archive):
    members=archive.infolist()
    if len(members) != 1:
        raise ValueError('Unexpected LVIS archive layout; requires review')
    item=members[0]
    if item.filename not in ('lvis_v1_train.json','lvis_v1_val.json') or item.flag_bits&1 or stat.S_ISLNK(item.external_attr>>16):
        raise ValueError('Unsafe LVIS archive entry')
    if not 0 < item.file_size <= 4*1024**3 or item.file_size > max(1,item.compress_size)*100:
        raise ValueError('LVIS archive expansion bound exceeded')
    return item.filename

def lvis_local(path, approved_rows):
    """Join by COCO ID, preserving existing partitions; overlap never implies new training permission."""
    known={int(r['coco_image_id']):r for r in approved_rows}
    cats={}; images={}; counts=C.Counter(); instance_counts=C.Counter(); matches={}; ids=set()
    # Official arrays need not occur in a particular order.
    for kind,r in stream(path):
        if kind=='categories':
            if r['id'] in cats:raise ValueError('Duplicate category')
            cats[r['id']]=r
        if kind=='images':
            if r['id'] in images:raise ValueError('Duplicate LVIS image')
            images[r['id']]=r
            if r['id'] in known:
                url=r.get('coco_url','');native_id=re.search(r'/(\d{12})\.jpg$',url)
                if not native_id or int(native_id.group(1)) != r['id']:
                    raise ValueError('LVIS/COCO identity mismatch')
                old=known[r['id']]
                if old.get('width',r['width']) != r['width'] or old.get('height',r['height']) != r['height']:
                    raise ValueError('LVIS/COCO dimension mismatch')
                matches[r['id']]=dict(source=old,annotations=[])
    for cid in LVIS:
        expected={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}[cid]
        if cid not in cats or cats[cid]['name'] != expected:raise ValueError('Changed LVIS taxonomy')
    positive=set()
    for kind,a in stream(path):
        if kind != 'annotations':continue
        if a['id'] in ids:raise ValueError('Duplicate LVIS annotation')
        ids.add(a['id'])
        if a['image_id'] not in images or a['category_id'] not in cats:raise ValueError('Dangling annotation')
        if a['category_id'] not in LVIS:continue
        im=images[a['image_id']];box(a['bbox'],im['width'],im['height'])
        if not a.get('segmentation'):raise ValueError('Missing LVIS mask')
        positive.add((a['image_id'],a['category_id']));instance_counts[a['category_id']]+=1
        if a['image_id'] in matches:matches[a['image_id']]['annotations'].append(a)
    for _,cid in positive:counts[cid]+=1
    scopes=C.Counter()
    for iid,im in images.items():
        for cid in LVIS:scopes[lvis_scope(im,cid,(iid,cid) in positive)]+=1
    return dict(source_sha256=sha(path),categories=[dict(id=k,category=v,images=counts[k],instances=instance_counts[k]) for k,v in LVIS.items()],
                matching_existing_coco_images=[dict(coco_id=i,partition=r['source']['proposed_split'],
                eligibility=r['source'].get('status','not_reassessed'),annotations=r['annotations'],
                federated_image_metadata=images[i],reuse_authorized=False) for i,r in sorted(matches.items())],
                scope_counts=dict(scopes),new_photograph_count=0,
                warning='Metadata only. Preserve all native masks/IDs; do not train or score existing holdouts. Existing COCO boxes are not overwritten.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lvis-json',type=Path);parser.add_argument('--oi-train-csv',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={'lvis':category_summary()}
    if args.lvis_json:
        rows=read(ROOT/'data/coco/image-pilots/phase6d-v1/screening-v2.json')['images']
        result['lvis_instance_join']=lvis_local(args.lvis_json,rows)
    if args.oi_train_csv:result['openimages_training']=oi_counts(args.oi_train_csv,'train')
    else:result['openimages_training']={'available':False,'per_category_images':None,'per_category_boxes':None,'reason':'Full training CSV not acquired; never infer training counts from validation/test'}
    write_new(args.output,result)

if __name__=='__main__':main()
