"""Model-free image integrity/quality/duplicate and annotation diagnostics.

No human annotation, negative-label invention, partition reassignment or model scoring.
"""
import collections as C
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from network import *
sys.path.insert(0,str(ROOT/'tools/phase5-v1'))
import detection_prep as d
META=ROOT/'data/open-images/metadata/phase5-v1'
CLASSES=d.CLASSES


def signatures(image):
    rgb=ImageOps.exif_transpose(image).convert('RGB')
    gray=np.asarray(rgb.convert('L'),dtype=float)
    small=np.asarray(rgb.convert('L').resize((9,8)),dtype=float)
    bits=(small[:,1:]>small[:,:-1]).ravel();dh=sum(int(bit)<<i for i,bit in enumerate(bits))
    a=np.asarray(rgb.convert('L').resize((32,32)),dtype=float)
    cos=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64)
    coeff=(cos@a@cos.T).ravel()[1:];ph=sum(int(bit)<<i for i,bit in enumerate(coeff>np.median(coeff)))
    lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
    return dict(decoded_pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
                raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),dhash64=f'{dh:016x}',phash63=f'{ph:016x}',gray_std=float(gray.std()),laplacian_variance=float(lap.var()),
                white_fraction=float(np.mean(np.all(np.asarray(rgb)>245,axis=2))))


def pair_diagnostics(rows):
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            exact=a['sha256']==b['sha256'];pixel=a['decoded_pixel_sha256']==b['decoded_pixel_sha256']
            dh=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count();ph=(int(a['phash63'],16)^int(b['phash63'],16)).bit_count()
            if exact or pixel or dh<=4 or ph<=6:
                pairs.append(dict(a=a['image_id'],b=b['image_id'],cross_partition=a['proposed_split']!=b['proposed_split'],exact_file=exact,exact_pixels=pixel,dhash_distance=dh,phash_distance=ph,
                                  interpretation='duplicate' if exact or pixel else 'unverified perceptual similarity; not proof of shared physical identity'))
    return pairs


def source_annotations(ids):
    result=C.defaultdict(list);labels=C.defaultdict(dict)
    for part in ('validation','test'):
        with (META/(part+'-annotations-bbox.csv')).open(newline='') as f:
            for r in csv.DictReader(f):
                if r['ImageID'] in ids:result[r['ImageID']].append(r)
        with (META/(part+'-annotations-human-imagelabels-boxable.csv')).open(newline='') as f:
            for r in csv.DictReader(f):
                if r['ImageID'] in ids:labels[r['ImageID']][r['LabelName']]=int(r['Confidence'])
    return result,labels


def annotation_diagnostics(row,all_annotations):
    mapping=d.class_mapping(META/'oidv7-class-descriptions-boxable.csv')['mapping'];related={}
    tree=json.loads((META/'bbox_labels_600_hierarchy.json').read_text())
    def walk(node,ancestors):
        mid=node['LabelName']
        if mid not in mapping:
            parents=[mapping[a] for a in ancestors if a in mapping]
            if parents:related[mid]=parents
        for child in node.get('Subcategory',[]):walk(child,ancestors+[mid])
    walk(tree,[])
    descendants=[]
    for raw in all_annotations:
        if raw['LabelName'] not in related:continue
        bbox=list(map(float,[raw['XMin'],raw['YMin'],raw['XMax'],raw['YMax']]))
        for parent in related[raw['LabelName']]:
            covered=any(g['label']==parent and d.iou(g['box'],bbox)>=.9 for g in row['annotations'])
            descendants.append(dict(source_mid=raw['LabelName'],parent_category=parent,original_box=bbox,native_parent_box_present=covered))
    overlap=[]
    for i,a in enumerate(row['annotations']):
        for b in row['annotations'][i+1:]:
            v=d.iou(a['box'],b['box'])
            if a['label']!=b['label'] and v>=.8:overlap.append(dict(a=a['source_mid'],b=b['source_mid'],iou=v,interpretation='overlap/semantic ambiguity, not a confirmed annotation error'))
    return dict(source_boxes=len(all_annotations),mapped_boxes=len(row['annotations']),ordinary_boxes=sum(not g['ignore'] for g in row['annotations']),
                occluded_mapped_boxes=sum(g['flags']['IsOccluded']==1 for g in row['annotations']),truncated_mapped_boxes=sum(g['flags']['IsTruncated']==1 for g in row['annotations']),
                ignored_mapped_boxes=sum(g['ignore'] for g in row['annotations']),unmapped_descendant_regions=descendants,
                hierarchy_coverage_unresolved=any(not g['native_parent_box_present'] for g in descendants),high_cross_category_overlap=overlap,
                unannotated_classes_are_not_negatives=True,whole_object_completeness='not established from box flags alone')


def make_contacts(rows,out):
    out.mkdir();links=[]
    for part in ('train','development','frozen_evaluation'):
        group=[r for r in rows if r.get('downloaded') and r['proposed_split']==part]
        for start in range(0,len(group),20):
            batch=group[start:start+20];canvas=Image.new('RGB',(1280,260*math.ceil(len(batch)/4)+40),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),part+' | original published boxes; assistant diagnostics only',fill='black')
            for i,r in enumerate(batch):
                with Image.open(DEST/r['file']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                scale=min(310/im.width,210/im.height);thumb=im.resize((round(im.width*scale),round(im.height*scale)))
                x=(i%4)*320;y=40+(i//4)*260;canvas.paste(thumb,(x,y));color={'sofa':'red','chair':'blue','table':'green','bed':'purple','desk':'orange','cabinet':'cyan','lamp':'magenta'}
                for g in r['original_annotations']:
                    b=g['box'];draw.rectangle([x+b[0]*thumb.width,y+b[1]*thumb.height,x+b[2]*thumb.width,y+b[3]*thumb.height],outline=color[g['label']],width=2)
                draw.text((x+2,y+212),r['image_id'],fill='black');draw.text((x+2,y+228),r['eligibility_status'],fill='black')
                draw.text((x+2,y+244),','.join(r['native_box_counts']),fill='black')
            name=part+'-'+str(start//20)+'.jpg';canvas.save(out/name,quality=90);links.append(name)
    return links


def run():
    m=load_manifest();out=DEST/'screening-v1'
    if out.exists():raise FileExistsError('Fresh screening output required')
    acquisition=json.loads((DEST/'acquisition-summary.json').read_text());downloaded={r['image_id']:r for r in acquisition['records']}
    licence={r['image_id']:r for r in json.loads((DEST/'licensing-audit.json').read_text())['records']}
    all_boxes,labels=source_annotations({r['image_id'] for r in m['images']})
    protected=d.reserved_identities();rows=[];image_rows=[]
    for original in m['images']:
        ident=original['image_id'];rec=dict(image_id=ident,proposed_split=original['proposed_split'],identity_group=original['identity_group'],
               source_identity=original['source_identity'],original_annotations=copy.deepcopy(original['annotations']),all_original_source_rows=all_boxes[ident],native_image_level_labels=labels[ident],
               native_box_counts=original['native_box_counts'],licensing_status=licence[ident]['status'],flags=[],human_reviewed=False,scene_verified=False,
               annotation_diagnostics=annotation_diagnostics(original,all_boxes[ident]),downloaded=ident in downloaded)
        if ident not in downloaded:
            rec.update(eligibility_status='quarantined_licensing_not_acquired',reason=licence[ident]['reason']);rows.append(rec);continue
        source=downloaded[ident];p=DEST/source['file']
        if sha(p)!=source['sha256']:raise ValueError('Acquired photo integrity changed')
        with Image.open(p) as im:sig=signatures(im)
        rec.update(source,**sig,licensing_status=licence[ident]['status'])
        if any(sig[k] in protected for k in ('decoded_pixel_sha256','raw_rgb_sha256')) or rec['sha256'] in protected:rec['flags'].append('protected_dataset_exact_overlap')
        if source['exif_orientation']!=1:rec['flags'].append('orientation_uncertain')
        if min(source['width'],source['height'])<200:rec['flags'].append('small_image')
        if sig['gray_std']<3:rec['flags'].append('blank_low_contrast')
        if sig['laplacian_variance']<15:rec['flags'].append('blur_or_low_detail_heuristic')
        if sig['white_fraction']>.8:rec['flags'].append('mostly_white_or_studio_risk')
        if rec['annotation_diagnostics']['hierarchy_coverage_unresolved']:rec['flags'].append('native_hierarchy_coverage_unresolved')
        if rec['annotation_diagnostics']['high_cross_category_overlap']:rec['flags'].append('cross_category_overlap_review')
        rec['eligibility_status']='provisional_candidate' if not rec['flags'] else 'unresolved_screening'
        if any(f in rec['flags'] for f in ('protected_dataset_exact_overlap','orientation_uncertain','blank_low_contrast')):rec['eligibility_status']='quarantined_screening'
        rows.append(rec);image_rows.append(rec)
    pairs=pair_diagnostics(image_rows)
    for pair in pairs:
        for rec in image_rows:
            if rec['image_id'] not in (pair['a'],pair['b']):continue
            if pair['exact_file'] or pair['exact_pixels'] or pair['cross_partition']:
                rec['flags'].append('duplicate_or_cross_partition_similarity');rec['eligibility_status']='quarantined_identity'
            else:
                rec['flags'].append('perceptual_similarity_review')
                if rec['eligibility_status']=='provisional_candidate':rec['eligibility_status']='unresolved_screening'
    summary={}
    for part in ('train','development','frozen_evaluation'):
        subset=[r for r in rows if r['proposed_split']==part]
        summary[part]=dict(approved_membership=len(subset),downloaded=sum(r['downloaded'] for r in subset),statuses=dict(C.Counter(r['eligibility_status'] for r in subset)),
            categories={c:{status:dict(images=sum(r['native_box_counts'].get(c,0)>0 for r in subset if r['eligibility_status']==status),native_boxes=sum(r['native_box_counts'].get(c,0) for r in subset if r['eligibility_status']==status),ordinary_boxes=sum(sum(g['label']==c and not g['ignore'] for g in r['original_annotations']) for r in subset if r['eligibility_status']==status)) for status in sorted({r['eligibility_status'] for r in subset})} for c in CLASSES})
    out.mkdir();atomic(out/'screening.json',dict(manifest_sha256=PIN,images=rows,summary=summary,duplicate_pairs=pairs,scene_screening='Pixel heuristics and published metadata; targeted assistant contact-sheet observations recorded separately; no human ground truth',partitions_reassigned=False))
    contacts=make_contacts(rows,out/'contacts')
    parts=['<!doctype html><meta charset="utf-8"><title>Phase 5 original photographs and published annotations</title><h1>Provisional diagnostics, no model predictions</h1><p>Original boxes and class IDs preserved. Human review fields remain false. Images stay in approved partitions.</p>']
    for r in image_rows:
        parts.append('<h2>'+html.escape(r['image_id']+' | '+r['proposed_split']+' | '+r['eligibility_status'])+'</h2>')
        rects=''.join(f'<rect x="{g["box"][0]*r["width"]}" y="{g["box"][1]*r["height"]}" width="{(g["box"][2]-g["box"][0])*r["width"]}" height="{(g["box"][3]-g["box"][1])*r["height"]}" stroke="red" fill="none" stroke-width="2"><title>{g["label"]} {g["source_mid"]}</title></rect>' for g in r['original_annotations'])
        parts.append(f'<svg width="800" viewBox="0 0 {r["width"]} {r["height"]}"><image width="{r["width"]}" height="{r["height"]}" href="../{r["file"]}"/>{rects}</svg><p>'+html.escape(', '.join(r['flags']))+'</p>')
    with (out/'gallery.html').open('x') as f:f.write(''.join(parts))
    atomic(out/'contact-index.json',dict(files=contacts))
    print(json.dumps(dict(summary={k:{a:b for a,b in v.items() if a!='categories'} for k,v in summary.items()},duplicate_pairs=len(pairs)),indent=2))

if __name__=='__main__':run()
