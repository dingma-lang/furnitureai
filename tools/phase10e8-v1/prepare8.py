"""Inspect all21saved E5/teacher queries,native polygons and training photo context."""
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common8q import *
from geometry8 import polygons,coverage,admission

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only preparation')
    if DATA.exists() or REPORT.exists():raise FileExistsError('New E8 destinations required')
    sys.path.insert(0,str(ROOT/'tools/phase10e7-v1'));import verify7
    verify7.check()
    rows=exact_candidates();c,train,dev=context()
    native_path=ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json'
    native=read(native_path);photos={p['image_id']:p for p in native['photos']}
    raw={r['image_id']:r for r in read(E7/'raw-complete.json')['records']}
    refs={r['image_id']:r for r in read(PREP/'teacher-index.json')['records']}
    reg={p['image_id']:p for p in read(ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json')['photos']}
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'));from review21 import semantic_protected
    files={str(p.relative_to(ROOT)):sha(p) for p in (E7/'analysis/candidate-manifest.json',E7/'analysis/inspection-samples.json',E7/'analysis/high-confidence-Chair.jsonl',E7/'raw-complete.json',PREP/'training-context.json',PREP/'teacher-index.json',native_path)}
    if sha(BASE/'model.safetensors')!=BASE_SHA:raise ValueError('Original checkpoint changed')
    for source in native['source_files']:
        if sha(ROOT/source['source'])!=source['sha256']:raise ValueError('Official source annotations changed')
        files[source['source']]=source['sha256']
    cache={};polys={};records=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    DATA.mkdir(parents=True);REPORT.mkdir(parents=True)
    for record in rows:
        row=require_train(record['image_id'],train);photo=photos[row['image_id']];ident=row['image_id'];q=record['query_index']
        if any(record[k]!=row[k] for k in ('width','height')) or record['source_sha256']!=row['sha256'] or photo['source_group']!=row['identity_group']:
            raise ValueError('Frozen candidate source/coordinate mismatch')
        if ident not in cache:
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Original image changed')
            files[row['path']]=row['sha256']
            for k,hk in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
                rel=photo['rights'][k]
                if sha(ROOT/rel)!=photo['rights'][hk]:raise ValueError('Rights evidence changed')
                files[rel]=photo['rights'][hk]
            r=raw[ident];t=refs[ident]
            if sha(E7/r['file'])!=r['sha256'] or sha(ROOT/t['cache'])!=t['cache_sha256']:raise ValueError('Saved model-query cache changed')
            files[str((E7/r['file']).relative_to(ROOT))]=r['sha256'];files[t['cache']]=t['cache_sha256']
            with np.load(E7/r['file'],allow_pickle=False) as z:student={k:z[k].copy() for k in ('logits','square_cxcywh','original_xyxy_pixels')}
            with np.load(ROOT/t['cache'],allow_pickle=False) as z:teacher={k:z[k].copy() for k in ('logits','square_cxcywh')}
            teacher['original_xyxy_pixels']=registered(teacher['square_cxcywh'],row['width'],row['height'])*np.asarray([row['width'],row['height']]*2)
            cache[ident]=(student,teacher)
        student,teacher=cache[ident];s=1/(1+np.exp(-student['logits'][q].astype(float)));t=1/(1+np.exp(-teacher['logits'][q].astype(float)))
        if abs(float(s[1])-record['student_Chair_score'])>1e-12 or abs(float(t[1])-record['original_same_query_Chair_score'])>1e-12 or not np.array_equal(student['original_xyxy_pixels'][q],np.asarray(record['box'])*np.asarray([row['width'],row['height']]*2)):
            raise ValueError('Actual query or original-image mapping changed')
        px=student['original_xyxy_pixels'][q].tolist();tp=teacher['original_xyxy_pixels'][q].tolist()
        entry=next(e for e in c['entries'] if e['row']['image_id']==ident)
        protected=[dict(key='active:'+str(i),box=a['box']) for i,a in enumerate(row['annotations'])]
        protected += [dict(key='registry:'+str(i),box=p['box']) for i,p in enumerate(reg[ident]['protected'])]
        protected += [dict(key='native:'+a['annotation_key'],box=a['box']) for a in photo['annotations'] if semantic_protected(a) or a['native_ignored']]
        for k in ('legacy_protected','distill_protected'):protected += [dict(key=k+':'+str(i),box=b) for i,b in enumerate(entry[k])]
        hits=[]
        for a in photo['annotations']:
            if digest(a['original_annotation'])!=a['original_annotation_sha256']:raise ValueError('Native source annotation checksum mismatch')
            if not overlap(record['box'],a['box']):continue
            key=(ident,a['annotation_key'])
            if key not in polys:polys[key]=polygons(a['original_annotation'],row['width'],row['height'])
            candidates=polys[key];fractions=[coverage(p,px) for p in candidates]
            gate=admission(px,row['width'],row['height'],candidates,protected)
            hits.append(dict(annotation_key=a['annotation_key'],dataset=a['dataset'],native_category_id=a['native_category_id'],native_label=a['native_label'],
                native_box=a['box'],source=a['source'],source_sha256=a['source_sha256'],original_annotation_sha256=a['original_annotation_sha256'],
                native_ignored=a['native_ignored'],protected_furniture=semantic_protected(a),
                bbox_query_fraction=intersection(record['box'],a['box'])/area(record['box']),
                maximum_single_polygon_query_fraction=max(fractions,default=0.),polygons=[[[float(x),float(y)] for x,y in p] for p in candidates],
                polygon_query_fractions=fractions,geometric_gate=gate,
                source_geometry_only_not_a_label=True))
        hits.sort(key=lambda a:(-a['maximum_single_polygon_query_fraction'],-a['bbox_query_fraction'],a['annotation_key']))
        r=dict(record,query_pixels=px,original_query_pixels=tp,E5_class_scores={k:float(s[i]) for i,k in enumerate(CLASSES)},
            original_class_scores={k:float(t[i]) for i,k in enumerate(CLASSES)},native_mask_hits=hits,protected_regions=protected,
            rights=photo['rights'],source_completeness=photo['native_source_scopes'],source_photo_id=photo['rights']['source_photo_id'],
            original_annotations=[{k:a[k] for k in ('annotation_key','native_label','native_category_id','dataset','box','has_mask','native_ignored','protect_furniture','part_or_semantics_unresolved','original_annotation_sha256')} for a in photo['annotations']],
            all_six_other_classes_unknown=True,whole_photo_Chair_absence_inferred=False)
        # Show both original context and enlarged actual query pixels. Source
        # masks are rasterized for display only;continuous geometry gates above
        # use original polygons with no raster-based promotion.
        selected=[a for a in hits if a['annotation_key']==record['native_annotation_key']]
        best=hits[:1];drawanns={a['annotation_key']:a for a in selected+best}
        with Image.open(ROOT/row['path']) as im:
            if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image orientation changed')
            image=im.convert('RGB');context_image=image.convert('RGBA');overlay=Image.new('RGBA',image.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
            for a in drawanns.values():
                for poly in a['polygons']:od.polygon([tuple(v) for v in poly],fill=(0,210,170,70),outline=(0,200,150,220))
            context_image=Image.alpha_composite(context_image,overlay).convert('RGB');d=ImageDraw.Draw(context_image)
            for a in photo['annotations']:
                b=a['box'];b=[b[0]*image.width,b[1]*image.height,b[2]*image.width,b[3]*image.height]
                d.rectangle(b,outline='#a500a5' if semantic_protected(a) or a['native_ignored'] else '#aaaaaa',width=1)
            d.rectangle(tp,outline='#176bff',width=3);d.rectangle(px,outline='#ff2020',width=3)
            raw_crop=image.crop((math.floor(px[0]),math.floor(px[1]),math.ceil(px[2]),math.ceil(px[3])))
            mask_crop=context_image.crop((math.floor(px[0]),math.floor(px[1]),math.ceil(px[2]),math.ceil(px[3])))
            output=Image.new('RGB',(1500,720),'white');dd=ImageDraw.Draw(output)
            title=f"{r['sample_id']} | {ident} | q={q} | E5 Chair {s[1]:.4f} | original {t[1]:.4f}"
            dd.text((10,8),title,font=font,fill='black')
            dd.text((10,32),'Red=E5 query;blue=original same-index box;purple=protected/native furniture;teal=original source masks',font=font,fill='black')
            dd.text((10,56),'Left=full photo context;middle=raw query crop enlarged;right=mask-overlay query crop enlarged',font=font,fill='black')
            context_image.thumbnail((720,600));output.paste(context_image,(10,100))
            for crop,x in ((raw_crop,740),(mask_crop,1120)):
                scale=min(355/max(1,crop.width),540/max(1,crop.height));crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.NEAREST);output.paste(crop,(x,105))
            text='bbox source '+str(record['native_label'])+'; best visible mask '+str(hits[0]['native_label'] if hits else None)
            dd.text((10,675),text,font=font,fill='black')
            p=DATA/'dashboard/assets'/f"{r['sample_id']}.jpg";stream=io.BytesIO();output.save(stream,format='JPEG',quality=92);atomic(p,stream.getvalue())
            r['evidence_plate']=str(p.relative_to(ROOT));r['evidence_plate_sha256']=sha(p)
        records.append(r)
        print('CPU E8 prepared',r['sample_id'],'bestmask',hits[0]['native_label'] if hits else None,round(hits[0]['maximum_single_polygon_query_fraction'],3) if hits else None,flush=True)
    emit(DATA/'candidate-source-audit.json',dict(records=records,policy=POLICY,policy_sha256=digest(POLICY),exact_candidates=21,photographs=10,source_groups=10,
         mask_fraction_definition='Maximum continuous intersection with a SINGLE original simple polygon;not summed across overlapping COCO/LVIS masks;does not establish negative class',
         image_rasterization_for_display_only=True,other_six_categories_unknown=True))
    emit(DATA/'input-binding.json',dict(parent=PARENT,files=files,candidate_ids=[r['candidate_id'] for r in rows],source_image_ids=sorted({r['image_id'] for r in rows}),
         policy=POLICY,training_authorized=False,only_optimization_images=True,new_GPU_forwards=0,downloads=0))
    print('All21source/mask/query/context audits prepared;no labels assigned')

if __name__=='__main__':run()
