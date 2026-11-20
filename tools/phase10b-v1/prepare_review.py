"""All130 inactive boxes: exact native overlays, crops, masks and a local review queue."""
import collections
import hashlib
from PIL import Image,ImageDraw
from p10 import *
from audit import SOURCES
from policy import normalized_box

def run():
    if RAW.exists():raise FileExistsError('Fresh preparation destination required')
    RAW.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    old=read(OLD/'prepared-manifest.json');manifest=dict(old,schema='phase10b-eligibility-v1',
        source_phase10a_sha256=sha(OLD/'prepared-manifest.json'),annotation_recovery_activated=0,
        training_authorized=False,unreviewed_overlay_annotations_still_inactive=33)
    save(RAW/'eligibility-manifest.json',manifest)
    src=read(ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json')['source_rows']
    lookup={r['stable_audit_annotation_id']:(s,r) for s in src if s['split']=='train' for r in s['resolved_targets']}
    caches={k:{r['image_id']:r for r in read(ROOT/path)[field]} for k,(_,path,field) in SOURCES.items()}
    inactive=[r for r in read(OLD/'annotation-dispositions.json')['records'] if not r['prepared_positive']]
    queue=[];tiles=[];folder=RAW/'review';folder.mkdir();(folder/'views').mkdir()
    for index,r in enumerate(inactive):
        source,t=lookup[r['annotation_id']];original=caches[source['source']][source['image_id']]
        p=ROOT/SOURCES[source['source']][0]/original['file']
        if sha(p)!=source['sha256']:raise ValueError('Source photograph changed')
        w,h=original['width'],original['height'];b=normalized_box(t['native_dataset'],t['original_annotation'],w,h)
        token=hashlib.sha256(r['annotation_id'].encode()).hexdigest()[:20]
        with Image.open(p) as im: im=im.convert('RGB')
        overlay=im.copy();draw=ImageDraw.Draw(overlay,'RGBA')
        segmentation=t['original_annotation'].get('segmentation',[])
        polygon_mask=isinstance(segmentation,list) and bool(segmentation)
        if polygon_mask:
            for polygon in segmentation:
                if len(polygon)>=6:draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=(30,230,80,65),outline=(30,230,80,220))
        xy=[b[0]*w,b[1]*h,b[2]*w,b[3]*h];draw.rectangle(xy,outline=(255,180,0,255),width=max(2,round(w/200)))
        context=overlay.copy();context.thumbnail((700,480))
        pad=max(8,int(max(xy[2]-xy[0],xy[3]-xy[1])*.15));region=(max(0,int(xy[0])-pad),max(0,int(xy[1])-pad),min(w,int(xy[2])+pad),min(h,int(xy[3])+pad))
        crop=overlay.crop(region);crop.thumbnail((600,480))
        canvas=Image.new('RGB',(1320,560),'white');d=ImageDraw.Draw(canvas);canvas.paste(context,(5,70));canvas.paste(crop,(710,70))
        title=f'{index+1:03d} {r["category"]} | {t["native_dataset"]}:{t["native_id"]} | inactive'
        d.text((8,8),title,fill='black');d.text((8,28),'Yellow: unchanged source box; green: native polygon. Context and crop; no human approval.',fill='black')
        d.text((8,48),'Border/truncation: '+str(r['border_or_truncation'])+' | scope: '+t['native_scope'],fill='black')
        canvas.save(folder/'views'/(token+'.jpg'),quality=90)
        thumb=canvas.copy();thumb.thumbnail((650,280));tiles.append((index+1,thumb))
        queue.append(dict(id=token,index=index+1,image_id=r['image_id'],annotation_id=r['annotation_id'],category=r['category'],
            native_dataset=t['native_dataset'],native_id=t['native_id'],source_photo=str(p.relative_to(ROOT)),source_sha256=source['sha256'],
            native_annotation=t['original_annotation'],normalized_box=b,has_mask=bool(segmentation),polygon_mask_rendered=polygon_mask,
            exclusion_reason=r['reason'],border_or_truncation=r['border_or_truncation'],native_scope=t['native_scope'],
            assistant_triage='pending_visual_inspection',human_reviewed=False,training_active=False))
    (folder/'contacts').mkdir()
    for page,start in enumerate(range(0,len(tiles),12)):
        canvas=Image.new('RGB',(1320,1710),'#ddd')
        for j,(_,thumb) in enumerate(tiles[start:start+12]):canvas.paste(thumb,(j%2*660,j//2*285))
        canvas.save(folder/'contacts'/f'page-{page+1:02d}.jpg',quality=90)
    save(folder/'queue.json',dict(source_dispositions_sha256=sha(OLD/'annotation-dispositions.json'),records=queue,
        count=len(queue),original_annotations_modified=False,missing_human_confirmations_not_filled=True))
    configs=[]
    for p in sorted((OLD/'configs').glob('*.json')):
        c=read(p);c.update(manifest_sha256=sha(RAW/'eligibility-manifest.json'),training_authorized=False,
                         run_destination=str(RUN.relative_to(ROOT)),precision='fp16',
                         expected_optimizer_updates=75)
        save(RAW/'configs'/p.name,c);configs.append(p.name)
    save(RAW/'authorization-proposal.json',dict(approved=False,manifest_sha256=sha(RAW/'eligibility-manifest.json'),
        config_sha256={n:sha(RAW/'configs'/n) for n in configs},training_updates_max=300,epochs_max=5,
        max_seconds=3600,max_allocated_vram_bytes=int(6.5*2**30),scope='two head-only Sofa/Chair/Bed diagnostic fits; no frozen scoring/deployment',
        code_hashes='Seal current implementation before explicit training approval'))
    print('Prepared130 native annotation overlays; eligibility unchanged78photos243boxes; zero targets reactivated',flush=True)

if __name__=='__main__':run()
