"""Read saved Phase 4 crops; produce a fresh, offline diagnostic run only."""
import argparse
import collections
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import detection_prep as d


def run(source, out):
    if out.exists(): raise FileExistsError('Fresh output directory required')
    # Hash all inputs, weights and completed historical artifacts; no holdout photo reads.
    guarded = {}
    for base in (d.ROOT/'models', source, d.ROOT/'data/abo/experiments/phase3-gpu-v1'):
        for p in base.rglob('*'):
            if p.is_file(): guarded[str(p.relative_to(d.ROOT))] = d.sha(p)
    out.mkdir(parents=True)
    d.write_new(out/'preservation-before.json',guarded)
    data=d.read(source/'results.json'); rooms=[]; pages=[]
    for room in data['images']:
        p=Path(room['file']).resolve()
        if p.is_relative_to(d.ROOT/'data/abo') or d.sha(p)!=room['sha256']: raise ValueError('Unexpected/changed room source')
        with Image.open(p) as im: image=ImageOps.exif_transpose(im).convert('RGB')
        rows=[]
        for r in room['crops']:
            cp=(source/r['crop_file']).resolve()
            if not cp.is_relative_to(source.resolve()) or d.sha(cp)!=r['crop_sha256']: raise ValueError('Unsafe or changed saved crop')
            with Image.open(cp) as ci:
                ci.load(); q=d.quality(r,room['crops'],image.size,ci)
            q.update(crop_sha256=r['crop_sha256'], crop_file=r['crop_file'],
                     detector_label=r['detector_label'], baseline_label=r['baseline']['category'],
                     expanded_label=r['expanded']['category'])
            rows.append(q)
        # Same boxes in both panels; right panel abstains on flagged crops.
        scale=800/image.width; h=round(image.height*scale)
        canvas=Image.new('RGB',(1600,h+54),'white'); draw=ImageDraw.Draw(canvas)
        for col in range(2):
            canvas.paste(image.resize((800,h)),(col*800,54))
            draw.text((col*800+8,8), 'Original saved OWLv2 boxes' if not col else 'Diagnostic gate: orange = unresolved; green = unverified candidate',fill='black')
            for q in rows:
                coords=[q['original_box'][0]*scale+col*800,q['original_box'][1]*scale+54,q['original_box'][2]*scale+col*800,q['original_box'][3]*scale+54]
                color='#1166dd' if not col else ('#1a9944' if q['status']=='candidate_unverified' else '#e57500')
                draw.rectangle(coords,outline=color,width=2)
                text=q['detection_id']+' '+(q['detector_label'] if not col else (q['advisory_category'] or 'unresolved'))
                draw.text((coords[0],coords[1]),text,fill='white',stroke_fill='black',stroke_width=1)
        visual=room['folder']+'.png'; canvas.save(out/visual)
        rooms.append(dict(image_id=room['image_id'],source_sha256=room['sha256'],rows=rows,visual=visual))
        pages.append('<h2>'+html.escape(room['image_id'])+'</h2><img width="1200" src="'+visual+'">')
        for q in rows:
            # Relative link to existing photograph crop; no copies or historical modifications.
            import os
            href=os.path.relpath(source/q['crop_file'],out)
            pages.append('<figure style="display:inline-block;width:280px;vertical-align:top"><img style="max-width:260px;max-height:200px" src="'+html.escape(href)+'"><figcaption>'+html.escape(q['detection_id']+' | '+q['status']+' | '+', '.join(q['heuristic_flags']))+'</figcaption></figure>')
    flat=[r for room in rooms for r in room['rows']]
    summary=dict(rooms=len(rooms),crops=len(flat),statuses=dict(collections.Counter(r['status'] for r in flat)),
                 heuristic_flags=dict(collections.Counter(f for r in flat for f in r['heuristic_flags'])),
                 classifier_disagreements=sum(r['measurements']['classifiers_disagree'] for r in flat),
                 expanded_vs_detector=sum(r['expanded_label']!=r['detector_label'] for r in flat),
                 ground_truth_metrics=None,production_changes=False,thresholds=d.THRESHOLDS,
                 threshold_policy='Unfitted diagnostic defaults; not calibrated from these rooms or holdouts',
                 true_object_completeness='Unknown; geometry and appearance cannot establish whole-object coverage')
    d.write_new(out/'diagnostics.json',dict(summary=summary,images=rooms))
    with (out/'gallery.html').open('x') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>Phase 5 diagnostic crops</title><h1>Heuristic crop suitability; no ground truth</h1><p>Candidate does not mean correct. Flags do not prove an error. Original predictions preserved.</p>'+''.join(pages))
    bad=[p for p,h in guarded.items() if not (d.ROOT/p).is_file() or d.sha(d.ROOT/p)!=h]
    d.write_new(out/'preservation-after.json',dict(checked=len(guarded),mismatches=bad,official_phase2b_images_opened=False))
    if bad: raise ValueError('Historical preservation mismatch')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=d.ROOT/'reports/phase4-room-comparison-v1'); ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();run(a.source,a.out)
