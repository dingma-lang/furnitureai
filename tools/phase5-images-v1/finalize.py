"""Apply explicit assistant diagnostics without altering source images, boxes or membership."""
import collections as C
import copy
import csv
import html
import json
from pathlib import Path
from network import *
from screen import CLASSES
from licensing import LD


def counts(rows):
    out={}
    for part in ('train','development','frozen_evaluation'):
        selected=[r for r in rows if r['proposed_split']==part]
        out[part]=dict(approved_membership=len(selected),downloaded=sum(r['downloaded'] for r in selected),statuses=dict(C.Counter(r['eligibility_status'] for r in selected)),
          categories={c:{status:dict(images=sum(r['native_box_counts'].get(c,0)>0 for r in selected if r['eligibility_status']==status),
                  native_boxes=sum(r['native_box_counts'].get(c,0) for r in selected if r['eligibility_status']==status),
                  ordinary_boxes=sum(sum(g['label']==c and not g['ignore'] for g in r['original_annotations']) for r in selected if r['eligibility_status']==status))
              for status in sorted({r['eligibility_status'] for r in selected})} for c in CLASSES})
    return out


def run():
    out=DEST/'screening-final-v1'
    if out.exists():raise FileExistsError('Fresh final diagnostic output required')
    initial=json.loads((DEST/'screening-v1/screening.json').read_text());rows=copy.deepcopy(initial['images']);obs=json.loads((DEST/'assistant-visual-observations.json').read_text());lookup={r['image_id']:r for r in rows}
    frozen=load_manifest();approved={r['image_id']:r for r in frozen['images']}
    for evidence in obs['observations']:
        row=lookup[evidence['image_id']]
        if not row['downloaded']:raise ValueError('Visual finding references unacquired photo')
        row['assistant_visual_observations']=row.get('assistant_visual_observations',[])+[evidence]
        row['flags'].append('assistant_scene_or_annotation_risk')
        if evidence['action']=='quarantine_scene':row['eligibility_status']='quarantined_scene'
        elif evidence['action']=='unresolved' and row['eligibility_status']=='provisional_candidate':row['eligibility_status']='unresolved_screening'
        elif evidence['action']!='unresolved':raise ValueError('Unknown assistant diagnostic action')
    for row in rows:
        original=approved[row['image_id']]
        if row['original_annotations']!=original['annotations'] or row['proposed_split']!=original['proposed_split']:raise ValueError('Original annotation/membership changed')
        if row['downloaded'] and sha(DEST/row['file'])!=row['sha256']:raise ValueError('Original image changed')
        if row['human_reviewed'] or row['scene_verified']:raise ValueError('Assistant diagnostic must not fabricate verification')
    summary=counts(rows);out.mkdir()
    atomic(out/'screening.json',dict(manifest_sha256=PIN,approved_membership=150,images=rows,summary=summary,duplicate_pairs=initial['duplicate_pairs'],partition_assignments_changed=False,
        status_policy='provisional_candidate means no flagged risk in this bounded screening, not independently verified quality/ground truth. Unresolved and quarantined examples remain excluded from automatic model input. Source partition tags and all boxes stay unchanged.',
        human_reviewed_examples=0,model_inference=False,training=False))
    fields=['image_id','proposed_split','downloaded','eligibility_status','licensing_status','file','sha256','flags']
    with (out/'eligibility.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:','.join(r['flags']) if k=='flags' else r.get(k,'') for k in fields})
    with (out/'category-counts.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['partition','category','status','images','native_boxes','ordinary_boxes']);writer.writeheader()
        for part,s in summary.items():
            for c,stats in s['categories'].items():
                for status,values in stats.items():writer.writerow(dict(partition=part,category=c,status=status,**values))
    parts=['<!doctype html><meta charset="utf-8"><title>Final Phase 5 screening</title><h1>Final provisional screening</h1><p>No model predictions or human confirmation. Original Open Images boxes and IDs retained. All 150 approved partition slots retained; unacquired and quarantined examples remain excluded.</p>']
    for r in rows:
        parts.append('<h2>'+html.escape(r['image_id']+' | '+r['proposed_split']+' | '+r['eligibility_status'])+'</h2>')
        if not r['downloaded']:
            parts.append('<p>'+html.escape(r['reason'])+'</p>');continue
        rects=''.join(f'<rect x="{g["box"][0]*r["width"]}" y="{g["box"][1]*r["height"]}" width="{(g["box"][2]-g["box"][0])*r["width"]}" height="{(g["box"][3]-g["box"][1])*r["height"]}" stroke="red" fill="none" stroke-width="2"><title>{g["label"]} {g["source_mid"]}</title></rect>' for g in r['original_annotations'])
        parts.append(f'<svg width="800" viewBox="0 0 {r["width"]} {r["height"]}"><image width="{r["width"]}" height="{r["height"]}" href="../{r["file"]}"/>{rects}</svg><p>'+html.escape(', '.join(r['flags']))+'</p>')
        for e in r.get('assistant_visual_observations',[]):parts.append('<p>Assistant diagnostic: '+html.escape(e['evidence'])+'</p>')
    with (out/'gallery.html').open('x') as f:f.write(''.join(parts))
    # Attribution for acquired photographs, including those later quarantined.
    rights={r['image_id']:r for r in json.loads((DEST/'licensing-audit.json').read_text())['records']}
    with (DEST/'ATTRIBUTION.md').open('x') as f:
        f.write('# Photograph attribution\n\nIndividual source-page evidence is preserved under licensing/. Published Open Images annotations: Google LLC, CC BY 4.0; cite Kuznetsova et al., IJCV 2020. Photographs retain their individual CC BY 2.0 licence. CVDF supplied resized JPEGs; originals were not edited. Contact sheets resize and overlay original published boxes; no endorsement implied.\n\n')
        for r in rows:
            if not r['downloaded']:continue
            a=rights[r['image_id']];title=approved[r['image_id']]['title']
            f.write(f"- **{r['image_id']}** — {title}; {a['author_name']}; [individual source]({a['source_url']}); [author]({a['author_url']}); [CC BY 2.0]({a['license_url']}); [CVDF object]({approved[r['image_id']]['image_url']}); SHA-256 `{r['sha256']}`.\n")
    # Detailed provenance for every failed licence, including observed alternate URL.
    ledger=json.loads((DEST/'network-ledger.json').read_text());details=[]
    for r in rows:
        if r['downloaded']:continue
        rec=next(x for x in ledger['requests'] if x['file']=='licensing/'+r['image_id']+'.response')
        item=dict(image_id=r['image_id'],proposed_split=r['proposed_split'],source_url=rec['url'],status=rec['status'],response_sha256=rec['sha256'],response_file=rec['file'],reason=r['reason'],observed_image_licence_urls=[])
        if rec['status']==200:
            import gzip
            data=(DEST/rec['file']).read_bytes()
            if next((v for k,v in rec['headers'].items() if k.lower()=='content-encoding'),'')=='gzip':data=gzip.decompress(data)
            parser=LD();parser.feed(data.decode())
            def visit(x):
                if isinstance(x,dict):
                    if x.get('@type')=='ImageObject':item['observed_image_licence_urls'].append(x.get('license'))
                    for v in x.values():visit(v)
                elif isinstance(x,list):
                    for v in x:visit(v)
            for doc in parser.documents:visit(doc)
        details.append(item)
    atomic(DEST/'licensing-quarantine-details.json',dict(records=details))
    print(json.dumps({part:{k:v for k,v in s.items() if k!='categories'} for part,s in summary.items()},indent=2))

if __name__=='__main__':run()
