"""Exhaust remaining local geometry; visually inspect further admissible interiors."""
from common6 import *
from geometry6 import propose

def run():
    inv=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json');base=read(RAW/'candidates.json')['candidates'];seen={r['annotation']['object_hypothesis_group'] for r in base}
    rp={p['image_id']:p for p in registry()['photos'] if p['role']=='optimization'};new=[];excluded=[];scanned=0
    for p in sorted(inv['photos'],key=lambda p:p['image_id']):
        if not p['eligible_for_primary_AB']:continue
        for a in sorted(p['annotations'],key=lambda a:a['annotation_key']):
            if not a['metadata_candidate_eligible'] or a['mapped_auxiliary_category']=='protected_furniture' or a['object_hypothesis_group'] in seen:continue
            scanned+=1;photo=rp[p['image_id']];region=propose(a['original_annotation'],p['width'],p['height'],photo['protected'])
            if region is None or min(region['box_pixels'][2]-region['box_pixels'][0],region['box_pixels'][3]-region['box_pixels'][1])<32:
                excluded.append(dict(image_id=p['image_id'],annotation_key=a['annotation_key'],object_hypothesis_group=a['object_hypothesis_group'],reason='No certified32px source-mask interior under fixed policy',visual_inspected=False));continue
            r=dict(candidate_id='C'+str(len(base)+len(new)+1).zfill(3),cohort='expanded-qualified-local',image_id=p['image_id'],sha256=p['sha256'],path=p['path'],width=p['width'],height=p['height'],identity_group=p['identity_group'],category=a['mapped_auxiliary_category'],native_label=a['native_label'],annotation=a,rights=p['licence'],protected=photo['protected'],proposal=region,visual_inspected=False,human_reviewed=False,decision_level=None)
            new.append(r);seen.add(a['object_hypothesis_group'])
    save(RAW/'qualified-expansion.json',dict(candidates=new,remaining_native_records_geometry_scanned=scanned,excluded_without_new_visual_review=excluded,source_inventory_sha256=sha(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json')))
    save(RAW/'all-candidates.json',dict(candidates=base+new,base_sha256=sha(RAW/'candidates.json'),expansion_sha256=sha(RAW/'qualified-expansion.json')))
    print('Additional geometry scanned',scanned,';new candidates',len(new),';geometry excluded',len(excluded),flush=True)

if __name__=='__main__':run()
