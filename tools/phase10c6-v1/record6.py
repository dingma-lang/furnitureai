"""Append explicit assistant visual observations after viewing a private plate.

This is not a human-review endpoint and cannot set human confirmation flags.
"""
import argparse
from datetime import datetime,timezone
from common6 import *

def record(observations):
    index={r['candidate_id']:r for r in candidates()}
    folder=RAW/'observations';folder.mkdir(exist_ok=True)
    for ident,(decision,note) in observations.items():
        r=index[ident];num=int(ident[1:]);plate_folder='inspection' if num<=187 else 'inspection-expansion';pn=(num-1)//6+1 if num<=187 else (num-188)//6+33
        plate=RAW/plate_folder/('plate-'+str(pn).zfill(2)+'.jpg')
        if decision not in ('clear_nonchair','automatic_only','mixed_or_uncertain','unreliable_resolution'):raise ValueError('Unknown automatic visual outcome')
        save(folder/(ident+'.json'),dict(candidate_id=ident,image_id=r['image_id'],sha256=r['sha256'],annotation_key=r['annotation']['annotation_key'],
            candidate_sha256=digest(r),visual_inspected=True,method='assistant directly viewed original-frame/native-crop/inner-region plate with view_image',
            plate=str(plate.relative_to(ROOT)),plate_sha256=sha(plate),visual_outcome=decision,visual_observation=note,
            observed_utc=datetime.now(timezone.utc).isoformat(),human_reviewed=False,human_confirmed=False,model_inference_used=False))
    print('RECORDED',len(observations),'assistant visual observations;0human approvals')
