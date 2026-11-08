"""No-download/no-inference integration check and prospective update schedule."""
from core import *
from training import update_schedule

def run():
 m=read(MANIFEST);scope={r['image_id']:set(r['optimization_classes_to_mask']) for r in read(OUT/'annotation-scope-audit.json')['records']}
 isolation(m['images'],set(read(OUT/'protected-identities.json')))
 rows=[dict(image_id=r['image_id'],split='train') for r in m['images'] if r['proposed_split']=='train' and set(r['eligible_box_counts'])-scope[r['image_id']]]
 schedules=update_schedule(rows)
 write_new(OUT/'dry-run.json',dict(status='PASS',photograph_gets=0,model_loads=0,optimizer_calls=0,metadata_only_potential_train_images=len(rows),
  potential_ten_epoch_updates=len(schedules),schedule=schedules,note='Synthetic execution plan over metadata IDs only, not a final dataset or approved training queue; image screening can only reduce eligibility. No old train images or frozen material were read.'))
 print('NO-DOWNLOAD DRY RUN PASS',len(rows),'potential new train photos;',len(schedules),'prospective updates; actual updates 0')

if __name__=='__main__':run()
