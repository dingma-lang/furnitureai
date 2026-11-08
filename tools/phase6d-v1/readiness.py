"""Phase6E source-pool assessment only: no splits, model loads or updates."""
import collections as C
import csv
import sys
from common import *

def counts(rows):
 result={}
 for label in ('sofa','chair','table','dining_table','bed','desk','cabinet','lamp'):
  result[label]=dict(images=sum(any(a['label']==label for a in r['usable_annotations']) for r in rows),boxes=sum(a['label']==label for r in rows for a in r['usable_annotations']))
 return result

def run():
 old=read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json');fresh=read(ROOT/'data/open-images/image-pilots/phase6a-v1/screening-v3.json')['images'];coco=read(OUT/'eligible-candidates-v1.json')['images']
 # Explicitly diagnose legacy scope without modifying any old annotation.
 sys.path.insert(0,str(ROOT/'tools/phase6a-v1'))
 from annotation_scope import descendant_parents,conflicts
 meta=ROOT/'data/open-images/metadata/phase5-v1';parents=descendant_parents(read(meta/'bbox_labels_600_hierarchy.json'));old_train=[r for r in old['rows'] if r['split']=='train'];ids={r['image_id'] for r in old_train};raw=C.defaultdict(list)
 with (meta/'validation-annotations-bbox.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   if r['ImageID'] in ids:raw[r['ImageID']].append(r)
 legacy=[]
 for r in old_train:
  extra=conflicts(raw[r['image_id']],r['annotations'],parents);mask={'desk','cabinet'}|{x['affected_parent'] for x in extra}
  usable=[a for a in r['annotations'] if not a['ignore'] and r['coverage'][a['label']] and a['label'] not in mask]
  legacy.append(dict(image_id=r['image_id'],source='legacy_openimages',split='train',original_source_group=r['identity_group'],masked_categories=sorted(mask),usable_annotations=usable,retained_after_conservative_positive_scope=bool(usable)))
 fresh_rows=[]
 for r in fresh:
  if r['status']!='provisional_automated_candidate':continue
  usable=[a for a in r['annotations'] if not a['ignore'] and r['coverage'][a['label']]]
  fresh_rows.append(dict(image_id=r['image_id'],source='phase6a_openimages',split=r['proposed_split'],original_source_group=r['identity_group'],usable_annotations=usable))
 coco_rows=[]
 for r in coco:
  # Crowd regions remain ignored, and all unknown classes/regions remain unknown.
  usable=[dict(annotation_id=a['id'],label=CLASSES[a['category_id']],original_annotation=a) for a in r['original_annotations'] if not a['iscrowd'] and a['id'] in r['annotation_diagnostics']['valid_noncrowd_annotation_ids']]
  coco_rows.append(dict(image_id=r['image_id'],source='phase6d_coco',split=r['proposed_split'],original_source_group=r['identity_group'],usable_annotations=usable))
 conservative=[r for r in legacy if r['retained_after_conservative_positive_scope']]+fresh_rows+coco_rows
 train=[r for r in conservative if r['split']=='train'];dev=[r for r in conservative if r['split']=='development']
 actual_counts=dict(legacy_openimages_train=27,phase6a_openimages_train=sum(r['split']=='train' for r in fresh_rows),phase6d_coco_train=sum(r['split']=='train' for r in coco_rows),phase6a_fresh_development=sum(r['split']=='development' for r in fresh_rows),phase6d_fresh_development=sum(r['split']=='development' for r in coco_rows))
 result=dict(schema='phase6e-training-readiness-assessment-v1',training_authorized=False,model_operations_performed=0,available_unchanged_source_pools=actual_counts,
  nominal_combined_train_images=27+actual_counts['phase6a_openimages_train']+actual_counts['phase6d_coco_train'],fresh_development_images=len(dev),historical_development_images_reserved_from_new_tuning=18,original_openimages_frozen_eligible_images_unused=18,
  conservative_positive_scope_train_images=len(train),conservative_positive_scope_legacy_train=sum(r['retained_after_conservative_positive_scope'] for r in legacy),conservative_positive_scope_train_known_groups=len({(r['source'],r['original_source_group']) for r in train}),
  conservative_train_counts=counts(train),fresh_development_counts=counts(dev),source_rows=conservative,legacy_scope_audit=legacy,
  scope='Metadata-only prospective positive-target accounting, not a new training lock or source annotation edit. Broad Open Images Table and COCO dining_table are deliberately separate strata.',
  observed_table_taxonomy_caveat='Actual COCO category67 regions include coffee-table-like/living-room surfaces. The source name dining_table is preserved, but fine-grained subtype and complete broad-Table coverage are not independently established; no relabeling or absence-as-negative inference.',
  independent_group_caveat='Source groups are metadata-known families, not proven physical product independence. Cross-dataset isolation is supported by the preserved Phase6C full creator/source graph, not by comparing unrelated group string namespaces.',
  readiness='Conditionally suitable for a small, explicitly provisional four-class source-agreement head-only experiment after new adapter/config approval; not ready for broad seven-class or independent real-world accuracy claims.',
  blockers=['No Phase6E training approval','Existing Phase5C worker is fixed at27 images; do not use unchanged on expanded rows','Implement/test source-specific partial-annotation loss before training; COCO class absence is not a negative image label','Mask Desk/Cabinetry, descendant-only or conflicting scope; keep dining-table subtype separate from broad Table in evaluation','Fresh development category support is small, especially Bed; no repeated optimization against historical or frozen panels','Source box/mask boundaries are not independently human verified; no validated detection accuracy claimed'],
  future_configuration_proposal=dict(approved=False,adaptation='class_head and box_head only; backbone/text/objectness frozen',batch_size=1,gradient_accumulation=4,epochs_max=10,per_fit_update_formula='ceil(approved final trainingN/4)*10, capped by a separately approved total budget',learning_rates_for_review=[1e-5,3e-6],seed=20261105,memory_fraction=.8,precision='FP16 only after smoke/gradient checks',vram_basis='Same-resolution Phase5C head-only measured about1.93GiB allocated; projection only, no new measurement',baselines=['Immutable original OWLv2','Immutable Phase5C selected detector'],selection='Fresh development only with identical source-native protocols; report per-category AP50/AP75/P/R/localization/duplicate errors and weak-label uncertainty',historical_or_frozen_scoring=False))
 commit(OUT/'phase6e-training-readiness.json',result)
 print('READINESS nominal train',result['nominal_combined_train_images'],'conservative positive-scope train',len(train),'freshdev',len(dev),'; training NOT authorized',flush=True)

if __name__=='__main__':run()
