"""Automatic evidence is separate from human review and production defaults."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c5-v1'))
from g5_common import read,save,sha,digest,manifest,optimizer_member,CLASSES,TAXONOMY,overlap
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c6-v1'
REPORT=ROOT/'reports/phase10c6-v1'
POLICY=dict(version='source-visible-instance-mask-chair-negative-v1',target='chair',
    source='COCO/LVIS native non-Chair visible instance polygon plus actual assistant visual inspection',
    spatial='One simple source polygon; fully interior rectangle; at least2px exact boundary clearance; zero protected/uncertain overlap',
    scope='Visible pixels of the named object only; no claim about hidden objects or full photograph completeness',
    minimum_side_pixels=12,minimum_area_pixels=256,mask_inward_margin_pixels=3,
    reviewed_by='assistant_visual_inspection',human_reviewed=False,
    missing_labels_are_not_negatives=True,other_categories_unknown=True,
    independent_from_detector_predictions=True,training_enabled=False)
ADMISSION=dict(POLICY,minimum_inspected_region_side_pixels=32,
    mixed_visual_candidate_boxes_protected=True,calibrated_probability_available=False)

def candidates():
    p=RAW/'all-candidates.json'
    return read(p if p.exists() else RAW/'candidates.json')['candidates']

def registry():
    import g5_review
    return g5_review.registry()
