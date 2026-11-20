"""Append-only, typed experiment history; missing historical evidence stays missing."""
from p10 import *

def record(root,event):
    required={'experiment_id','dataset_version','dataset_sha256','git_commit','checkpoint_sha256','settings','metrics','resources','decision','next_experiment','evaluation_scope'}
    if required-set(event):raise ValueError('Incomplete experiment-history event')
    if event['decision'] not in ('retain','investigate','reject'):raise ValueError('Unknown experiment decision')
    if event['evaluation_scope'] not in ('provisional_weak_label_internal','weak_label_training_only','exposed_development','independent_verified','not_evaluated'):
        raise ValueError('Unknown evaluation scope')
    if event['evaluation_scope']=='independent_verified':
        raise ValueError('This provisional tracker cannot certify independent accuracy; use the protected verified evaluator')
    save(Path(root)/(uuid.uuid4().hex+'.json'),event)

def score(metrics,reference):
    """Do not select higher precision produced solely by losing supported recalls."""
    candidates=[p for p in metrics['pr_curve'] if p['recall'] is not None and p['recall']>=reference['recall']]
    matched=max((p['precision'] for p in candidates),default=-1)
    valid=(matched>=0 and metrics['ap75']>=reference['ap75']-.02)
    for c in ('sofa','chair','bed'):
        a=metrics['categories'][c];b=reference['categories'][c]
        if a['tp']<b['tp'] or (b['ap75'] is not None and a['ap75']<b['ap75']-.05):valid=False
    return (int(valid),matched,metrics['ap75'],metrics['f1'] or 0)
