"""Reproduce supplementary operating-point and same-instance localization tables.

Default output requires fresh filenames. Use --output for a separate CPU-only
reproduction directory; it never regenerates predictions or touches weights.
"""
import argparse,importlib.util
from common8 import *

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    verify_binding(read(RUN/'binding.json'))
    entries=read(RUN/'metric-results.json')['entries'];operating={}
    for e in entries:
        if e['epoch'] not in (0,5):continue
        curve=e['metrics']['pr_curve'];at90=[p for p in curve if p['recall']>=.9];at80=[p for p in curve if p['precision']>=.8]
        operating[e['arm']+'-E'+str(e['epoch'])]=dict(best_precision_at_90_recall=max(at90,key=lambda p:p['precision']) if at90 else None,
            best_recall_at_80_precision=max(at80,key=lambda p:p['recall']) if at80 else None,
            joint_80P90R=any(p['precision']>=.8 and p['recall']>=.9 for p in curve),descriptive_only_no_threshold_adopted=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('supplement_cpu8',ROOT/'tools/phase10c-v1/metric_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    _,dev=subsets();scoped=[m.scopes(r) for r in dev];pred=[read(RUN/a/'evaluation/epoch-5/predictions.json') for a in ARMS];result={}
    for c in ('sofa','chair','bed'):
        maps=[]
        for ps in pred:
            events,_,_=m.events(scoped,ps,c,floor=.15);maps.append({(e['image_id'],e['gt_index']):e['iou'] for e in events if e['outcome']=='tp'})
        common=set(maps[0])&set(maps[1]);onlyA=set(maps[0])-set(maps[1]);onlyB=set(maps[1])-set(maps[0])
        a=sum(maps[0][k] for k in common)/len(common);b=sum(maps[1][k] for k in common)/len(common)
        result[c]=dict(common_TP_instances=len(common),common_TP_mean_IoU_A=a,common_TP_mean_IoU_B=b,delta=b-a,
            A_only_TP_IoUs=[maps[0][k] for k in sorted(onlyA)],B_only_TP_IoUs=[maps[1][k] for k in sorted(onlyB)])
    save(output/'operating-points.json',operating)
    save(output/'matched-localization.json',dict(scope='Same GT instances matched at confidence0.15 IoU0.5 on19internal photos',categories=result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(REPORT));run(p.parse_args().output)
