"""Bounded CPU-only probes of the unchanged historical validator/metric path."""
import argparse
import os
import resource
import sys
import time
from c10 import *

def run(mode,torch_import=False):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.environ['CUDA_VISIBLE_DEVICES']=''
    started=time.monotonic()
    if torch_import:import torch
    sys.path.insert(0,str(ROOT/'tools/phase9-v1'))
    from fusion import valid
    from metrics import events,ap
    calls=0
    if mode=='validator':
        p=dict(label='chair',box=[.1,.2,.7,.8],score=.5)
        for _ in range(500000):valid(p);calls+=1
    elif mode=='metrics':
        lock=read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')
        rows=[dict(r,fp_coverage=r['coverage']) for r in lock['rows'] if r['split']=='development']
        pred=read(ROOT/'data/open-images/experiments/phase9-multiview-v1/evaluations/exposed/adapted-scale_nms.json')['predictions']
        for _ in range(3):
            for c in CLASSES:
                ev,n,_=events(rows,pred,c);ap(ev,n);calls+=1
    else:raise ValueError('Unknown stress mode')
    print(json.dumps(dict(status='PASS_NOT_REPRODUCED',mode=mode,torch_import=torch_import,calls=calls,
        seconds=time.monotonic()-started,GPU_initialized=False if not torch_import else torch.cuda.is_initialized(),
        root_cause_proven=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validator','metrics'],required=True);p.add_argument('--torch-import',action='store_true');a=p.parse_args();run(a.mode,a.torch_import)
