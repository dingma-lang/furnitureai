"""Saved E5-versus-C11 output proximity;no new scoring or inference."""
import numpy as np
from common5 import *
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from analyzeD import load_raw
    dev=read(PREP/'training-context.json')['development']
    a=load_raw(RUN/'predictions',dev)
    b=load_raw(ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/evaluation/epoch-5',dev)
    x=np.concatenate([v['logits'] for v in a]).astype(float);y=np.concatenate([v['logits'] for v in b]).astype(float)
    boxes=np.concatenate([v['boxes'] for v in a]).astype(float)-np.concatenate([v['boxes'] for v in b]).astype(float)
    emit(REPORT/'C11-proximity.json',dict(logits_bit_identical=bool(np.array_equal(x,y)),mean_logit_difference=float((x-y).mean()),
        RMS_logit_difference=float(np.sqrt(np.mean((x-y)**2))),maximum_absolute_logit_difference=float(np.abs(x-y).max()),
        raw_argmax_disagreements=int((x.argmax(1)!=y.argmax(1)).sum()),
        maximum_absolute_normalized_box_difference=float(np.abs(boxes).max()),
        inference_runs=0,interpretation='Distinct newly saved checkpoint/outputs,nearly identical benchmark behavior;cannot isolate KD from numerical-backward differences.'))
if __name__=='__main__':run()
