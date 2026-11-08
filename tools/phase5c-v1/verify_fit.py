"""Independent committed-checkpoint verification before advancing the queue."""
import argparse
import torch
from safetensors import safe_open
from common import *
from control import RUN_ROOT,FIT_IDS,check_resume
from checkpoints import recover

def run(fit_id,expected_updates):
 out=RUN_ROOT/fit_id;binding=read(out/'binding.json');latest=recover(out,binding)
 if latest is None:raise ValueError('No committed checkpoint')
 _,state,path,digest=latest
 if state['step']!=expected_updates or state['optimizer_hook_calls']!=expected_updates:raise ValueError('Checkpoint/update count differs')
 check_resume(fit_id,expected_updates)
 names=set(read(out/'scope.json')['trainable_names'])
 if set(state['weights'])!=names or any(not n.startswith(('class_head.','box_head.')) for n in names):raise ValueError('Checkpoint includes non-head parameters')
 if not all(torch.isfinite(t).all() for t in state['weights'].values()):raise ValueError('Nonfinite checkpoint')
 steps=[int(s['step'].item()) for s in state['optimizer']['state'].values()]
 if not steps or any(n!=expected_updates for n in steps):raise ValueError('Adam state update count mismatch')
 changed=0;max_delta=0.
 with safe_open(str(OWL/'model.safetensors'),framework='pt',device='cpu') as baseline:
  for name,value in state['weights'].items():
   delta=(value-baseline.get_tensor(name)).abs();changed+=bool(torch.any(delta));max_delta=max(max_delta,float(delta.max()))
 if changed==0:raise ValueError('No head tensor changed despite optimizer accounting')
 if expected_updates==70:
  done=read(out/'complete.json')
  if not done['frozen_parameters_unchanged'] or done['epochs']!=10:raise ValueError('Incomplete or frozen backbone changed')
  best=done['best']
  if sha(best['checkpoint'])!=best['sha256']:raise ValueError('Selected checkpoint checksum mismatch')
  vals=[read(p) for p in sorted(out.glob('validation-epoch-*.json'))]
  if len(vals)!=10:raise ValueError('Incomplete epoch validation')
  selected=max(vals,key=lambda r:(r['metrics']['macro_ap50_supported'],-r['epoch']))
  if selected['epoch']!=best['epoch'] or selected['metrics']['macro_ap50_supported']!=best['score']:raise ValueError('Incorrect development selection')
  expected_frozen=read(out/'scope.json')['frozen_parameter_sha256']
  if done['frozen_parameter_sha256']!=expected_frozen:raise ValueError('Frozen fingerprint mismatch')
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original base weights changed')
 result=dict(fit_id=fit_id,updates=expected_updates,committed_checkpoints_verified=len(list((out/'checkpoints').glob('*.receipt.json'))),latest_checkpoint=path,latest_sha256=digest,
     optimizer_state_steps=sorted(set(steps)),head_tensors_changed=changed,max_abs_head_delta=max_delta,non_head_tensors_in_checkpoint=0,original_checkpoint_unchanged=True)
 target=out/f'verification-step-{expected_updates:05d}.json'
 save(target,result);print(json.dumps(result,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fit-id',choices=FIT_IDS,required=True);ap.add_argument('--expected-updates',type=int,required=True);a=ap.parse_args();run(a.fit_id,a.expected_updates)
