"""Future max-two-fit queue; no parameter update is authorized in Phase 5B itself."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import *
from checkpoints import Lease
STOP=False

def stop(*_):
 global STOP
 STOP=True

def run(parent,resume=False,allow=False):
 if not allow:raise PermissionError('Separate user approval and --allow-parameter-updates required')
 parent=Path(parent).resolve()
 if not parent.is_relative_to(ROOT/'data/open-images/experiments') or parent==OUT:raise ValueError('Isolated queue root required')
 if parent.exists() and not resume:raise FileExistsError('Explicit queue --resume required')
 parent.mkdir(parents=True,exist_ok=resume)
 with Lease(parent/'queue.lock'):
  configs=[OUT/'configs/heads-lr1e-5.json',OUT/'configs/heads-lr3e-5.json']
  binding={str(p):sha(p) for p in configs}
  if (parent/'queue-binding.json').exists():
   if read(parent/'queue-binding.json')!=binding:raise ValueError('Queue configs changed')
  else:save(parent/'queue-binding.json',binding)
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  leaderboard=[]
  for config in configs:
   target=parent/config.stem;done=target/'complete.json'
   if STOP or (parent/'STOP').exists():break
   if not done.exists():
    args=[sys.executable,'-B',str(HERE/'train.py'),'--config',str(config),'--out',str(target),'--allow-parameter-updates']
    if target.exists():args+=['--resume']
    with (parent/(config.stem+'.log')).open('a') as log:
     process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     started=time.monotonic();stop_sent=False
     while process.poll() is None:
      if (STOP or (parent/'STOP').exists() or time.monotonic()-started>1800) and not stop_sent:
       process.send_signal(signal.SIGTERM);stop_sent=True;stop_started=time.monotonic()
      if stop_sent and time.monotonic()-stop_started>60:
       process.kill();process.wait();raise RuntimeError('Worker exceeded graceful-stop deadline; committed snapshots preserved')
      time.sleep(1)
     if process.returncode:raise RuntimeError('Training worker failed; inspect append-only log and resume committed state')
    if not done.exists():break # graceful stop: do not advance the queue
   if done.exists():
    completed=read(done);fit_binding=read(target/'binding.json')
    if fit_binding['config']!=read(config) or fit_binding['dataset_sha256']!=sha(OUT/'dataset-lock.json'):raise ValueError('Completed fit binding differs')
    best=completed['best']
    if best and sha(best['checkpoint'])!=best['sha256']:raise ValueError('Selected checkpoint changed')
    leaderboard.append(dict(configuration=config.stem,**completed))
  # Own mutable registry only, never historical or checkpoint files.
  sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
  from acquire import atomic
  atomic(parent/'leaderboard.json',dict(experiments=leaderboard,selection_split='development',frozen_used=False,max_configurations=2,max_optimizer_updates_per_fit=70))
  print(json.dumps(leaderboard,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out-parent',required=True,type=Path);ap.add_argument('--resume',action='store_true');ap.add_argument('--allow-parameter-updates',action='store_true');a=ap.parse_args();run(a.out_parent,a.resume,a.allow_parameter_updates)
