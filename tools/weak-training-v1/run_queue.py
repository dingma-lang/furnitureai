"""Prepare, smoke-test, resume the bounded GPU queue and generate its report."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import pipeline as p

child=None
stop_requested=False

def stop(signum, frame):
 global stop_requested
 stop_requested=True
 if child is not None and child.poll() is None:child.send_signal(signum)

def call(args):
 global child
 child=subprocess.Popen([sys.executable,'-B',*map(str,args)])
 code=child.wait();child=None
 if code:raise SystemExit(code)
 if stop_requested:raise SystemExit(0)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=p.OUTPUT);parser.add_argument('--source-inventory',type=Path,default=p.PREP/'eligibility-inventory.json');args=parser.parse_args();out=args.output.resolve()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 script=p.HERE/'pipeline.py'
 if not (out/'manifests-v1').exists():call([script,'prepare','--output',out,'--source-inventory',args.source_inventory])
 if not (out/'smoke-gate.json').exists():
  call([script,'smoke','--output',out])
  if not (out/'smoke-gate.json').exists():return
 call([script,'run','--output',out])
 if (out/'completion.json').exists():call([p.HERE/'report.py','--output',out])

if __name__=='__main__':main()
