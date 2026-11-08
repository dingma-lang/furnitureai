"""Resume the bounded queue and the single frozen comparison; completed stages skip safely."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import train as t

child=None
stopping=False

def stop(signum,frame):
 global stopping
 stopping=True
 if child is not None and child.poll() is None:child.send_signal(signum)

def call(*args):
 global child
 child=subprocess.Popen([sys.executable,'-B',*map(str,args)]);code=child.wait();child=None
 if code:raise SystemExit(code)
 if stopping:raise SystemExit(0)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=t.OUT);args=parser.parse_args();out=args.output.resolve()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 if not (out/'input-lock.json').exists():call(t.HERE/'train.py','prepare','--output',out)
 if not (out/'smoke-gate.json').exists():
  call(t.HERE/'train.py','smoke','--output',out)
  if not (out/'smoke-gate.json').exists():return
 if not (out/'queue-complete.json').exists():
  call(t.HERE/'train.py','run','--output',out)
  if not (out/'queue-complete.json').exists():return
 call(t.HERE/'compare.py','--output',out)
 call(t.HERE/'report.py','--output',out)

if __name__=='__main__':main()
