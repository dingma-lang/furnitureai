"""Bounded CPU scalar-kernel measurement; not detector or inference benchmarking."""
import ast,random,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from furniture_bench.geometry import iou as current_iou
from audit12 import emit,OUT
PARENT='5818ab333dfd239d581129e02af93238bc85e191'
def main():
    text=subprocess.check_output(['git','show',PARENT+':src/furniture_bench/geometry.py'],text=True)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='iou')
    scope={'Sequence':list};exec(compile(ast.Module(body=[node],type_ignores=[]),'preserved-original-iou','exec'),scope)
    original=scope['iou'];rng=random.Random(20261123);pairs=[]
    for _ in range(2000):
        def box():
            x,y=rng.random()*100,rng.random()*100;return [x,y,x+rng.random()*100+1,y+rng.random()*100+1]
        pairs.append((box(),box()))
    times={'before':[],'after':[]};checksums={}
    for round in range(8):
        order=[('before',original),('after',current_iou)]
        if round%2:order.reverse()
        for name,fn in order:
            start=time.perf_counter();total=0.
            for repeat in range(40):
                for a,b in pairs:total+=fn(a,b)
            times[name].append(time.perf_counter()-start);checksums[name]=total.hex()
    if checksums['before']!=checksums['after']:raise ValueError('Scalar behavior changed')
    medians={k:statistics.median(v) for k,v in times.items()}
    emit(OUT/'CPU-microbenchmark.json',dict(benchmark='scalar IoU kernel only',calls_per_round=80000,rounds_per_version=8,
         alternating_order=True,seconds=times,median_seconds=medians,ratio_after_to_before=medians['after']/medians['before'],
         identical_accumulated_float_hex=checksums,threads_or_GPU_used=False,actual_model_inference=False,
         limitation='Single-process microbenchmark with warm data and shared VM; do not infer detector latency, memory savings, or application speedup.'))
    print('CPU scalar median seconds',medians,'after/before',medians['after']/medians['before'])
if __name__=='__main__':main()
