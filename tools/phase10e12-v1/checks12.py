"""Offline CPU fixture tests and exact before/after behavior fingerprints."""
import argparse,contextlib,hashlib,io,json,os,random,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/phase10e12-v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from audit12 import emit,digest
MODULES=('test_benchmark','test_dataset_quality','test_detection_integration','test_product_identification','test_retrieval_optimization','test_catalog_expansion')
def flat(suite):
 for test in suite:
  if isinstance(test,unittest.TestSuite):yield from flat(test)
  else:yield test

def tests(stage):
 if any((REPORT/('CPU-'+stage+suffix)).exists() for suffix in ('.log','.json')):raise FileExistsError('Audit test report already exists; choose a fresh version')
 from unittest.mock import patch
 tests=list(flat(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in MODULES)))
 excluded=[t.id() for t in tests if 'TinyTransformersTests.' in t.id()]
 chosen=[t for t in tests if t.id() not in excluded]
 chosen_ids={id(t):t.id() for t in chosen}
 start=time.perf_counter();log=io.StringIO()
 with patch('socket.create_connection',side_effect=AssertionError('PhaseE12 tests must not access network')),contextlib.redirect_stdout(log):
  result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite(chosen))
 text=log.getvalue();(REPORT/('CPU-'+stage+'.log')).write_text(text)
 value=dict(status='PASS' if result.wasSuccessful() else 'FAIL',ran=result.testsRun,failures=len(result.failures),errors=len(result.errors),
   skipped=[dict(test=chosen_ids[id(t)],reason=r) for t,r in result.skipped],excluded_real_model_forwards=excluded,
   test_ids=[chosen_ids[id(t)] for t in chosen],elapsed_seconds=time.perf_counter()-start,log_sha256=digest(REPORT/('CPU-'+stage+'.log')),
   CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),downloads=0,actual_trained_model_forwards=0)
 emit(REPORT/('CPU-'+stage+'.json'),value);print(json.dumps({k:v for k,v in value.items() if k not in ('test_ids',)},indent=2))
 if not result.wasSuccessful():print(text[-6500:]);raise SystemExit(1)

def behavior(stage):
 import numpy as np
 from PIL import Image
 from furniture_bench.geometry import iou,nms
 from furniture_bench.postprocessing import containment,FilterConfig,filter_detections
 from furniture_bench.evaluation import in_ignored_region,match_objects
 from furniture_bench.dataset_quality import fingerprint
 def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 rng=random.Random(20261123);h=hashlib.sha256();cases=25000
 for i in range(cases):
  def box():
   x,y=rng.uniform(0,10000),rng.uniform(0,10000);return [x,y,x+rng.uniform(.001,1000),y+rng.uniform(.001,1000)]
  a,b=box(),box()
  if i%3==0:b=a.copy()
  if i%5==0:b=[a[0]+.1,a[1]+.2,a[2]+.3,a[3]+.4]
  h.update(canonical([float(iou(a,b)).hex(),float(containment(a,b)).hex(),in_ignored_region(dict(box=a),[dict(box=b)])]))
 saved=ROOT/'reports/gpu-owlv2-rtx4070-test01/predictions.json';run=json.loads(saved.read_text());configs=[FilterConfig(),FilterConfig(containment_threshold=.8),FilterConfig(cross_class_iou=.7)]
 captures=[]
 for record in run['images']:
  detections=record.get('detections',[])
  for cfg in configs:
   kept,report=filter_detections(detections,cfg);captures.append(dict(image_id=record['image_id'],detections=kept,report=report,nms=nms(detections,cfg.nms_iou)))
 # Synthetic matching only: never score official/reserved or new held-out photographs.
 p=[dict(label='chair',score=.8,box=[0,0,10,10]),dict(label='chair',score=.8,box=[1,1,11,11]),dict(label='bed',score=.2,box=[0,0,10,10])]
 gt=[dict(label='chair',box=[0,0,10,10]),dict(label='chair',box=[0,0,10,10])]
 matches=[match_objects(p,gt,t) for t in (.5,.75,1.)]
 fingerprints=[];np_rng=np.random.default_rng(20261123)
 with tempfile.TemporaryDirectory(prefix='furniture-e12-fixtures-') as tmp:
  for orientation in range(1,9):
   array=np.full((64,91,3),255,dtype=np.uint8);array[7:55,9:75]=np_rng.integers(0,230,size=(48,66,3),dtype=np.uint8)
   image=Image.fromarray(array);exif=Image.Exif();exif[274]=orientation
   path=Path(tmp)/f'orientation-{orientation}.jpg';image.save(path,exif=exif,quality=90)
   f=fingerprint(path);fingerprints.append(dict(name=path.name,sha256=f['sha256'],rgb_sha256=hashlib.sha256(f['rgb'].tobytes()).hexdigest(),bits_sha256=hashlib.sha256(f['bits'].tobytes()).hexdigest()))
  for mode,suffix in [('RGBA','.png'),('L','.png'),('CMYK','.jpg'),('P','.gif')]:
   image=Image.new('RGB',(20,13),'white');image.paste((40,70,120),(2,2,17,11));image=image.convert(mode)
   path=Path(tmp)/(mode+suffix);image.save(path);f=fingerprint(path);fingerprints.append(dict(name=path.name,sha256=f['sha256'],rgb_sha256=hashlib.sha256(f['rgb'].tobytes()).hexdigest(),bits_sha256=hashlib.sha256(f['bits'].tobytes()).hexdigest()))
 value=dict(geometry_cases=cases,geometry_float_hex_and_ignore_SHA256=h.hexdigest(),saved_prediction_source=str(saved.relative_to(ROOT)),saved_prediction_source_sha256=digest(saved),
   saved_room_images=len(run['images']),saved_filter_outputs_sha256=hashlib.sha256(canonical(captures)).hexdigest(),saved_filter_cases=len(captures),
   synthetic_matching_sha256=hashlib.sha256(canonical(matches)).hexdigest(),fingerprint_fixtures=fingerprints,models_loaded=0,official_holdout_scoring=False)
 emit(REPORT/('behavior-'+stage+'.json'),value);print('Behavior fingerprints',stage,'saved filters',len(captures),'geometry',cases,'image fixtures',len(fingerprints))
if __name__=='__main__':
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('HF_HUB_OFFLINE')!='1':raise PermissionError('CPU offline environment required')
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','all-tests','behavior']);p.add_argument('stage',choices=['before','before-v2','after','extended-v2']);a=p.parse_args()
 if a.action=='all-tests':
  MODULES=tuple(p.stem for p in sorted((ROOT/'tests').glob('test_*.py')));tests(a.stage)
 else:globals()[a.action](a.stage)
