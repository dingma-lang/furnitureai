"""Independently verify the approved acquisition and opaque historical hashes."""
import collections as C
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *
from rights import verify as verify_rights
from screen import signature,duplicate
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical

def sums(base,name):
 n=0
 for line in (base/name).read_text().splitlines():
  expected,rel=line.split('  ',1);p=(base/rel).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=expected:raise ValueError('Historical artifact changed '+str(p))
  n+=1
 return n

def category_counts(rows,masked=False):
 return {c:dict(images=sum(any(g['label']==c and (not masked or (not g['ignore'] and r['coverage'][c])) for g in r['original_annotations'] if True) for r in rows),
  boxes=sum(g['label']==c and (not masked or (not g['ignore'] and r['coverage'][c])) for r in rows for g in (r['annotations'] if masked else r['original_annotations']))) for c in CLASSES}

def run():
 from PIL import Image
 import gzip,io
 manifest=read(MANIFEST);acq=read(DEST/'acquisition-complete.json');final=read(DEST/'screening-v3.json');ledger=read(DEST/'network-ledger.json')
 assert sha(MANIFEST)=='c6937c1486909e2b665a8740222c313a61a1c8cf6dba07ee29a610be5046f7a8'
 lookup={r['image_id']:r for r in manifest['images']};rows=final['images'];source_rows=C.defaultdict(list)
 with (META/'validation-annotations-bbox.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   if row['ImageID'] in lookup:source_rows[row['ImageID']].append(row)
 assert len(lookup)==len(acq['records'])==len(rows)==41
 assert {r['image_id'] for r in rows}==set(lookup)
 assert C.Counter(r['proposed_split'] for r in rows)==dict(train=32,development=9)
 source_boxes=0;dimensions=[]
 for acquired in acq['records']:
  ident=acquired['image_id'];r=next(x for x in rows if x['image_id']==ident);p=DEST/acquired['file'];source=lookup[ident]
  assert acquired['source']==source and r['proposed_split']==source['proposed_split']
  assert sha(p)==r['sha256']==acquired['sha256'] and p.stat().st_size==acquired['bytes']==source['remote_object']['bytes']
  assert r['original_annotations']==source['annotations']
  for g in r['original_annotations']:
   box(g['box']);assert g['source_mid']==MIDS[g['label']]
   matches=[q for q in source_rows[ident] if q['LabelName']==g['source_mid'] and [float(q[k]) for k in ('XMin','YMin','XMax','YMax')]==g['box'] and all(int(q[k])==v for k,v in g['flags'].items())]
   assert matches;source_boxes+=1
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:
   im.load();assert im.size==(r['width'],r['height']);assert im.getexif().get(274,1)==1
   actual=signature(im)
  for key,value in actual.items():assert r[key]==value
  dimensions.append([r['width'],r['height']])
  rights=read(ROOT/source['license_record']);assert sha(ROOT/source['license_record'])==source['license_record_sha256']
  body=OUT/'rights-audit'/rights['response_file'];assert sha(body)==rights['response_sha256']
  data=body.read_bytes()
  if data[:2]==b'\x1f\x8b':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:data=stream.read(4194305)
  current=verify_rights(data.decode('utf-8'),source);assert current['creator']==rights['creator'] and current['license_url']==source['license_url']
 # Compare ABO manifests by existing hashes only; no historical image decode.
 abo=[]
 for relative in ('data/abo/image-pilots/phase1b-v1/final-screening.json','data/abo/image-pilots/phase3-expansion-v1/audit-v3/screening.json'):
  abo.extend(read(ROOT/relative)['images'])
 abo_exact=[dict(new=r['image_id'],historical=b['image_id']) for r in rows for b in abo if (r['sha256']==b.get('sha256') or r['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'))]
 assert not abo_exact
 isolation(rows,set(read(OUT/'protected-identities.json')))
 assert not final['duplicates']
 for i,a in enumerate(rows):
  for b in rows[i+1:]:assert duplicate(a,b) is None
 assert len(ledger['requests'])==41 and all(r['complete'] and r['attempt']==1 and r['kind']=='image' for r in ledger['requests'])
 assert sum(r['actual_bytes'] for r in ledger['requests'])==ledger['actual_bytes']==10487710
 assert ledger['charged']+manifest['rights_conservative_charge']<TRANSFER
 selected=[r for r in rows if r['status']=='provisional_automated_candidate']
 assert C.Counter(r['proposed_split'] for r in selected)==dict(train=9,development=5)
 assert all(not r['human_verified'] for r in rows)
 for r in selected:
  assert not r['coverage']['desk'] and not r['coverage']['cabinet']
  assert all(g['ignore'] or r['coverage'][g['label']] for g in r['annotations'])
 old=historical(META);verified={}
 for name,base,filename in [('phase6a_preparation',OUT,'ARTIFACT-SHA256SUMS'),('metadata',META,'ARTIFACT-SHA256SUMS'),('old_photos',OLD,'ARTIFACT-SHA256SUMS'),('phase5b',ROOT/'data/open-images/experiments/phase5b-v1','ARTIFACT-SHA256SUMS'),('phase5c',ROOT/'data/open-images/experiments/phase5c-v1','ARTIFACT-SHA256SUMS')]:verified[name]=sums(base,filename)
 for base in (OUT,ROOT/'data/open-images/experiments/phase5b-v1',ROOT/'data/open-images/experiments/phase5c-v1'):
  for path,digest in read(base/'implementation-sha256.json').items():assert sha(ROOT/path)==digest
 for split in ('train','development'):
  subset=[r for r in selected if r['proposed_split']==split]
  # Image counts use effective optimization annotations, not native ignored flags.
  counts={c:dict(images=sum(any(g['label']==c and not g['ignore'] and r['coverage'][c] for g in r['annotations']) for r in subset),boxes=sum(g['label']==c and not g['ignore'] and r['coverage'][c] for r in subset for g in r['annotations'])) for c in CLASSES}
  write_new(DEST/(split+'-eligible-counts.json'),dict(images=len(subset),categories=counts))
 write_new(DEST/'CURRENT-SCREENING.json',dict(path='screening-v3.json',sha256=sha(DEST/'screening-v3.json'),supersedes=['screening-v1.json','screening-v2.json'],training_approved=False))
 write_new(DEST/'eligible-candidates-v1.json',dict(images=selected,screening_sha256=sha(DEST/'screening-v3.json'),training_approved=False,final_training_dataset=False,
  warning='Use CURRENT-SCREENING and this post-acquisition record, never the historical preparation training.py function hardcoded to screening-v1. A future adapter/queue must bind final screening and receive separate training approval.'))
 result=dict(status='PASS',downloaded_photos=41,train_slots=32,development_slots=9,provisionally_eligible_train=9,provisionally_eligible_development=5,quarantined=27,
  source_confirmed_native_boxes=source_boxes,source_confirmation_scope='Exact match to official local source rows, MID and source flags plus normalized geometry. Not independent human verification of visual category or boundary accuracy.',
  actual_photo_body_bytes=ledger['actual_bytes'],prior_rights_actual_body_bytes=manifest['rights_audit_body_bytes'],actual_combined_body_bytes=ledger['actual_bytes']+manifest['rights_audit_body_bytes'],conservative_combined_charge=ledger['charged']+manifest['rights_conservative_charge'],
  retries=0,partial_transfers=0,exact_duplicates=0,suspected_perceptual_duplicates=0,abo_metadata_hash_records_checked=len(abo),abo_exact_overlap=abo_exact,
  historical=old,historical_checksum_ledgers=verified,all_prior_implementation_hashes_match=True,original_annotations_preserved=True,partitions_unchanged=True,
  source_rights_documents_reverified=41,original_model_hashes_unchanged=True,frozen_pixels_decoded_or_scored=0,optimizer_updates=0,model_inference=0,mac_application_accessed=False,
  dimensions_min=[min(d[i] for d in dimensions) for i in (0,1)],dimensions_max=[max(d[i] for d in dimensions) for i in (0,1)],
  remaining_uncertainties=['No human review fabricated','Unknown cross-author physical identity remains unproved','Perceptual matching covers new images and comparable old Open Images signatures; ABO is existing-hash only, with no old or frozen pixels opened','Source-native Table/Desk/Cabinetry units remain ambiguous and masked or unresolved','Eligible bed crops remain partial; new cohort is too small and imbalanced for general seven-class training'])
 assert allocated(DEST)+allocated(OUT)+allocated(HERE)+allocated(Path(__file__).parent)<STORAGE
 write_new(DEST/'integrity-verification.json',result);print('ACQUISITION AUDIT PASS',source_boxes,'source boxes;',len(selected),'provisional photos')

if __name__=='__main__':run()
