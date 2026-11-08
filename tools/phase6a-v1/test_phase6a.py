import copy
import io
import tempfile
import unittest
from unittest.mock import patch
from core import *
from network import Budget,approval
from rights import verify
from finalize import split
from screen import signature,duplicate

def candidate(ident='a',group='g',part='train'):
 return dict(image_id=ident,source_identity='flickr-photo:'+ident,identity_group=group,photographer_identity='author:'+group,metadata_original_md5='md5:'+ident,proposed_split=part,eligible_box_counts={'chair':1})

class Response:
 status=200
 def __init__(self,body=b'hello',fail=False):self.body=io.BytesIO(body);self.fail=fail;self.calls=0;self.headers={'Content-Length':str(len(body)),'Content-Type':'text/html'}
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self,n):
  self.calls+=1
  if self.fail and self.calls>1:raise OSError('Simulated interrupted response')
  return self.body.read(min(n,2) if self.fail else n)

class Tests(unittest.TestCase):
 def test_allowed_and_restricted_licences(self):
  for url in ['https://creativecommons.org/licenses/by/2.0/','https://creativecommons.org/licenses/by/4.0/','https://creativecommons.org/publicdomain/zero/1.0/']:self.assertEqual(license_url(url),url)
  for url in ['https://creativecommons.org/licenses/by-nc/4.0/','https://creativecommons.org/licenses/by-sa/4.0/','https://creativecommons.org/licenses/by-nd/4.0/','https://creativecommons.org.evil/licenses/by/4.0/','https://creativecommons.org:123/licenses/by/4.0/']:
   with self.assertRaises(ValueError):license_url(url)
 def test_individual_evidence_is_required(self):
  row=dict(source_identity='flickr-photo:123',license_url='https://creativecommons.org/licenses/by/2.0/')
  evidence={'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/owner/123','license':row['license_url'],'author':{'name':'Creator','url':'https://www.flickr.com/photos/owner'}}
  text='<script type="application/ld+json">'+json.dumps(evidence)+'</script>'
  self.assertEqual(verify(text,row)['creator'],'Creator')
  with self.assertRaises(ValueError):verify(text,dict(row,source_identity='flickr-photo:999'))
  with self.assertRaises(ValueError):verify('Dataset photos are CC BY',row)
  evidence['author']['url']='https://www.flickr.com/photos/other'
  with self.assertRaises(ValueError):verify('<script type="application/ld+json">'+json.dumps(evidence)+'</script>',row)
 def test_licence_change_is_not_silently_accepted(self):
  e={'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/a/1','license':'https://creativecommons.org/licenses/by/4.0/','author':{'name':'A','url':'https://www.flickr.com/photos/a'}}
  with self.assertRaises(ValueError):verify('<script type="application/ld+json">'+json.dumps(e)+'</script>',dict(source_identity='flickr-photo:1',license_url='https://creativecommons.org/licenses/by/2.0/'))
 def test_coco_xywh_and_crowd_preserved(self):
  d=dict(categories=[dict(id=62,name='chair')],images=[dict(id=1,width=100,height=200)],annotations=[dict(id=8,image_id=1,category_id=62,bbox=[10,20,50,100],iscrowd=1)])
  r=parse_coco(d)[0];self.assertEqual(r['box'],[.1,.1,.6,.6]);self.assertTrue(r['ignore']);self.assertEqual(r['original_annotation'],d['annotations'][0])
 def test_coco_rejects_schema_and_reference_conflicts(self):
  d=dict(categories=[dict(id=62,name='desk')],images=[],annotations=[])
  with self.assertRaises(ValueError):parse_coco(d)
  with self.assertRaises(ValueError):coco_box([-1,0,10,20],100,100)
  with self.assertRaises(ValueError):box([0,0,float('nan'),1])
 def test_source_native_taxonomy_not_fabricated(self):
  self.assertEqual(set(x[1] for x in COCO.values()),{'chair','sofa','bed','table'})
  self.assertEqual(COCO[67][0],'dining table');self.assertNotIn('desk',{x[1] for x in COCO.values()})
 def test_descendant_only_regions_are_not_background_or_invented_gt(self):
  from annotation_scope import descendant_parents,conflicts
  tree={'LabelName':MIDS['table'],'Subcategory':[{'LabelName':'/m/child'}]};parents=descendant_parents(tree)
  raw=[dict(LabelName='/m/child',XMin='0',YMin='0',XMax='1',YMax='1')]
  result=conflicts(raw,[],parents);self.assertEqual(result[0]['affected_parent'],'table')
  self.assertEqual(conflicts(raw,[dict(label='table',box=[0,0,1,1])],parents),[])
 def test_identity_and_source_group_leakage(self):
  a=candidate();b=candidate('b','g','development')
  with self.assertRaises(ValueError):isolation([a,b])
  b=candidate('b','h','development');self.assertTrue(isolation([a,b]))
  with self.assertRaises(ValueError):isolation([a,b],{'g'})
  with self.assertRaises(ValueError):isolation([dict(a,proposed_split='frozen_evaluation')])
 def test_exact_hash_leakage(self):
  a=dict(candidate(),sha256='hash');b=dict(candidate('b','h','development'),sha256='hash')
  with self.assertRaises(ValueError):isolation([a,b])
 def test_stratified_splits_are_reproducible(self):
  rows=[candidate(str(i),str(i)) for i in range(20)]
  for r in rows[:3]:r['eligible_box_counts']['bed']=1
  one=split(rows,4);two=split(list(reversed(rows)),4)
  self.assertEqual({r['image_id']:r['proposed_split'] for r in one},{r['image_id']:r['proposed_split'] for r in two})
  self.assertEqual(sum(r['proposed_split']=='development' for r in one),4)
  self.assertTrue(all(any(r['eligible_box_counts'].get('bed') and r['proposed_split']==part for r in one) for part in ('train','development')))
 def test_photograph_url_whitelist(self):
  self.assertEqual(image_url(dict(image_id='a'*16,source_partition='validation')),'https://open-images-dataset.s3.amazonaws.com/validation/'+'a'*16+'.jpg')
  for part in ['test','train','../validation']:
   with self.assertRaises(ValueError):image_url(dict(image_id='a'*16,source_partition=part))
 def test_no_implicit_acquisition_approval(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);write_new(p/'manifest.json',{});write_new(p/'approval.json',dict(image_acquisition_approved=True,manifest_sha256='wrong'))
   with self.assertRaises(PermissionError):approval(p/'manifest.json',p/'approval.json')
 def test_transfer_cache_recovery_and_hashes(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Budget(Path(tmp)/'run',cap=10);response=Response()
   try:
    with patch.object(b.opener,'open',return_value=response):r=b.request('https://www.flickr.com/photos/a/1','a.html',10,'rights')
    self.assertEqual(b.state['actual_bytes'],5);self.assertEqual(b.state['charged'],5)
    with patch.object(b.opener,'open',side_effect=AssertionError('No duplicate network request')):self.assertEqual(b.request(r['url'],'a.html',10,'rights'),r)
    (b.root/'a.html').write_text('tampered')
    with self.assertRaises(ValueError):b.request(r['url'],'a.html',10,'rights')
   finally:b.close()
 def test_interrupted_response_retains_budget_and_partial(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Budget(Path(tmp)/'run',cap=20)
   try:
    with patch.object(b.opener,'open',return_value=Response(fail=True)):
     with self.assertRaises(OSError):b.request('https://www.flickr.com/photos/a/1','partial.html',10,'rights')
    self.assertEqual(b.state['charged'],10);self.assertEqual(b.state['actual_bytes'],2);self.assertEqual((b.root/'partial.html').read_bytes(),b'he')
   finally:b.close()
   b=Budget(Path(tmp)/'run',cap=20)
   try:
    with patch.object(b.opener,'open',return_value=Response()):b.request('https://www.flickr.com/photos/a/1','retry.html',10,'rights')
    self.assertEqual(b.state['charged'],15);self.assertEqual(b.state['actual_bytes'],7)
   finally:b.close()
 def test_committed_image_body_recovers_before_metadata_commit(self):
  from acquire import recovery_filename
  records=[dict(url='url',file='photos/id-a1.jpg',complete=True)]
  self.assertEqual(recovery_filename(records,'url','id'),'photos/id-a1.jpg')
  self.assertEqual(recovery_filename([dict(records[0],complete=False)],'url','id'),'photos/id-a2.jpg')
 def test_transfer_size_cap_and_unsafe_path(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Budget(Path(tmp)/'run',cap=4)
   try:
    with self.assertRaises(RuntimeError):b.request('https://www.flickr.com/photos/a/1','x',5,'rights')
    with self.assertRaises(ValueError):b.request('https://www.flickr.com/photos/a/1','../escape',1,'rights')
    with self.assertRaises(ValueError):b.request('https://example.com/photo','x',1,'image')
    self.assertEqual(b.state['actual_bytes'],0)
   finally:b.close()
 def test_blank_and_perceptual_duplicate_screen(self):
  from PIL import Image
  im=Image.new('RGB',(300,300),'white');s=signature(im)
  self.assertEqual(s['gray_std'],0);self.assertEqual(s['white_fraction'],1)
  self.assertEqual(duplicate(s,s),'exact')
  self.assertEqual(duplicate(dict(dhash64='0'),dict(dhash64='1')),'suspected_perceptual')
 def test_prepared_manifest_and_rights_are_consistent(self):
  m=read(MANIFEST);self.assertLessEqual(len(m['images']),400);self.assertFalse(DEST.exists())
  self.assertTrue(isolation(m['images'],set(read(OUT/'protected-identities.json'))))
  for r in m['images']:
   self.assertEqual(sha(ROOT/r['license_record']),r['license_record_sha256']);self.assertTrue(r['license_verified']);self.assertIsNone(r['image_sha256'])
   for g in r['annotations']:box(g['box'])
   self.assertEqual(r['image_url'],image_url(r));self.assertLessEqual(r['remote_object']['bytes'],IMAGE_MAX)
 def test_no_training_without_separate_approval(self):
  from training import training_gate
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)
   for name in ['config','approval','lock']:write_new(p/(name+'.json'),{})
   with self.assertRaises(PermissionError):training_gate(p/'approval.json',p/'config.json',p/'lock.json')
 def test_variable_size_training_schedule_excludes_development(self):
  from training import batches,update_schedule
  rows=[dict(image_id=str(i),split='train') for i in range(401)]
  schedule=update_schedule(rows)
  self.assertEqual(len(schedule),500)
  self.assertEqual(sum(len(r['images']) for r in schedule[:101]),401)
  self.assertEqual(schedule[100]['loss_normalizer'],1)
  self.assertEqual(schedule,update_schedule(rows))
  with self.assertRaises(ValueError):batches(rows+[dict(image_id='holdout',split='development')],0)
  with self.assertRaises(ValueError):update_schedule(rows,epochs=11)
 def test_download_requires_two_attempt_cap_and_no_redirect(self):
  from network import NoRedirect
  self.assertIsNone(NoRedirect().redirect_request(None,None,302,'',{},'https://evil.com'))
  with tempfile.TemporaryDirectory() as tmp:
   b=Budget(Path(tmp)/'run',cap=30)
   try:
    for i in range(2):
     with patch.object(b.opener,'open',return_value=Response(fail=True)):
      with self.assertRaises(OSError):b.request('https://www.flickr.com/photos/a/1',f'partial-{i}',10,'rights')
    with self.assertRaises(RuntimeError):b.request('https://www.flickr.com/photos/a/1','third',10,'rights')
   finally:b.close()
 def test_original_and_phase5c_checkpoint_hashes(self):
  from training import checkpoint_spec
  self.assertEqual(checkpoint_spec('phase5c')['delta_sha256'],SELECTED_SHA)
  with self.assertRaises(ValueError):checkpoint_spec('lora')

if __name__=='__main__':unittest.main()
