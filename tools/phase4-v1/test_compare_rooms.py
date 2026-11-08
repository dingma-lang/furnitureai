import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import compare_rooms as p

class RoomTests(unittest.TestCase):
 def checkpoint(self,path,classes=p.CLASSES):
  value={'classes':classes,'config':{'kind':'cosine'},'feature_model_fingerprint':'fixture','state_dict':{'weight':torch.eye(7,768)}}
  torch.save(value,path);return p.sha(path)
 def test_checkpoint_hash_mapping_and_fingerprint(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'head.pt';h=self.checkpoint(path);head=p.load_head(path,h,'fixture');self.assertEqual(head['weight'].shape,(7,768))
   with self.assertRaises(ValueError):p.load_head(path,'wrong','fixture')
   with self.assertRaises(ValueError):p.load_head(path,h,'wrong_space')
   h=self.checkpoint(path,tuple(reversed(p.CLASSES)))
   with self.assertRaises(ValueError):p.load_head(path,h,'fixture')
 def test_actual_checkpoint_logits_match_training_interface(self):
  fingerprint=p.read(p.MODEL/'checkpoint-manifest.json')['model_processor_fingerprint']
  generator=np.random.default_rng(20261105);features=p.w.unit(generator.standard_normal((5,768)).astype(np.float32))
  for arm,path in [('baseline',p.BASELINE),('expanded',p.SELECTED)]:
   head=p.load_head(path,p.EXPECTED[arm],fingerprint);saved=torch.load(path,map_location='cpu',weights_only=True)
   original=p.w.network(saved['config'],768);original.load_state_dict(saved['state_dict']);original.eval()
   with torch.inference_mode():expected=original(torch.from_numpy(features)).numpy()
   actual=np.asarray([r['all_logits'] for r in p.predict(features,head)])
   self.assertTrue(np.allclose(actual,expected,atol=1e-6,rtol=0))
 def test_original_checkpoint_not_modified_on_load_or_predict(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'head.pt';h=self.checkpoint(path);before=path.read_bytes();head=p.load_head(path,h,'fixture');p.predict(torch.eye(7,768),head);self.assertEqual(path.read_bytes(),before)
 def test_comparison_mapping_and_disagreement(self):
  x=torch.eye(7,768);left={'weight':torch.eye(7,768)};right={'weight':torch.eye(7,768).roll(1,0)}
  a=p.predict(x,left);b=p.predict(x,right)
  self.assertEqual([r['category'] for r in a],list(p.CLASSES));self.assertTrue(all(l['category']!=r['category'] for l,r in zip(a,b)))
  for row in a:self.assertAlmostEqual(sum(row['all_softmax7']),1,places=5);self.assertFalse(row['calibrated'])
 def test_invalid_embeddings_fail_not_fake_predictions(self):
  with self.assertRaises(ValueError):p.predict(torch.zeros(1,768),{'weight':torch.eye(7,768)})
 def test_same_crop_shared_preprocessing_features_and_oom_recovery(self):
  class Processor:
   calls=0
   def __call__(self,images,return_tensors):
    self.calls+=1;return {'pixel_values':torch.tensor(np.stack([np.asarray(i,dtype=np.float32).transpose(2,0,1) for i in images]))}
  class Model:
   seen=[]
   def get_image_features(self,pixel_values):
    self.seen.append(pixel_values.clone())
    if len(pixel_values)>1:raise torch.cuda.OutOfMemoryError('synthetic batch OOM')
    return torch.ones(len(pixel_values),768)
  processor=Processor();model=Model();images=[Image.new('RGB',(8,8),'red'),Image.new('RGB',(8,8),'blue')]
  with patch.object(p.torch.cuda,'empty_cache'):
   vectors,proof,events=p.embed_shared(images,model,processor,'cpu',2)
  a=p.predict(vectors,{'weight':torch.eye(7,768)});b=p.predict(vectors,{'weight':torch.eye(7,768)})
  self.assertEqual(a,b);self.assertEqual(len(events),1);self.assertTrue(all(r['same_embedding_for_both_classifiers'] for r in proof));self.assertEqual(vectors.shape,(2,768))
 def test_generic_inference_error_not_suppressed(self):
  with self.assertRaisesRegex(RuntimeError,'bad kernel'):p.gpu_retry(lambda:(_ for _ in ()).throw(RuntimeError('bad kernel')))
 def test_oom_retry_bounded(self):
  calls=[]
  def f():
   calls.append(1)
   if len(calls)==1:raise torch.cuda.OutOfMemoryError('fixture')
   return 42
  with patch.object(p.torch.cuda,'empty_cache'):self.assertEqual(p.gpu_retry(f),42)
  self.assertEqual(len(calls),2)
 def test_original_boxes_preserved_by_visualization(self):
  image=Image.new('RGB',(100,60),'white');rows=[{'detection_id':'d000','original_box':[1.2,2.3,89.1,59.],'detector_label':'chair','detector_score':.3,'classification_status':'unavailable'}];old=copy.deepcopy(rows)
  with tempfile.TemporaryDirectory() as tmp:p.draw_comparison(image,rows,Path(tmp)/'comparison.png')
  self.assertEqual(rows,old)
 def test_inputs_are_authorized_existing_rooms_not_official_holdout(self):
  rows=p.inputs_from_manifest(p.MANIFEST);self.assertEqual(len(rows),3);self.assertTrue(all(not r['prior_annotations_used'] for r in rows))
  with tempfile.TemporaryDirectory() as tmp:
   file=Path(tmp)/'rooms.json';source=p.read(p.MANIFEST);source['images'][0]['file']=str(p.ROOT/'data/abo/any.jpg');file.write_text(json.dumps(source))
   with self.assertRaises(ValueError):p.inputs_from_manifest(file)
 def test_unapproved_or_corrupted_source_rejected(self):
  source=p.read(p.MANIFEST);source['images']=source['images'][:1];source['images'][0]['file']=str(p.ROOT/'data/room-validation/nathan-fertig-FBXuXp57eM0.jpg')
  with tempfile.TemporaryDirectory() as tmp:
   file=Path(tmp)/'rooms.json';source['images'][0]['image_sha256']='wrong';file.write_text(json.dumps(source))
   with self.assertRaises(ValueError):p.inputs_from_manifest(file)
   source['images'][0]['rights']['allowed_for_benchmark']=False;file.write_text(json.dumps(source))
   with self.assertRaises(ValueError):p.inputs_from_manifest(file)
 def test_existing_output_not_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'existing';path.mkdir();sentinel=path/'predictions.json';sentinel.write_text('historical')
   with self.assertRaises(FileExistsError):p.initialize(path,p.MANIFEST,False)
   self.assertEqual(sentinel.read_text(),'historical')
 def test_no_photograph_mutation_when_loading_or_cropping(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'room.png';Image.new('RGB',(20,20),'red').save(path);before=p.sha(path);image=p.load_image(path);bounds=p.crop_bounds([1.1,2.2,18.2,19.2],20,20);image.crop(bounds);self.assertEqual(p.sha(path),before)

if __name__=='__main__':unittest.main(verbosity=2)
