import copy
import tempfile
import unittest
from common import *
from audit import allowed_license,flickr_id,box_status,segmentation_status,rank_select,group_split
from streaming import entries

class Tests(unittest.TestCase):
 def parse(self,data,chunk=7):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'a.json';p.write_text(data);return list(entries(p,chunk=chunk))
 def test_streamed_arrays_match_source_values(self):
  doc={'info':{'year':2017},'images':[{'name':'a "quoted" unicode é and long value'*20,'n':1.234e-20}],'annotations':[1,2,123456789],'categories':[],'licenses':[]}
  result=self.parse(json.dumps(doc),chunk=3)
  self.assertEqual(result,[('info',doc['info']),('images',doc['images'][0]),('annotations',1),('annotations',2),('annotations',123456789)])
 def test_streaming_rejects_invalid_json(self):
  for value in ['{"images":[1,]}','{"images":[1]} garbage','{"images":[],"images":[]}','{"images":[NaN]}','{"info":1,}']:
   with self.assertRaises(ValueError):self.parse(value)
 def test_licence_allowlist_not_blanket_clearance(self):
  self.assertTrue(allowed_license('http://creativecommons.org/licenses/by/2.0/'));self.assertTrue(allowed_license('https://creativecommons.org/publicdomain/zero/1.0/'))
  for value in ['https://creativecommons.org/licenses/by-nc/2.0/','https://creativecommons.org/licenses/by-sa/2.0/','http://flickr.com/commons/usage/','https://creativecommons.org:44/licenses/by/2.0/','https://evil/licenses/by/2.0/']:self.assertFalse(allowed_license(value))
 def test_original_photo_id_extraction(self):
  self.assertEqual(flickr_id('http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg'),'8119368305')
  self.assertEqual(flickr_id('https://www.flickr.com/photos/owner/123/'),'123')
  self.assertIsNone(flickr_id('https://example.com/123_x.jpg'))
 def test_box_quality_no_clipping(self):
  im={'width':100,'height':200}
  self.assertEqual(box_status({'bbox':[0,0,100,200]},im),'valid')
  self.assertEqual(box_status({'bbox':[0,0,0,1]},im),'nonpositive_extent')
  self.assertEqual(box_status({'bbox':[-1,0,100,200]},im),'outside_image')
 def test_polygons_and_crowd_rle_are_distinct(self):
  im={'width':2,'height':2}
  self.assertEqual(segmentation_status({'segmentation':[[0,0,1,0,1,1]]},im),'polygon')
  self.assertEqual(segmentation_status({'segmentation':{'size':[2,2],'counts':[1,2,1]}},im),'rle_uncompressed')
  self.assertEqual(segmentation_status({'segmentation':{'size':[2,2],'counts':[1,2]}},im),'invalid')
  self.assertEqual(segmentation_status({'segmentation':[]},im),'missing')
 def test_selection_reproducible_and_known_groups_unique(self):
  rows=[dict(coco_image_id=i,identity_group=str(i//2),usable_category_ids=[list(MAP)[i%4]],indoor_evidence_rank=3,scene_context_category_count=4) for i in range(40)]
  a=rank_select(rows,12);b=rank_select(list(reversed(rows)),12)
  self.assertEqual(a,b);self.assertEqual(len({r['identity_group'] for r in a}),12)
  splits=group_split(a);self.assertTrue(all(r['proposed_split'] in ('train','development') for r in splits))
 def test_actual_candidate_manifest_is_not_acquisition_permission(self):
  path=OUT/'proposed-image-candidates-v1.json'
  if not path.exists():self.skipTest('Audit not completed yet')
  p=read(path);self.assertEqual(p['image_count'],400);self.assertEqual(p['proposed_partition_counts'],{'train':320,'development':80})
  self.assertFalse(p['ready_for_photograph_acquisition']);self.assertEqual(p['individually_verified_rights'],0)
  self.assertTrue(all(not r['license_verified'] and r['creator'] is None and r['image_sha256'] is None and r['source_partition']=='train2017' for r in p['images']))
  self.assertEqual(len({r['identity_group'] for r in p['images']}),400)
 def test_actual_counts_and_extracted_hashes(self):
  if not (OUT/'metadata-audit.json').exists():self.skipTest('Audit not completed yet')
  a=read(OUT/'metadata-audit.json');self.assertEqual(a['source_partitions']['train2017']['images'],118287);self.assertEqual(a['source_partitions']['val2017']['images'],5000)
  self.assertEqual(a['source_partitions']['train2017']['annotations'],860001);self.assertEqual(a['source_partitions']['val2017']['annotations'],36781)
  self.assertEqual(a['annotation_issue_count'],0)
  for r in read(OUT/'extraction-complete.json')['members']:self.assertEqual(sha(OUT/r['file']),r['sha256'])

if __name__=='__main__':unittest.main()
