import copy
import unittest
from audit_core import *
spec=importlib.util.spec_from_file_location('phase8b_identity_under_test',HERE/'finalize.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
identity_decisions=module.identity_decisions
category_counts=module.category_counts

class Union:
 def __init__(self):self.parent={}
 def find(self,x):
  self.parent.setdefault(x,x)
  if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
  return self.parent[x]
 def join(self,a,b):self.parent[self.find(a)]=self.find(b)

class IdentityTests(unittest.TestCase):
 def fixtures(self):
  rows=[dict(coco_image_id=i,original_source_asset_url=f'https://source.example/{i}') for i in range(1,4)]
  records=[dict(coco_image_id=i,source_photo_id=f'flickr-photo:{i}',provisional_partition='train',licensing_status=STATUSES[0],image_object_status='available',
    creator_aliases=[f'flickr-account:{i}'],image_object={'etag':str(i),'bytes':i}) for i in range(1,4)]
  return rows,records
 def test_cross_partition_creator_group_quarantined_not_reassigned(self):
  rows,records=self.fixtures();records[1]['creator_aliases']=records[0]['creator_aliases'];records[1]['provisional_partition']='development'
  d,_,_=identity_decisions(rows,records,Union(),set());byid={r['coco_image_id']:r for r in d}
  self.assertTrue(byid[1]['quarantined']);self.assertTrue(byid[2]['quarantined']);self.assertEqual(byid[2]['provisional_partition'],'development');self.assertTrue(byid[3]['eligible_for_acquisition_proposal'])
 def test_excluded_bridge_still_connects_identity_family(self):
  rows,records=self.fixtures();records[0]['creator_aliases']=['A'];records[1]['creator_aliases']=['A','B'];records[1]['licensing_status']=STATUSES[2];records[2]['creator_aliases']=['B'];records[2]['provisional_partition']='development'
  d,_,_=identity_decisions(rows,records,Union(),set());self.assertTrue(all(r['quarantined'] for r in d))
 def test_protected_roots_recomputed_after_merging(self):
  rows,records=self.fixtures();uf=Union();uf.find('flickr-account:1')
  d,_,_=identity_decisions(rows,records,uf,{'flickr-account:1'});r=next(r for r in d if r['coco_image_id']==1);self.assertIn('historical_source_or_creator_group',r['identity_flags'])
 def test_equal_head_not_claimed_as_proven_pixel_duplicate(self):
  rows,records=self.fixtures();records[1]['image_object']=records[0]['image_object'];d,_,suspected=identity_decisions(rows,records,Union(),set());self.assertEqual(suspected,[[1,2]]);self.assertTrue(all(r['quarantined'] for r in d if r['coco_image_id'] in [1,2]))
 def test_source_url_duplicates_quarantined(self):
  rows,records=self.fixtures();rows[1]['original_source_asset_url']=rows[0]['original_source_asset_url'];d,duplicates,_=identity_decisions(rows,records,Union(),set());self.assertEqual(duplicates,[[1,2]])
 def test_unverified_rights_never_eligible(self):
  rows,records=self.fixtures();records[0]['licensing_status']=STATUSES[1];d,_,_=identity_decisions(rows,records,Union(),set());self.assertFalse(next(r for r in d if r['coco_image_id']==1)['eligible_for_acquisition_proposal'])
 def test_native_counts_do_not_merge_lampshades(self):
  rows=[dict(original_lvis_furniture_annotations=[dict(category_id=626,segmentation=[[1,2,3,4,5,6]]),dict(category_id=628,segmentation=[[1,2,3,4,5,6]])])]
  d=category_counts(rows);self.assertEqual(d['lamp']['instances'],1);self.assertEqual(d['lamp']['images'],1);self.assertEqual(d['desk']['images'],0)

if __name__=='__main__':unittest.main()
