import hashlib
import io
import pathlib
import tempfile
import unittest
from PIL import Image
import download as d
import screen as s

class AcquisitionTests(unittest.TestCase):
 def test_decodable_file_etag_hash_checked(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);file=root/'test.png';Image.new('RGB',(128,128),'red').save(file);body=file.read_bytes()
   class Ledger:
    dest=root
    def commit(self):pass
   record={'sha256':hashlib.sha256(body).hexdigest()};remote={'bytes':len(body),'etag':'"'+hashlib.md5(body).hexdigest()+'"'}
   d.downloaded_integrity(Ledger(),{'destination':'test.png'},record,remote);self.assertTrue(record['etag_md5_verified']);self.assertTrue(record['full_decode_verified'])
   with self.assertRaises(ValueError):d.downloaded_integrity(Ledger(),{'destination':'test.png'},record,{**remote,'etag':'"'+'0'*32+'"'})
 def test_candidate_states_are_provisional(self):
  self.assertEqual(s.classify_candidate([],[],True,{'metadata_label_agreement':True})[0],'provisional_accepted')
  self.assertEqual(s.classify_candidate(['blur'],[],True,{})[0],'rejected_quality')
  self.assertEqual(s.classify_candidate([],['overlap'],True,{})[0],'quarantined_duplicate_or_identity_risk')
  self.assertEqual(s.classify_candidate([],[],False,{'metadata_label_agreement':True})[0],'unresolved_low_margin')
  self.assertEqual(s.classify_candidate([],[],False,{'metadata_label_agreement':False})[0],'rejected_model_disagreement')

if __name__=='__main__':unittest.main(verbosity=2)
