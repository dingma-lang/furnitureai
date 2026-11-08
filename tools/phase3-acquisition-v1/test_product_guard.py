import unittest
from product_guard import guard

class GuardTests(unittest.TestCase):
 def test_reference_risk_quarantines_whole_product_without_mutating_input(self):
  rows=[{'image_id':'a','product_id':'p','status':'quarantined_duplicate_or_identity_risk'}, {'image_id':'b','product_id':'p','status':'provisional_accepted'}, {'image_id':'c','product_id':'q','status':'provisional_accepted'}]
  pair={'left_product_id':'p','right_product_id':'reserved','reference_origin':'official_holdout_reference'}
  result,risks=guard(rows,[pair]);self.assertIn('p',risks);self.assertEqual(result[1]['status'],'quarantined_duplicate_or_identity_risk');self.assertEqual(result[2]['status'],'provisional_accepted');self.assertEqual(rows[1]['status'],'provisional_accepted')
 def test_cross_new_product_suspects_quarantine_both_identities(self):
  rows=[{'image_id':'a','product_id':'p','status':'provisional_accepted'},{'image_id':'b','product_id':'q','status':'provisional_accepted'}]
  result,risks=guard(rows,[{'left_product_id':'p','right_product_id':'q'}]);self.assertEqual(set(risks),{'p','q'});self.assertTrue(all(r['status']!='provisional_accepted' for r in result))

if __name__=='__main__':unittest.main(verbosity=2)
