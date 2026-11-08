import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from audit_core import *
from transport import Transport
from test_safety import Response

class Boundaries(unittest.TestCase):
 def transport(self):
  t=Transport.__new__(Transport);t.state=dict(candidates={},requests=[],circuits={},conservative_charge=0,actual_body_bytes=0,last_source_utc=None);t.save=lambda:None
  return t
 def test_403_stops_provider_not_just_429(self):
  t=self.transport();r=manifest()['images'][0];response=Response();response.status=403
  t.opener=SimpleNamespace(open=lambda *a,**k:response);t.begin(r)
  with patch('transport.space'):t.request(r,r['source_page_resolver'],'HEAD')
  self.assertTrue(t.circuit_open());self.assertEqual(t.state['actual_body_bytes'],0)
  with self.assertRaises(HostSuspended):t.begin(manifest()['images'][1])
 def test_TLS_failure_stops_with_no_claim_of_verified_TLS(self):
  t=self.transport();r=manifest()['images'][0];t.begin(r)
  def fail(*a,**k):raise OSError('certificate verification failed fixture')
  t.opener=SimpleNamespace(open=fail)
  with patch('transport.space'),self.assertRaises(OSError):t.request(r,r['source_page_resolver'],'HEAD')
  self.assertTrue(t.circuit_open());self.assertFalse(t.state['requests'][0]['tls_verified'])
 def test_body_cap_reserved_before_request(self):
  t=self.transport();r=manifest()['images'][0];t.begin(r);t.state['conservative_charge']=CAP-PAGE+1
  t.opener=SimpleNamespace(open=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden')))
  with patch('transport.space'),self.assertRaises(ResourceLimit):t.request(r,r['source_page_resolver'],'GET')
  self.assertFalse(t.state['requests'])
 def test_single_attempt_no_same_transaction_retry(self):
  t=self.transport();r=manifest()['images'][0];t.begin(r);response=Response();response.status=200
  t.opener=SimpleNamespace(open=lambda *a,**k:response)
  with patch('transport.space'):t.request(r,r['source_page_resolver'],'HEAD')
  with self.assertRaises(ValueError):t.request(r,r['source_page_resolver'],'HEAD')

if __name__=='__main__':unittest.main()
