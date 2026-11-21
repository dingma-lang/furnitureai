"""Synthetic-only edit ledger, leakage and category-mask regression fixtures."""
import unittest,copy,json,tempfile,threading
from unittest.mock import patch
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from review2 import *
from dataset2 import compile_dataset,category_masks
from test_g5 import fixture_registry,submission

def fixture_payload(reg,negative=False,index=0):
    p=submission(reg,negative,index);p['schema']=SCHEMA;p['history']=[];p['draft_geometry']={}
    r=p['reviews'][0];p['history'].append(event(reg,'review',r['task_id'],None,r))
    return p

def event(reg,kind,key,before,after,sequence=1,photo_key=None):
    pk=photo_key or next(t['photo_key'] for t in reg['tasks'] if t['task_id']==key);p=next(p for p in reg['photos'] if p['photo_key']==pk)
    return dict(event_id=uuid.uuid4().hex,sequence=sequence,at='2026-11-20T20:00:00Z',action='SYNTHETIC test action',kind=kind,key=key,photo_key=pk,image_id=p['image_id'],sha256=p['sha256'],before=copy.deepcopy(before),after=copy.deepcopy(after))

def rows(reg):
    return [dict(image_id=p['image_id'],sha256=p['sha256'],identity_group=p['identity_group'],historical_partition='train',internal_partition='optimization' if p['role']=='optimization' else 'inner_development',annotations=[dict(label='chair',box=[.7,.1,.9,.4],ignore=False,native_dataset='COCO')],explicit_native_negative={},coverage={}) for p in reg['photos']]

class ReviewHistoryTests(unittest.TestCase):
    def setUp(self):self.reg=fixture_registry();self.p=fixture_payload(self.reg)
    def test_valid_ledger(self):self.assertEqual(validate(self.p,self.reg)['history_events'],1)
    def test_no_evidence_from_other_object(self):self.assertEqual(validate(self.p,self.reg)['accepted_hard_negatives'],0)
    def test_absence_requires_explicit_action(self):self.assertEqual(validate(fixture_payload(self.reg,True),self.reg)['accepted_hard_negatives'],1)
    def test_history_required_for_decisions(self):
        self.p['history']=[]
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_replayed_before_state(self):
        self.p['history'][0]['before']={'fake':'prior'}
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_tampered_final_decision(self):
        self.p['reviews'][0]['decision']='uncertain'
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_history_photo_sha_binding(self):
        self.p['history'][0]['sha256']='altered'
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_duplicate_history_id(self):
        self.p['history'].append(dict(self.p['history'][0],sequence=2))
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_past_invalid_human_decision_is_rejected(self):
        self.p['history'][0]['after']['reviewer']='';self.p['reviews']=[];self.p['history'].append(event(self.reg,'review','task0',self.p['history'][0]['after'],None,2))
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_undo_is_additive_history(self):
        r=self.p['reviews'].pop();self.p['history'].append(event(self.reg,'review','task0',r,None,2));v=validate(self.p,self.reg);self.assertEqual(v['reviewed'],0);self.assertEqual(v['history_events'],2)
    def test_validation_never_becomes_negative(self):
        with self.assertRaises(PermissionError):validate(fixture_payload(self.reg,True,2),self.reg)
    def test_geometry_must_replay(self):
        self.p['draft_geometry']={'task0':[.1,.1,.2,.2]}
        with self.assertRaises(ValueError):validate(self.p,self.reg)
    def test_old_confirmation_cannot_survive_edit(self):
        p=fixture_payload(self.reg,True);p['draft_geometry']={'task0':[.1,.1,.2,.2]};p['history'].append(event(self.reg,'geometry','task0',None,p['draft_geometry']['task0'],2))
        with self.assertRaises(ValueError):validate(p,self.reg)

class DatasetTests(unittest.TestCase):
    def setUp(self):self.reg=fixture_registry();self.p=fixture_payload(self.reg);self.rows=rows(self.reg)
    def compile(self,p=None,rs=None):return compile_dataset(p or self.p,self.reg,rows=rs or self.rows,inventory=dict(photos=[]))
    def test_preserve_positives_and_exclude_validation(self):
        d=self.compile();self.assertEqual((d['optimization_photos'],d['positive_boxes'],d['accepted_Chair_negatives']),(2,2,0));self.assertFalse(d['training_authorized']);self.assertTrue(all(r['internal_partition']=='optimization' for r in d['rows']))
    def test_group_leakage_rejected(self):
        self.rows[-1]['identity_group']=self.rows[0]['identity_group']
        with self.assertRaises(ValueError):self.compile()
    def test_hash_leakage_rejected(self):
        self.rows[-1]['sha256']=self.rows[0]['sha256']
        with self.assertRaises(ValueError):self.compile()
    def test_unreviewed_unknown_has_no_mask(self):
        r=self.compile()['rows'][0];t,m=category_masks([[.1,.1,.2,.2]],r);self.assertFalse(any(m[0]))
    def test_category_specific_chair_only(self):
        d=self.compile(fixture_payload(self.reg,True));r=d['rows'][0];t,m=category_masks([[.12,.12,.2,.2]],r);self.assertEqual(m[0],[False,True,False,False,False,False,False]);self.assertEqual(t[0],[0]*7);self.assertEqual(d['accepted_negative_evidence'][0]['native_category'],62)
    def test_full_box_containment_not_center(self):
        r=self.compile(fixture_payload(self.reg,True))['rows'][0];t,m=category_masks([[.05,.05,.2,.2]],r);self.assertFalse(any(m[0]))
    def test_positive_wins(self):
        r=self.compile(fixture_payload(self.reg,True))['rows'][0];t,m=category_masks([[.12,.12,.2,.2]],r,[(0,'chair')]);self.assertEqual(t[0][1],1)
    def test_no_positive_without_native_support(self):
        with self.assertRaises(ValueError):category_masks([[.12,.12,.2,.2]],self.compile()['rows'][0],[(0,'lamp')])
    def test_ignored_regions_protected(self):
        r=self.compile(fixture_payload(self.reg,True))['rows'][0];r['source_ignored_boxes']=[dict(box=[.1,.1,.2,.2])];t,m=category_masks([[.12,.12,.2,.2]],r);self.assertFalse(any(m[0]))
    def test_reproducible(self):self.assertEqual(digest(self.compile()),digest(self.compile()))
    def test_unsafe_query_rejected(self):
        with self.assertRaises(ValueError):category_masks([[-1,0,1,1]],self.compile()['rows'][0])
    def test_legacy_targets_masks_preserved(self):
        r=self.compile(fixture_payload(self.reg,True))['rows'][0];base_t=[[1,0,0,0,0,0,0]];base_m=[[True,False,True,False,False,False,False]]
        t,m=category_masks([[.12,.12,.2,.2]],r,legacy_targets=base_t,legacy_mask=base_m)
        self.assertEqual(t,base_t);self.assertTrue(m[0][0] and m[0][1] and m[0][2]);self.assertFalse(base_m[0][1])
    def test_auxiliary_label_keeps_native_lineage_no_absence(self):
        self.p['reviews'][0]['source_label_confirmed']=True;self.p['history'][0]['after']['source_label_confirmed']=True
        inv=dict(photos=[dict(annotations=[dict(annotation_key='synthetic-object',metadata_candidate_eligible=True,native_category_id=1,native_label='Person')])])
        d=compile_dataset(self.p,self.reg,rows=self.rows,inventory=inv);a=d['rows'][0]['confirmed_named_source_objects'][0]
        self.assertEqual(a['lineages'][0]['native_category_id'],1);self.assertFalse(a['Chair_negative_implied']);self.assertFalse(a['training_activated']);self.assertEqual(d['accepted_Chair_negatives'],0)
    def test_unknown_auxiliary_native_key_rejected(self):
        self.p['reviews'][0]['source_label_confirmed']=True;self.p['history'][0]['after']['source_label_confirmed']=True
        with self.assertRaises(KeyError):self.compile()

class ServerTests(unittest.TestCase):
    def setUp(self):
        from server2 import make_server
        self.reg=fixture_registry();self.s=make_server(0,self.reg);self.t=threading.Thread(target=self.s.serve_forever,daemon=True);self.t.start();self.base=f'http://127.0.0.1:{self.s.server_port}'
    def tearDown(self):self.s.shutdown();self.s.server_close();self.t.join()
    def test_loopback_only(self):
        self.assertEqual(self.s.server_address[0],'127.0.0.1')
        with urlopen(self.base+'/api/registry') as r:self.assertEqual(json.load(r)['schema'],SCHEMA)
    def test_csrf_guard(self):
        with self.assertRaises(HTTPError) as cm:urlopen(Request(self.base+'/api/export',data=b'{}',headers={'Content-Type':'application/json'}))
        self.assertEqual(cm.exception.code,403);cm.exception.close()
    def test_progress_and_history_controls_present(self):
        with urlopen(self.base+'/') as r:b=r.read()
        self.assertIn(b'auto-next',b);self.assertIn(b'edit-history',b)

class ExportTests(unittest.TestCase):
    def test_atomic_completion_and_unique_export(self):
        import review2
        reg=fixture_registry();p=fixture_payload(reg)
        with tempfile.TemporaryDirectory() as temp,patch.object(review2,'RAW',Path(temp)),patch('dataset2.compile_dataset',return_value=dict(schema='SYNTHETIC',training_authorized=False)):
            a=review2.export(p,reg);b=review2.export(p,reg);self.assertNotEqual(a['export_id'],b['export_id'])
            folder=Path(temp)/'exports'/a['export_id'];self.assertEqual(read(folder/'human-review.json'),p);self.assertEqual(read(folder/'COMPLETE.json')['sha256sums'],sha(folder/'SHA256SUMS'))
            for line in (folder/'SHA256SUMS').read_text().splitlines():
                value,name=line.split('  ',1);self.assertEqual(value,sha(folder/name))

if __name__=='__main__':unittest.main()
