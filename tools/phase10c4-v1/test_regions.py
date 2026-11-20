import copy
import http.client
import json
import threading
import unittest
from core import *
from server import make_server

def fixture_registry():
    photos=[]
    for i in range(3):
        photos.append(dict(key=f'fixture-{i}',image_id=f'SYNTHETIC-{i}',sha256=f'fake-sha-{i}',identity_group=f'fake-group-{i}',
            width=1000,height=800,orientation=1,display_width=1000,display_height=800,native_annotation_digest='fake-annotations',
            annotations=[dict(label='chair',ignore=False,box_pixels=[100,100,300,400],native_dataset='OpenImages',native_id='/m/01mzpv')],
            restricted=[dict(box_pixels=[100,100,300,400],reason='known Chair'),dict(box_pixels=[700,100,900,250],reason='ignored native')],priority_scope_review=True))
    clar=[dict(item_id=f'fixture-review-{i}',partition='optimization' if i<3 else 'inner_development',decision='uncertain_defer',evidence='Synthetic old observation',clarification_flags=['Synthetic question']) for i in range(24)]
    return dict(schema='phase10c4-registry-v1',photos=photos,clarifications=clar,taxonomy=TAXONOMY,taxonomy_sha256=digest(TAXONOMY),source_manifest_sha256='fixture')

def session(reg):return dict(schema=SCHEMA,registry_sha256=digest(reg),taxonomy_sha256=digest(TAXONOMY),regions=[],clarifications=[])
def rectangle(p,ident='fixture-rect',box=None,status='draft',kind='negative'):
    box=box or [20,500,250,700]
    r={k:p[k] for k in ('image_id','sha256','identity_group','width','height','orientation','native_annotation_digest')}
    r.update(region_id=ident,photo_key=p['key'],box_pixels=box,revision=0,kind=kind,status=status,human_reviewed=False,confirmed=False,
        taxonomy_sha256=digest(TAXONOMY),historical_partition='train',internal_partition='optimization')
    if status=='confirmed_negative':r.update(human_reviewed=True,confirmed=True,review_covers_seating_stools=True,no_chair_stool_or_ambiguous_part=True,
        confirmed_revision=0,confirmed_box_pixels=box.copy(),reviewer='SYNTHETIC TEST FIXTURE',review_date='2026-11-19',evidence='Synthetic test rectangle, never a user submission')
    return r

class GeometryTests(unittest.TestCase):
    def test_all_exif_orientations_roundtrip_pixel_edges(self):
        for o in range(1,9):
            b=[13,17,799,599];d=transform_box(b,800,600,o);self.assertEqual(transform_box(d,800,600,o,True),b)
            dw,dh=display_size(800,600,o);self.assertTrue(0<=d[0]<d[2]<=dw and 0<=d[1]<d[3]<=dh)
    def test_pixel_boundaries_not_center_coordinates(self):
        for o in range(1,9):self.assertEqual(transform_box([0,0,800,600],800,600,o),[0,0,*display_size(800,600,o)])
    def test_overlap_touching_edges_is_not_overlap(self):
        self.assertFalse(overlaps([0,0,10,10],[10,0,20,10]));self.assertTrue(overlaps([0,0,10,10],[9,0,20,10]))
    def test_pixel_boxes_reject_floats_nan_reversed_or_outside(self):
        for b in ([0,0,10.5,10],[0,0,float('nan'),10],[10,0,0,10],[0,0,1001,10],[0,0,True,10]):self.assertRaises(ValueError,pixel_box,b,1000,800)
    def test_normalized_original_coordinates_exact(self):self.assertEqual(normalized([20,80,300,400],1000,800),[.02,.1,.3,.5])

class EvidenceTests(unittest.TestCase):
    def setUp(self):self.reg=fixture_registry();self.p=self.reg['photos'][0];self.s=session(self.reg)
    def test_empty_initial_state_never_manufactures_six(self):
        x=validate(self.s,self.reg);self.assertEqual(x['confirmed_negative_rectangles'],0);self.assertFalse(x['minimum_region_group_gate'])
    def test_draft_uncertain_rejected_never_enter_masks(self):
        for status in ('draft','uncertain','rejected'):
            self.s['regions']=[rectangle(self.p,status=status)];self.assertEqual(validate(self.s,self.reg)['accepted'],[])
    def test_confirmation_requires_flags_revision_and_bound_geometry(self):
        r=rectangle(self.p,status='confirmed_negative');self.s['regions']=[r];self.assertEqual(validate(self.s,self.reg)['confirmed_negative_rectangles'],1)
        for field,value in [('confirmed',False),('human_reviewed',False),('review_covers_seating_stools',False),('confirmed_revision',1),('confirmed_box_pixels',[0,0,1,1])]:
            self.s['regions']=[dict(r,**{field:value})];self.assertRaises(ValueError,validate,self.s,self.reg)
    def test_identity_dimensions_and_taxonomy_must_match(self):
        r=rectangle(self.p,status='confirmed_negative')
        for field,value in [('sha256','changed'),('identity_group','changed'),('width',999),('orientation',6),('native_annotation_digest','changed')]:
            self.s['regions']=[dict(r,**{field:value})];self.assertRaises(ValueError,validate,self.s,self.reg)
        self.s['taxonomy_sha256']='old';self.assertRaises(ValueError,validate,self.s,self.reg)
    def test_positive_and_ignored_overlap_block_confirmed_negative(self):
        for b in ([150,150,200,200],[750,150,800,200]):
            self.s['regions']=[rectangle(self.p,box=b,status='confirmed_negative')];v=validate(self.s,self.reg);self.assertEqual(v['accepted'],[]);self.assertEqual(len(v['blocked']),1)
    def test_new_uncertainty_blocks_already_confirmed_negative(self):
        self.s['regions']=[rectangle(self.p,status='confirmed_negative'),rectangle(self.p,'unknown',[30,550,100,650],kind='potential_positive')]
        self.assertEqual(validate(self.s,self.reg)['accepted'],[])
    def test_overlapping_negatives_not_double_counted(self):
        self.s['regions']=[rectangle(self.p,'a',status='confirmed_negative'),rectangle(self.p,'b',[30,550,100,650],status='confirmed_negative')]
        self.assertEqual(validate(self.s,self.reg)['accepted'],[])
    def test_full_photo_needs_separate_review(self):
        self.s['regions']=[rectangle(self.p,box=[0,0,1000,800],status='confirmed_negative')];self.assertRaises(ValueError,validate,self.s,self.reg)
    def test_six_regions_three_groups_gate_and_one_group_failure(self):
        self.s['regions']=[rectangle(p,f'{i}-{j}',[20+300*j,500,250+300*j,700],status='confirmed_negative') for i,p in enumerate(self.reg['photos']) for j in range(2)]
        v=validate(self.s,self.reg);self.assertEqual(v['confirmed_negative_rectangles'],6);self.assertEqual(v['distinct_source_groups'],3);self.assertTrue(v['minimum_region_group_gate']);self.assertFalse(v['training_authorized'])
    def test_validation_or_unknown_photo_is_denied(self):
        self.s['regions']=[dict(rectangle(self.p),photo_key='validation-photo')];self.assertRaises(PermissionError,validate,self.s,self.reg)
    def test_clarifications_separate_from_negative_counts(self):
        self.s['clarifications']=[dict(item_id='fixture-review-5',action='deferred',confirmed=True,reviewer='SYNTHETIC FIXTURE',review_date='2026-11-19',evidence='Synthetic clarification only')]
        v=validate(self.s,self.reg);self.assertEqual(v['confirmed_negative_rectangles'],0);self.assertEqual(v['clarification_responses'],1);self.assertEqual(v['unresolved_clarifications'],24)
    def test_coordinate_edits_cannot_reuse_old_confirmation(self):
        r=rectangle(self.p,status='confirmed_negative');r['box_pixels']=[21,500,250,700];self.s['regions']=[r];self.assertRaises(ValueError,validate,self.s,self.reg)

class ServerTests(unittest.TestCase):
    def setUp(self):
        self.reg=fixture_registry();self.srv=make_server(0,self.reg,lambda p,r:dict(export_id='SYNTHETIC_ONLY',validation=validate(p,r),mask_changed_terms=0))
        self.thread=threading.Thread(target=self.srv.serve_forever,daemon=True);self.thread.start();self.port=self.srv.server_port
        self.conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5);self.conn.request('GET','/api/registry');resp=self.conn.getresponse();self.data=json.loads(resp.read())
    def tearDown(self):self.conn.close();self.srv.shutdown();self.srv.server_close();self.thread.join()
    def request(self,path,body=None,headers=None):
        h={'Content-Type':'application/json','X-Annotation-Token':self.data['csrf_token']};h.update(headers or {})
        self.conn.request('POST' if body is not None else 'GET',path,json.dumps(body) if body is not None else None,h);r=self.conn.getresponse();data=r.read();return r.status,data
    def test_local_index_and_assets(self):
        for path in ('/','/app.js','/style.css'):self.assertEqual(self.request(path)[0],200)
    def test_foreign_origins_host_and_missing_token_denied(self):
        for h in ({'Origin':'https://example.com'},{'X-Annotation-Token':'wrong'},{'Host':'example.com'}):self.assertEqual(self.request('/api/validate',session(self.reg),h)[0],403)
    def test_traversal_and_model_training_routes_absent(self):
        for path in ('/../../models/config.json','/api/photo/../secret','/api/train','/api/infer'):self.assertEqual(self.request(path)[0],404)
    def test_validate_partial_and_mock_export_never_train(self):
        code,body=self.request('/api/validate',session(self.reg));self.assertEqual(code,200);self.assertFalse(json.loads(body)['training_authorized'])
        code,body=self.request('/api/export',session(self.reg));self.assertEqual(code,200);self.assertEqual(json.loads(body)['export_id'],'SYNTHETIC_ONLY')
    def test_bad_registry_payload_is_rejected(self):self.assertEqual(self.request('/api/export',{'schema':'wrong'})[0],422)

class ActualRegistryTests(unittest.TestCase):
    def test_actual_registry_only59_optimization_photos_and_zero_reviews(self):
        r=registry();self.assertEqual(len(r['photos']),59);self.assertEqual(len({p['identity_group'] for p in r['photos']}),56)
        self.assertTrue(all(p['internal_partition']=='optimization' for p in r['photos']));self.assertEqual(len(r['clarifications']),24)
        self.assertEqual(validate(read(RAW/'empty-session.json'),r)['confirmed_negative_rectangles'],0)
    def test_no_real_human_export_manufactured(self):self.assertFalse((RAW/'exports').exists())

class ExportFixtureTests(unittest.TestCase):
    def fixture(self):
        reg=fixture_registry();rows=[]
        for p in reg['photos']:
            anns=[dict(label='chair',ignore=False,box=[.1,.125,.3,.5],native_id='/m/01mzpv',native_dataset='OpenImages')]
            p['native_annotation_digest']=digest(anns)
            rows.append(dict(image_id=p['image_id'],sha256=p['sha256'],identity_group=p['identity_group'],historical_partition='train',internal_partition='optimization',
                width=1000,height=800,annotations=anns,explicit_native_negative={c:False for c in ('sofa','chair','table','bed','desk','cabinet','lamp')}))
        s=session(reg);s['regions']=[rectangle(p,f'{i}-{j}',[20+300*j,500,250+300*j,700],status='confirmed_negative') for i,p in enumerate(reg['photos']) for j in range(2)]
        return reg,dict(rows=rows),s
    def test_synthetic_export_produces_only_regional_masks_and_immutable_versions(self):
        import tempfile
        from unittest.mock import patch
        import exporter
        reg,m,s=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);destination=root/'exports/SYNTHETIC-v1'
            with patch.object(exporter,'RAW',root),patch.object(exporter,'manifest',lambda:m),patch.object(exporter,'photo_path',lambda p:root):
                result=exporter.save_export(s,reg,destination)
                self.assertEqual(result['validation']['confirmed_negative_rectangles'],6);self.assertGreater(result['mask_changed_terms'],0)
                overlay=read(destination/'supervision-manifest.json');self.assertFalse(overlay['training_authorized']);self.assertTrue(overlay['rest_of_images_ignored_for_new_negatives'])
                self.assertEqual(len(overlay['accepted_negative_evidence']),3)
                self.assertRaises(FileExistsError,exporter.save_export,s,reg,destination)
                # Source labels were not rewritten; only additional regional evidence.
                self.assertTrue(all(len(r['annotations'])==1 for r in m['rows']))
    def test_blocked_export_does_not_write_any_human_evidence(self):
        import tempfile
        from unittest.mock import patch
        import exporter
        reg,m,s=self.fixture();s['regions'][0]['box_pixels']=[100,100,200,200];s['regions'][0]['confirmed_box_pixels']=[100,100,200,200]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(exporter,'RAW',root):
                self.assertRaises(ValueError,exporter.save_export,s,reg,root/'exports/SYNTHETIC-v1');self.assertFalse((root/'exports').exists())

if __name__=='__main__':unittest.main()
