"""Small synthetic intake tests, not another broad code audit or model experiment."""
import copy
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from html.parser import HTMLParser

from intake15 import validate,validate_source,CATEGORIES,SCHEMA,TAXONOMY,FRAME
from serve15 import Handler


def record():
    return dict(schema=SCHEMA,taxonomy_version=TAXONOMY,photo_ids=['synthetic-photo'],
        creator_id='synthetic-creator',source_group_id='synthetic-source',capture_group_id='synthetic-session',
        room_group_ids=['synthetic-room'],credit='Synthetic test only, not a real grant',source='synthetic generated fixture',
        purpose='training',permission_evidence='synthetic-only-evidence',permission_evidence_sha256='a'*64,
        permission_date='2026-11-24',explicit_confirmation=True,privacy_and_third_party_rights_resolved=True,
        scopes={k:True for k in ('storage','annotation_and_modification','model_training_or_evaluation','commercial_model_distribution')},
        orientation_explicitly_checked=True,photo=dict(image_sha256='b'*64,width=100,height=80,bytes=100,box_frame=FRAME),
        annotations=[],class_completeness={c:dict(status='unknown') for c in CATEGORIES})


class IntakeTests(unittest.TestCase):
    def test_complete_metadata_is_not_a_rights_or_training_approval(self):
        r=validate(record());self.assertEqual(r['errors'],[])
        for k in ('rights_verified','group_independence_verified','annotations_active','eligible_for_training','eligible_for_independent_evaluation'):
            self.assertFalse(r[k])

    def test_grant_scope_evidence_and_confirmation_are_required(self):
        for key in ('explicit_confirmation','permission_evidence_sha256','privacy_and_third_party_rights_resolved'):
            r=record();r.pop(key);self.assertTrue(validate(r)['errors'],key)
        r=record();r['scopes']['commercial_model_distribution']=False;self.assertTrue(validate(r)['errors'])

    def test_all_categories_have_unknown_completeness_by_default(self):
        self.assertEqual(len(record()['class_completeness']),7)
        self.assertFalse(validate(record())['complete_seven_class_review'])

    def test_invalid_box_and_fake_confirmation_rejected(self):
        r=record();r['annotations']=[dict(id='a',category='chair',box=[0,0,200,20],decision='confirmed')]
        result=validate(r);self.assertIn('Invalid annotation box',result['errors'])
        self.assertIn('Confirmed object requires explicit human evidence',result['errors'])

    def test_unreviewed_source_boxes_do_not_become_verified(self):
        r=record();source=dict(schema='modern-room-source-annotations-v1',source_id='synthetic-source',source_file_sha256='c'*64,
            annotation_license='synthetic-only',**{k:r['photo'][k] for k in ('image_sha256','width','height','box_frame')},
            annotations=[dict(id='native-1',box=[1,2,20,30],native_label='chair',native_annotation={'id':1},suggested_category='chair')])
        self.assertEqual(validate_source(source,r['photo'])[0]['native_annotation'],{'id':1})
        r['source_annotations']=source;r['annotations']=[dict(source['annotations'][0],origin='source_proposal',source_file_sha256='c'*64,decision='proposal',category='chair')]
        self.assertEqual(validate(r)['confirmed_object_records'],0)
        source['image_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'exact oriented'):validate_source(source,r['photo'])
        self.assertTrue(validate(r)['errors'])

    def test_source_annotation_ids_and_native_provenance_required(self):
        r=record();source=dict(schema='modern-room-source-annotations-v1',source_id='source',source_file_sha256='c'*64,
            annotation_license='synthetic',**{k:r['photo'][k] for k in ('image_sha256','width','height','box_frame')},
            annotations=[dict(id='1',box=[0,0,2,3],native_label='chair',native_annotation={})]*2)
        with self.assertRaisesRegex(ValueError,'Unique source'):validate_source(source,r['photo'])

    def test_cross_purpose_capture_creator_and_room_group_conflicts(self):
        old=record();old['purpose']='independent_evaluation'
        self.assertTrue(any('crosses purposes' in e for e in validate(record(),[old])['errors']))
        r=record();r['capture_group_id']=[];self.assertTrue(validate(r)['errors'])

    def test_historical_hashes_and_groups_prevent_new_independent_use(self):
        r=record();result=validate(r,exclusions=dict(sha256=['b'*64],source_groups=['synthetic-room']))
        self.assertIn('Historical/exposed image hash conflict',result['errors'])
        self.assertIn('Historical/exposed group conflict',result['errors'])

    def test_independent_review_must_be_complete_and_unexposed(self):
        r=record();r['purpose']='independent_evaluation'
        self.assertTrue(validate(r)['errors'])
        r['exposure_status']='not_model_exposed_attested'
        for c in CATEGORIES:r['class_completeness'][c]=dict(status='reviewed_absent',reviewer='Synthetic reviewer',reviewed_at='2026-11-24',evidence='Synthetic fixture inspection only')
        result=validate(r);self.assertEqual(result['errors'],[])
        self.assertTrue(result['complete_seven_class_review']);self.assertFalse(result['eligible_for_independent_evaluation'])
        r['annotations']=[dict(id='p',box=[1,1,20,20],category='chair',decision='confirmed',explicit_human_confirmation=True,
            reviewer='Synthetic reviewer',reviewed_at='2026-11-24',evidence='Synthetic object')]
        self.assertIn('Absent category contradicts confirmed positive: chair',validate(r)['errors'])

    def test_actual_image_hash_and_EXIF_dimensions_checked_when_explicitly_requested(self):
        from PIL import Image
        from furniture_bench.data import digest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic.png';e=Image.Exif();e[274]=6;Image.new('RGB',(100,80),'red').save(p,exif=e)
            r=record();r['photo'].update(image_sha256=digest(p),width=80,height=100)
            result=validate(r,image_path=p);self.assertTrue(result['image_content_verified'])
            r['photo']['image_sha256']='e'*64
            with self.assertRaises(ValueError):validate(r,image_path=p)

    def test_UI_has_no_prechecked_grants_or_external_scripts(self):
        root=Path(__file__).parent;html=(root/'intake.html').read_text();js=(root/'intake.js').read_text()
        class Checks(HTMLParser):
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=='input' and a.get('type')=='checkbox':self.assert_not_checked('checked' in a)
                if tag=='script':self.assert_local(a.get('src'))
        p=Checks();p.assert_not_checked=lambda v:self.assertFalse(v);p.assert_local=lambda v:self.assertEqual(v,'/intake.js');p.feed(html)
        self.assertNotIn('fetch(',js);self.assertNotIn('innerHTML',js)
        self.assertIn('decision:\'proposal\'',js);self.assertIn('annotations_active:false',js)
        self.assertIn('imageEpoch',js);self.assertIn(".checked=false",js)

    def test_static_server_denies_uploads_private_paths_and_rebinding(self):
        with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for method,path,status in [('GET','/',200),('GET','/intake.js',200),('GET','/../.git/config',404),('POST','/',405)]:
                    c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request(method,path);r=c.getresponse();r.read()
                    self.assertEqual(r.status,status)
                    if status==200:self.assertIn("connect-src 'none'",r.getheader('Content-Security-Policy'))
                    c.close()
                c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET','/',headers={'Host':'attacker.invalid'})
                r=c.getresponse();r.read();self.assertEqual(r.status,403);c.close()
            finally:server.shutdown();thread.join(3)


if __name__=='__main__':unittest.main()
