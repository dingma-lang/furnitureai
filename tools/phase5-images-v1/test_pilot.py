import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import network as n
import licensing as lic

class Response:
    def __init__(self,body,status=200,headers=None):self.body=io.BytesIO(body);self.status=status;self.headers=headers or {'Content-Length':str(len(body))}
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self,size):return self.body.read(size)

class Tests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'new'
    def test_completed_acquisition_cannot_be_reopened(self):
        self.root.mkdir();(self.root/'completion.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'immutable'):n.Budget(self.root,resume=True)

    def test_matching_individual_rights_only(self):
        row=dict(source_identity='flickr-photo:123',author='Author',author_profile_url='https://www.flickr.com/people/a/')
        obj={'@type':'ImageObject','acquireLicensePage':'https://www.flickr.com/photos/a/123','license':'https://creativecommons.org/licenses/by/2.0/','author':{'name':'Author','url':'https://www.flickr.com/photos/a'}}
        html=lambda x:'<script type="application/ld+json">'+json.dumps({'@graph':[x]})+'</script>'
        self.assertEqual(lic.individual_rights(html(obj),row)['status'],'verified_individual_current_source')
        for field,value in [('license','https://creativecommons.org/licenses/by-nc/2.0/'),('acquireLicensePage','https://www.flickr.com/photos/a/456')]:
            bad=dict(obj,**{field:value})
            with self.assertRaises(ValueError):lic.individual_rights(html(bad),row)
        with self.assertRaises(ValueError):lic.individual_rights('Dataset CC BY 2.0',row)
    def test_response_budget_and_cached_integrity(self):
        b=n.Budget(self.root)
        try:
            with patch.object(b.opener,'open',return_value=Response(b'body')) as op:
                rec=b.fetch('https://example.test','licensing/one',20)
                self.assertEqual(b.state['actual_body_bytes'],4);self.assertEqual(b.state['charged_bytes'],4)
                b.fetch('https://example.test','licensing/one',20);self.assertEqual(op.call_count,1)
                (self.root/'licensing/one').write_bytes(b'altered')
                with self.assertRaises(ValueError):b.fetch('https://example.test','licensing/one',20)
        finally:b.close()
    def test_gzip_lowercase_header_and_decode_limit(self):
        b=n.Budget(self.root)
        try:
            data=gzip.compress(b'licensed HTML')
            with patch.object(b.opener,'open',return_value=Response(data,headers={'Content-Length':str(len(data)),'content-encoding':'gzip'})):
                rec=b.fetch('https://example.test','page',100)
                self.assertEqual(b.decoded(rec),'licensed HTML')
                with self.assertRaises(ValueError):b.decoded(rec,max_decoded=3)
        finally:b.close()
    def test_approved_cap_stops_before_request(self):
        b=n.Budget(self.root)
        try:
            with patch.object(b.opener,'open',side_effect=AssertionError('must not request')):
                with self.assertRaises(RuntimeError):b.fetch('https://example.test','page',n.CAP+1)
        finally:b.close()
    def test_image_two_attempt_limit_and_header_binding(self):
        b=n.Budget(self.root)
        try:
            for i in range(2):
                with patch.object(b.opener,'open',return_value=Response(b'bad',headers={'Content-Length':'3','ETag':'changed'})):
                    with self.assertRaises(ValueError):b.fetch('https://example.test/image',f'photo-{i}',3,kind='image',expected=3,etag='fixed')
            with self.assertRaisesRegex(RuntimeError,'two-attempt'):b.fetch('https://example.test/image','photo-3',3,kind='image',expected=3,etag='fixed')
            self.assertEqual(b.state['charged_bytes'],6) # uncertain failures remain conservatively charged
        finally:b.close()
    def test_frozen_manifest_pin_and_membership(self):
        m=n.load_manifest();self.assertEqual(len(m['images']),150)
        self.assertEqual({p:sum(r['proposed_split']==p for r in m['images']) for p in ('train','development','frozen_evaluation')},dict(train=60,development=30,frozen_evaluation=60))

class ScreeningTests(unittest.TestCase):
    def test_signatures_and_cross_partition_duplicates(self):
        from PIL import Image
        import screen
        im=Image.new('RGB',(64,64),'gray');sig=screen.signatures(im)
        a=dict(image_id='a',proposed_split='train',sha256='same',**sig)
        b=dict(a,image_id='b',proposed_split='frozen_evaluation')
        pairs=screen.pair_diagnostics([a,b]);self.assertEqual(len(pairs),1)
        self.assertTrue(pairs[0]['exact_file']);self.assertTrue(pairs[0]['exact_pixels']);self.assertTrue(pairs[0]['cross_partition'])
    def test_jpeg_integrity_rejects_non_image(self):
        import acquire_photos
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'bad.jpg';bad.write_bytes(b'not an image')
            with self.assertRaises(Exception):acquire_photos.verify_jpeg(bad)
            good=Path(tmp)/'good.jpg';Image.new('RGB',(64,48),'green').save(good)
            self.assertEqual(acquire_photos.verify_jpeg(good)['width'],64)
    def test_actual_licensing_and_body_accounting(self):
        state=json.loads((n.DEST/'network-ledger.json').read_text());rights=json.loads((n.DEST/'licensing-audit.json').read_text())
        self.assertEqual(rights['verified'],118);self.assertEqual(rights['quarantined'],32)
        self.assertEqual(state['actual_body_bytes'],55307887);self.assertEqual(state['charged_bytes'],55307887)
        self.assertEqual(sum(r['actual_body_bytes'] for r in state['requests']),state['actual_body_bytes'])
        self.assertLess(state['charged_bytes'],n.CAP)
        self.assertTrue(all(v==1 for v in state['image_attempts'].values()))
        for rec in state['requests']:
            self.assertTrue(rec['complete']);self.assertEqual(n.sha(n.DEST/rec['file']),rec['sha256'])
    def test_no_unlicensed_photo_download_and_fixed_membership(self):
        rights={r['image_id']:r for r in json.loads((n.DEST/'licensing-audit.json').read_text())['records']}
        records=json.loads((n.DEST/'acquisition-summary.json').read_text())['records']
        approved={r['image_id']:r for r in n.load_manifest()['images']}
        self.assertEqual(len(records),118)
        for row in records:
            self.assertEqual(rights[row['image_id']]['status'],'verified_individual_current_source')
            self.assertEqual(row['proposed_split'],approved[row['image_id']]['proposed_split'])
            self.assertLessEqual(row['bytes'],n.PHOTO_MAX);self.assertEqual(n.sha(n.DEST/row['file']),row['sha256'])
        self.assertEqual(len(list((n.DEST/'photos').glob('*.jpg'))),118)
    def test_final_annotation_immutability_and_unverified_flags(self):
        result=json.loads((n.DEST/'screening-final-v1/screening.json').read_text());approved={r['image_id']:r for r in n.load_manifest()['images']}
        self.assertEqual(len(result['images']),150);self.assertFalse(result['partition_assignments_changed'])
        self.assertEqual(sum(r['eligibility_status']=='provisional_candidate' for r in result['images']),63)
        for row in result['images']:
            self.assertEqual(row['original_annotations'],approved[row['image_id']]['annotations'])
            self.assertEqual(row['proposed_split'],approved[row['image_id']]['proposed_split'])
            self.assertFalse(row['human_reviewed']);self.assertFalse(row['scene_verified'])
        self.assertFalse(result['duplicate_pairs'])
    def test_missing_labels_not_negative_and_hierarchy_flag(self):
        import screen
        row=dict(annotations=[dict(label='chair',source_mid='/m/01mzpv',box=[0,0,.2,.2],ignore=False,flags=dict(IsOccluded=0,IsTruncated=0))])
        raw=[dict(LabelName='/m/078n6m',XMin='.3',YMin='.3',XMax='.8',YMax='.8')]
        result=screen.annotation_diagnostics(row,raw)
        self.assertTrue(result['hierarchy_coverage_unresolved']);self.assertTrue(result['unannotated_classes_are_not_negatives'])

if __name__=='__main__':unittest.main()
