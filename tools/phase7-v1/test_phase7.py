import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from contracts import CATEGORIES, LVIS, OI, assert_isolated, box, mask_valid, lvis_scope, validate, write_new
from audit import category_summary, literal, oi_counts, lvis_local, safe_lvis_member, stream
from local_dry_run import oi_source_index
from synthetic import asset_plan, intersects, scenes, visible_annotations
from render_worker import approved_inputs

def record():
    rights=dict(status='verified',license_url='https://creativecommons.org/licenses/by/2.0/',creator='fixture',
                source_url='https://example.org/fixture',evidence_sha256='a'*64,verified_utc='2026-11-06T00:00:00Z',attribution='fixture')
    return dict(schema='furniture-detection-v1',source='fixture',source_image_id='1',domain='real',partition='train',
        width=10,height=10,image_sha256='b'*64,source_groups=['fixture-author:a'],photo_rights=rights,
        annotation_rights=copy.deepcopy(rights),annotations=[dict(source_annotation_id='a',category='chair',original={'category_id':232},
        mapping_rule='lvis-core-v1:232',bbox_xywh=[1,1,3,3],segmentation=None)],
        coverage={k:('positive_nonexhaustive' if k=='chair' else 'unknown') for k in CATEGORIES},
        native_coverage={'positive_native_category_ids':[232],'not_exhaustive_category_ids':[232],'neg_category_ids':[]},provenance={'fixture':True})

class Contracts(unittest.TestCase):
    def test_valid_contract(self):
        r=record();self.assertIs(validate(r),r);self.assertNotIn('human_verified',r)
    def test_unverified_rights_excluded(self):
        r=record();r['photo_rights']['status']='potentially_compatible'
        with self.assertRaises(ValueError):validate(r)
    def test_restricted_rights_excluded(self):
        r=record();r['photo_rights']['license_url']='https://creativecommons.org/licenses/by-nc/2.0/'
        with self.assertRaises(ValueError):validate(r)
    def test_missing_evidence_excluded(self):
        r=record();r['photo_rights']['evidence_sha256']=''
        with self.assertRaises(ValueError):validate(r)
    def test_unknown_scope_cannot_be_background(self):
        self.assertEqual(lvis_scope({},232,False),'unknown')
        self.assertEqual(lvis_scope({'not_exhaustive_category_ids':[232]},232,True),'positive_nonexhaustive')
        self.assertEqual(lvis_scope({},232,True),'positive_exhaustive')
        self.assertEqual(lvis_scope({'neg_category_ids':[232]},232,False),'explicit_negative')
    def test_conflicting_scope_rejected(self):
        with self.assertRaises(ValueError):lvis_scope({'neg_category_ids':[232]},232,True)
        r=record();r['coverage']['chair']='explicit_negative'
        with self.assertRaises(ValueError):validate(r)
    def test_broad_coverage_cannot_replace_original_native_scope(self):
        r=record();r.pop('native_coverage')
        with self.assertRaises(ValueError):validate(r)
    def test_box_nan_outside_degenerate(self):
        for b in [[float('nan'),0,1,1],[-1,0,2,2],[0,0,0,2],[9,0,2,1]]:
            with self.assertRaises(ValueError):box(b,10,10)
    def test_native_mask_formats_and_invalid_runs(self):
        mask_valid([[0,0,2,0,2,2]],10,10,'coco_polygons')
        mask_valid({'size':[10,10],'counts':[10,5,85]},10,10,'coco_uncompressed_rle')
        for mask,kind in [([[0,0,2]],'coco_polygons'),({'size':[10,10],'counts':[4,5]},'coco_uncompressed_rle'),({'counts':'opaque'},'coco_compressed_rle')]:
            with self.assertRaises(ValueError):mask_valid(mask,10,10,kind)
    def test_transitive_identity_bridge_rejected(self):
        a=record();b=copy.deepcopy(a);c=copy.deepcopy(a)
        b.update(source_image_id='2',image_sha256='c'*64,source_groups=['fixture-author:a','fixture-author:b'])
        c.update(source_image_id='3',image_sha256='d'*64,source_groups=['fixture-author:b'],partition='development')
        with self.assertRaises(ValueError):assert_isolated([a,b,c])
    def test_duplicate_hash_leakage_rejected(self):
        a=record();b=copy.deepcopy(a);b.update(source_image_id='2',partition='development',source_groups=['other'])
        with self.assertRaises(ValueError):assert_isolated([a,b])
    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):assert_isolated([record(),record()])
    def test_independent_groups_pass(self):
        a=record();b=copy.deepcopy(a);b.update(source_image_id='2',partition='development',source_groups=['other'],image_sha256='d'*64)
        self.assertEqual(assert_isolated([a,b]),2)
    def test_synthetic_real_evaluation_separation(self):
        r=record();r.update(domain='synthetic',partition='development')
        with self.assertRaises(ValueError):validate(r)
    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'result.json';write_new(p,{'old':1})
            with self.assertRaises(FileExistsError):write_new(p,{'new':2})
            self.assertEqual(json.loads(p.read_text()),{'old':1})

class Audit(unittest.TestCase):
    def test_cross_source_join_does_not_count_coco_self_matches(self):
        rows=[dict(source='coco2017',source_aliases=['flickr-photo:1']),dict(source='openimages',source_aliases=['flickr-photo:2'])]
        self.assertEqual(set(oi_source_index(rows)),{'flickr-photo:2'})
    def test_zip_stream_without_extraction(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'lvis.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('lvis_v1_train.json','{"images":[],"categories":[{"id":1}],"annotations":[]}')
            self.assertEqual(list(stream(p)),[('categories',{'id':1})]);self.assertEqual(len(list(Path(t).iterdir())),1)
    def test_zip_traversal_and_extra_members_rejected(self):
        for names in [['../lvis_v1_train.json'],['lvis_v1_train.json','extra.txt']]:
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'x.zip'
                with zipfile.ZipFile(p,'w') as z:
                    for name in names:z.writestr(name,'{}')
                with zipfile.ZipFile(p) as z:
                    with self.assertRaises(ValueError):safe_lvis_member(z)
    def test_zip_expansion_bomb_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.zip'
            with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('lvis_v1_train.json',' '*1000000)
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):safe_lvis_member(z)
    def test_published_native_taxonomy_and_counts(self):
        d=category_summary();core={r['id']:r for r in d['categories'] if r['mapping']=='core_native'}
        self.assertEqual(set(core),set(LVIS));self.assertEqual(core[361]['published_train_images'],1100)
        self.assertIsNone(d['additional_instances_for_local_photos']);self.assertNotIn(628,LVIS)
    def test_literal_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'table.py';p.write_text("raise RuntimeError('never execute')\nDATA=[{'id':1}]\n")
            self.assertEqual(literal(p,'DATA'),[{'id':1}])
    def test_oi_native_count_and_flags(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'train.csv';fields=['ImageID','LabelName','XMin','XMax','YMin','YMax','Confidence','IsGroupOf','IsDepiction','IsInside','IsOccluded','IsTruncated']
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for ident in ['a'*16,'a'*16,'b'*16]:
                    w.writerow(dict.fromkeys(fields,'0')|dict(ImageID=ident,LabelName='/m/01y9k5',XMin='.1',XMax='.9',YMin='.2',YMax='.8',Confidence='1',IsOccluded='1'))
            d=oi_counts(p,'train');desk=next(x for x in d['classes'] if x['category']=='desk')
            self.assertEqual((desk['images'],desk['boxes'],desk['flags']['IsOccluded']),(2,3,3))
    def test_oi_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.csv';p.write_text('ImageID,bad\na,b\n')
            with self.assertRaises(ValueError):oi_counts(p,'train')
    def test_lvis_existing_join_preserves_partition_and_unknowns(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'lvis.json';d=dict(categories=[{'id':k,'name':v} for k,v in LVIS.items()],
                images=[dict(id=123,width=10,height=10,coco_url='http://images.cocodataset.org/train2017/000000000123.jpg',neg_category_ids=[],not_exhaustive_category_ids=[361])],
                annotations=[dict(id=1,image_id=123,category_id=361,bbox=[1,2,3,4],segmentation=[[1,2,4,2,4,6]])])
            p.write_text(json.dumps(d));r=lvis_local(p,[dict(coco_image_id=123,proposed_split='development',status='quarantined')])
            m=r['matching_existing_coco_images'][0];self.assertEqual(m['partition'],'development');self.assertFalse(m['reuse_authorized']);self.assertEqual(m['eligibility'],'quarantined');self.assertEqual(r['new_photograph_count'],0)

class Synthetic(unittest.TestCase):
    def test_exact_asset_budget_and_no_download(self):
        p=asset_plan();self.assertEqual(p['expected_asset_body_bytes'],11026401);self.assertEqual(len(p['unique_objects']),33);self.assertFalse(p['approved'])
    def test_reproducible_seed_and_bounded_count(self):
        self.assertEqual(scenes(),scenes());self.assertNotEqual(scenes(1),scenes(2));self.assertEqual(len(scenes()),100)
        with self.assertRaises(ValueError):scenes(count=101)
    def test_floorplan_no_interpenetration(self):
        for p in scenes()[::4]:
            w,l,_=p['room_metres']
            for i,a in enumerate(p['objects']):
                self.assertGreater(a['x']-a['sx']/2,0);self.assertLess(a['x']+a['sx']/2,w)
                self.assertGreater(a['y']-a['sy']/2,0);self.assertLess(a['y']+a['sy']/2,l)
                for b in p['objects'][i+1:]:self.assertFalse(intersects(a,b))
    def test_all_categories_and_known_instance_ids(self):
        union=set()
        for p in scenes():
            union.update(r['category'] for r in p['objects'])
            self.assertEqual(len({r['instance_id'] for r in p['objects']}),len(p['objects']))
        self.assertEqual(union,set(CATEGORIES))
    def test_visible_bbox_and_rle_roundtrip(self):
        m=np.array([[0,1,1,0],[0,1,0,2],[0,0,0,2]],dtype=np.uint16)
        meta={1:dict(category='chair',asset_id='a'),2:dict(category='table',asset_id='b'),3:dict(category='bed',asset_id='c')}
        rows=visible_annotations(m,meta);self.assertEqual(len(rows),2);self.assertEqual(rows[0]['bbox_xywh'],[1,0,2,2]);self.assertEqual(rows[0]['area'],3)
        for r in rows:
            bits=[];bit=0
            for n in r['segmentation']['counts']:bits.extend([bit]*n);bit=1-bit
            decoded=np.array(bits).reshape(m.shape,order='F');np.testing.assert_array_equal(decoded,m==r['instance_id'])
    def test_unknown_instance_not_assigned_category(self):
        with self.assertRaises(ValueError):visible_annotations(np.array([[4]],dtype=np.int32),{})
    def test_unapproved_renderer_never_imports_blender(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'bundle.json').write_text('{}');(p/'approval.json').write_text('{}')
            with self.assertRaises(ValueError):approved_inputs(p/'bundle.json',p/'approval.json',p)
    def test_offline_no_socket_use(self):
        with patch('socket.socket',side_effect=AssertionError('Network forbidden in dry run')):
            asset_plan();scenes(count=8);category_summary()

if __name__=='__main__':unittest.main()
