import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import detection_prep as d


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def fixture(self, name, text):
        p=self.root/name;p.write_text(text);return p

    def manifest_row(self, ident='im1', part='train'):
        return dict(image_id=ident,sha256=d.hashlib.sha256(ident.encode()).hexdigest(),source_identity='source-'+ident,identity_group='family-'+ident,split=part,identity_uncertain=False)

    def test_iou(self):
        self.assertEqual(d.iou([0,0,1,1],[0,0,1,1]),1)
        self.assertEqual(d.iou([0,0,1,1],[1,1,2,2]),0)
        self.assertAlmostEqual(d.iou([0,0,2,2],[1,1,3,3]),1/7)
        for b in ([0,0,0,1],[0,0,float('nan'),1]):
            with self.assertRaises(ValueError):d.box(b)

    def test_dictionary_exact_only(self):
        p=self.fixture('dict.csv','/m/test,Couch\n/m/notfurniture,Drum\n/m/chair,Chair\n')
        m=d.class_mapping(p)
        self.assertEqual(m['mapping'],{'/m/test':'sofa','/m/chair':'chair'})
        self.assertIn('Desk',m['missing_exact_names'])

    def test_annotation_schema_flags_and_geometry(self):
        header='ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'
        row='im1,x,/m/test,1,0.1,0.9,0.2,0.8,0,1,0,0,0\n'
        p=self.fixture('box.csv',header+row)
        a=d.parse_annotations(p,{'/m/test':'sofa'})
        self.assertEqual(a[0]['box'],[.1,.2,.9,.8]);self.assertFalse(a[0]['ignore'])
        self.fixture('box.csv',header+row.replace(',0,0,0\n',',1,0,0\n'))
        self.assertTrue(d.parse_annotations(p,{'/m/test':'sofa'})[0]['ignore'])
        for text in ('ImageID\nim1\n',header+row+row,header+row.replace('0.9','1.1'),header+row.replace(',0,1,0,0,0',',9,1,0,0,0')):
            self.fixture('box.csv',text)
            with self.assertRaises(ValueError):d.parse_annotations(p,{'/m/test':'sofa'})

    def test_partial_image_labels(self):
        p=self.fixture('labels.csv','ImageID,LabelName,Confidence\nim1,/m/c,1\nim1,/m/t,0\n')
        self.assertEqual(d.parse_image_labels(p,{'/m/c':'chair','/m/t':'table'}),{('im1','chair'):1,('im1','table'):0})
        p.write_text(p.read_text()+'im1,/m/c,0\n')
        with self.assertRaises(ValueError):d.parse_image_labels(p,{'/m/c':'chair'})

    def test_duplicate_matching(self):
        gt=[dict(label='chair',box=[.1,.1,.8,.8])]
        preds=[dict(label='chair',box=[.1,.1,.8,.8],score=s) for s in (.9,.8)]
        r=d.evaluate_image(preds,gt,['chair'])
        self.assertEqual([r['categories']['chair'][k] for k in ('tp','fp','fn','duplicate_fp')],[1,1,0,1])
        result=d.summarize([r]);self.assertEqual(result['categories']['chair']['precision'],.5)
        self.assertEqual(result['categories']['chair']['recall'],1)

    def test_unknown_negative_and_group_ignore(self):
        preds=[dict(label='chair',box=[0,0,.5,.5],score=.9)]
        self.assertEqual(d.evaluate_image(preds,[],[])['unknown_class_predictions'],1)
        self.assertEqual(d.evaluate_image(preds,[],['chair'])['categories']['chair']['fp'],1)
        ignored=[dict(label='chair',box=[0,0,1,1],ignore=True)]
        self.assertEqual(d.evaluate_image(preds,ignored,['chair'])['ignored_predictions'],1)
        # Ordinary GT gets a chance before group ignore.
        gt=ignored+[dict(label='chair',box=[0,0,.5,.5])]
        self.assertEqual(d.evaluate_image(preds,gt,['chair'])['categories']['chair']['tp'],1)

    def test_fragments_are_diagnostic_false_positives(self):
        gt=[dict(label='table',box=[0,0,1,1])]
        p=[dict(label='table',box=[0,0,.4,.4],score=.8)]
        r=d.evaluate_image(p,gt,['table'])['categories']['table']
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['fragment_candidates']),(0,1,1,1))

    def test_isolation_all_keys_and_reserved(self):
        a=self.manifest_row();b=self.manifest_row('im2','frozen_evaluation')
        d.assert_isolation([a,b])
        for key in ('sha256','source_identity','identity_group'):
            c=copy.deepcopy(b);c[key]=a[key]
            with self.assertRaises(ValueError):d.assert_isolation([a,c])
        for flag in ('official_phase2b','abo_frozen_panel'):
            c=dict(b,**{flag:True})
            with self.assertRaises(ValueError):d.assert_isolation([a,c])
        with self.assertRaises(ValueError):d.assert_isolation([a,a])
        with self.assertRaises(ValueError):d.assert_isolation([dict(a,image_id='../escape')])

    def test_deterministic_selection_and_uncertainty(self):
        rows=[]
        for i in range(200):
            r=self.manifest_row('im'+str(i));r.update(source_partition='validation' if i<120 else 'test',license_verified=True,indoor_verified=True,annotations_verified=True,box_counts={d.CLASSES[i%7]:1})
            rows.append(r)
        a=d.select_pilot(rows);b=d.select_pilot(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(a['actual'],dict(train=60,development=30,frozen_evaluation=60))
        rows[0]['identity_uncertain']=True
        self.assertNotIn('im0',{r['image_id'] for r in d.select_pilot(rows)['rows']})

    def test_gate_preserves_and_abstains(self):
        row=dict(detection_id='d0',original_box=[0,10,60,100],detector_label='chair',detector_score=.5,baseline=dict(category='chair'),expanded=dict(category='table',softmax7_margin=.7))
        before=copy.deepcopy(row);q=d.quality(row,[row],(200,200))
        self.assertEqual(row,before);self.assertEqual(q['status'],'unresolved_risk');self.assertIsNone(q['advisory_category'])
        self.assertTrue(q['measurements']['classifiers_disagree'])
        row['original_box']=[20,20,100,100]
        q=d.quality(row,[row],(200,200))
        self.assertEqual(q['status'],'candidate_unverified') # disagreement alone is not an exclusion
        self.assertFalse(q['human_verified'])

    def test_gate_overlaps_and_content(self):
        from PIL import Image
        a=dict(detection_id='a',original_box=[20,20,120,120],detector_label='table',detector_score=.5,baseline=dict(category='table'),expanded=dict(category='table',softmax7_margin=.7))
        b=dict(a,detection_id='b',original_box=[30,30,80,80],detector_label='chair')
        q=d.quality(a,[a,b],(200,200),Image.new('RGB',(100,100),'white'))
        self.assertIn('mixed_object_overlap_risk',q['heuristic_flags']);self.assertIn('blank_low_contrast_risk',q['heuristic_flags'])
        tiny=d.quality(a,[a],(200,200),Image.new('RGB',(1,1),'white'))
        self.assertEqual(tiny['measurements']['mean_adjacent_gray_difference'],0.0)

    def test_checkpoint_hash_and_readonly(self):
        model=self.root/'model';model.mkdir();p=model/'model.safetensors';p.write_bytes(b'protected')
        before=p.read_bytes()
        with self.assertRaises(ValueError):d.detector_loader(model,'0'*64)
        self.assertEqual(p.read_bytes(),before)
        out=self.root/'existing.json';out.write_text('historical')
        with self.assertRaises(FileExistsError):d.write_new(out,{})
        self.assertEqual(out.read_text(),'historical')

    def test_refuse_provisional_or_phase2b_evaluation(self):
        p=self.fixture('provisional.json',json.dumps(dict(schema='phase5-public-boxes-v1',source='assistant_provisional')))
        with self.assertRaises(ValueError):d.evaluate_bundle(p,self.root/'missing',self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_no_model_inference_on_missing_bundle(self):
        import run_baseline
        with patch.object(d,'detector_loader',side_effect=AssertionError('must not load')):
            with self.assertRaises(FileNotFoundError):run_baseline.infer(self.root/'missing.json',self.root/'out',self.root/'model','0'*64)

    def test_public_gt_must_match_hashed_sources(self):
        dictionary=self.fixture('dictionary.csv','/m/test,Chair\n')
        boxes=self.fixture('boxes.csv','ImageID,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\nim1,/m/test,1,0,1,0,1,0,0,0,0,0\n')
        labels=self.fixture('labels.csv','ImageID,LabelName,Confidence\nim1,/m/test,1\n')
        record=lambda p,kind: dict(file=p.name,sha256=d.sha(p),url='https://storage.googleapis.com/openimages/'+p.name,partition='test',kind=kind)
        row=self.manifest_row('im1','frozen_evaluation');row.update(source_partition='test',verified_classes=['chair'],annotations=d.parse_annotations(boxes,{'/m/test':'chair'}))
        bundle=dict(schema='phase5-public-boxes-v1',source='Open Images',class_dictionary=record(dictionary,'dictionary'),annotation_files=[record(boxes,'boxes'),record(labels,'human_labels')],images=[row])
        p=self.fixture('bundle.json',json.dumps(bundle));d.validate_public_bundle(p)
        bundle['images'][0]['annotations'][0]['box']=[0,0,.5,.5];p.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError,'differs from published'):d.validate_public_bundle(p)
        p.write_text(json.dumps(dict(bundle,images=[])));boxes.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):d.validate_public_bundle(p)

    def test_offline_metadata_audit_counts_and_no_overwrite(self):
        import audit_openimages
        import contextlib,io
        dictionary=self.fixture('dictionary.csv','/m/test,Chair\n')
        hierarchy=self.fixture('hierarchy.json',json.dumps(dict(LabelName='/m/entity',Subcategory=[dict(LabelName='/m/test')])))
        boxes=[];labels=[];images=[]
        for part,ident in [('validation','im1'),('test','im2')]:
            boxes.append(self.fixture(part+'boxes.csv','ImageID,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'+ident+',/m/test,1,0,1,0,1,0,0,0,0,0\n'))
            labels.append(self.fixture(part+'labels.csv','ImageID,LabelName,Confidence\n'+ident+',/m/test,1\n'))
            images.append(self.fixture(part+'images.csv','ImageID,OriginalURL,OriginalLandingURL,License,Author,AuthorProfileURL,Rotation\n'+ident+',https://example/image,https://example/landing,https://creativecommons.org/licenses/by/2.0/,author,https://example/author,0\n'))
        out=self.root/'audit'
        with contextlib.redirect_stdout(io.StringIO()):audit_openimages.audit(dictionary,hierarchy,boxes,labels,images,out)
        actual=d.read(out/'metadata-audit.json')
        self.assertEqual(actual['partitions']['test']['category_counts']['chair']['boxes'],1)
        self.assertEqual(actual['partitions']['validation']['total_furniture_images'],1)
        self.assertIsNone(actual['actual_selected_boxes'])
        with self.assertRaises(FileExistsError):audit_openimages.audit(dictionary,hierarchy,boxes,labels,images,out)

    def test_actual_reserved_hash_is_blocked_without_flags(self):
        protected=d.reserved_identities()
        reserved_sha=next(v for v in protected if len(v)==64)
        r=self.manifest_row();r['sha256']=reserved_sha
        with self.assertRaisesRegex(ValueError,'Reserved historical'):d.assert_isolation([r])

    def test_inference_failure_resume_and_protection(self):
        import run_baseline
        from PIL import Image
        model=self.root/'model';model.mkdir();(model/'model.safetensors').write_bytes(b'pinned')
        rows=[]
        for ident in ('im1','im2'):
            photo=self.root/(ident+'.jpg');Image.new('RGB',(100,100),'gray').save(photo)
            r=self.manifest_row(ident,'frozen_evaluation');r.update(file=photo.name,sha256=d.sha(photo),license_verified=True,annotations_verified=True,orientation_verified=True,annotation_provenance='published_open_images')
            rows.append(r)
        bundle=self.fixture('bundle.json',json.dumps(dict(schema='phase5-public-boxes-v1',source='Open Images',images=rows)))
        class Detector:
            def __init__(self):self.calls=0
            def detect(self,im):
                self.calls+=1
                if self.calls==2:raise RuntimeError('simulated CUDA failure')
                return [dict(label='chair',score=.8,box=[10,10,80,80])],.01
        detector=Detector();out=self.root/'run'
        with patch.object(d,'detector_loader',return_value=detector), patch.object(d,'validate_public_bundle',return_value=d.read(bundle)):
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):run_baseline.infer(bundle,out,model,d.sha(model/'model.safetensors'))
            saved=(out/'im1.json').read_bytes()
            self.assertFalse((out/'predictions.json').exists())
            run_baseline.infer(bundle,out,model,d.sha(model/'model.safetensors'),resume=True)
            self.assertEqual(detector.calls,3) # only unfinished image recomputed
            self.assertEqual((out/'im1.json').read_bytes(),saved)
            run_baseline.infer(bundle,out,model,d.sha(model/'model.safetensors'),resume=True)
            self.assertEqual(detector.calls,3)
            with self.assertRaises(FileExistsError):run_baseline.infer(bundle,out,model,d.sha(model/'model.safetensors'))
            (out/'im1.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'corrupted stage'):run_baseline.infer(bundle,out,model,d.sha(model/'model.safetensors'),resume=True)
        self.assertEqual((model/'model.safetensors').read_bytes(),b'pinned')

if __name__=='__main__': unittest.main()
