"""CPU-only source geometry,identity,automatic evidence and dashboard regressions."""
import ast,copy,csv,http.client,http.server,threading,unittest,urllib.parse
import numpy as np
from common8q import *
from geometry8 import coverage,admission,polygon_area
from qualify8 import decide
from serve8 import load_dashboard,handler_for

class GeometryTests(unittest.TestCase):
    def test_exact_interior_requires_boundary_clearance(self):
        poly=[(0,0),(100,0),(100,100),(0,100)]
        self.assertTrue(admission([10,10,60,60],100,100,[poly],[])['geometric_eligible'])
        self.assertFalse(admission([0,0,60,60],100,100,[poly],[])['geometric_eligible'])

    def test_partial_area_cannot_certify_absence(self):
        poly=[(0,0),(60,0),(60,100),(0,100)]
        self.assertAlmostEqual(coverage(poly,[0,0,100,100]),.6)
        self.assertFalse(admission([10,10,80,80],100,100,[poly],[])['geometric_eligible'])

    def test_concave_mask_notch_is_not_filled(self):
        poly=[(0,0),(100,0),(100,40),(40,40),(40,100),(0,100)]
        self.assertAlmostEqual(polygon_area(poly),6400.)
        self.assertAlmostEqual(coverage(poly,[0,0,100,100]),.64)
        self.assertFalse(admission([10,10,90,90],100,100,[poly],[])['geometric_eligible'])

    def test_no_sum_across_overlapping_masks(self):
        poly=[(0,0),(50,0),(50,100),(0,100)]
        fractions=[coverage(poly,[0,0,100,100]),coverage(poly,[0,0,100,100])]
        self.assertEqual(max(fractions),.5)

    def test_positive_or_ignored_overlap_blocks_admission(self):
        p=[(0,0),(100,0),(100,100),(0,100)]
        r=admission([10,10,60,60],100,100,[p],[dict(key='Chair_or_stool',box=[.5,.5,.9,.9])])
        self.assertFalse(r['geometric_eligible']);self.assertEqual(r['protected_overlaps'],['Chair_or_stool'])

    def test_tiny_and_out_of_frame_rejected(self):
        p=[(0,0),(100,0),(100,100),(0,100)]
        self.assertFalse(admission([10,10,25,30],100,100,[p],[])['minimum_side_ok'])
        self.assertFalse(admission([-2,10,60,60],100,100,[p],[])['bounds_ok'])

class DecisionTests(unittest.TestCase):
    def fixture(self):
        gate=dict(bounds_ok=True,minimum_side_ok=True,geometric_eligible=False)
        return dict(student_Chair_score=.8,native_mask_hits=[dict(geometric_gate=gate,native_ignored=False,protected_furniture=False)])

    def test_clear_appearance_alone_does_not_invent_negative(self):
        r=self.fixture();status,_=decide(r,('other_object_handbag','Synthetic fixture only',False,False))
        self.assertEqual(status,'unverified_candidate')

    def test_real_seating_or_furniture_fragment_always_protected(self):
        r=self.fixture();r['native_mask_hits'][0]['geometric_gate']['geometric_eligible']=True
        self.assertEqual(decide(r,('possible_furniture_fragment','Fixture',True,False))[0],'protected_unsuitable')

    def test_ignored_native_annotation_cannot_authorize_negative(self):
        r=self.fixture();r['native_mask_hits'][0]['geometric_gate']['geometric_eligible']=True;r['native_mask_hits'][0]['native_ignored']=True
        self.assertEqual(decide(r,('other_object_appliance','Fixture',False,False))[0],'unverified_candidate')

    def test_clear_source_supported_and_high_confidence_needed(self):
        r=self.fixture();r['native_mask_hits'][0]['geometric_gate']['geometric_eligible']=True
        self.assertEqual(decide(r,('other_object_handbag','Synthetic source fixture only',False,False))[0],'trusted_negative')
        r['student_Chair_score']=.1
        self.assertEqual(decide(r,('other_object_handbag','Fixture',False,False))[0],'unverified_candidate')

    def test_ambiguous_and_overexposed_remain_inactive(self):
        r=self.fixture()
        self.assertEqual(decide(r,('ambiguous_background','Fixture',False,False))[0],'unverified_candidate')
        self.assertEqual(decide(r,('ambiguous_glare','Fixture',False,True))[0],'protected_unsuitable')

class ActualDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=read(DATA/'automatic-decisions-v1.json')['records'];cls.manifest=read(DATA/'hard-negative-dataset-v1.json')

    def test_exact21_and_source_training_isolation(self):
        _,train,dev=context()
        self.assertEqual({r['candidate_id'] for r in self.rows},{r['candidate_id'] for r in exact_candidates()})
        self.assertEqual(len(self.rows),21)
        for r in self.rows:
            require_train(r['image_id'],train)
            self.assertEqual(r['source_sha256'],train[r['image_id']]['sha256'])
        with self.assertRaises(PermissionError):require_train(dev[0]['image_id'],train)

    def test_original_decisions_preserved_in_separate_phase(self):
        old=read(E7/'analysis/candidate-manifest.json');self.assertEqual(len(old['B_automatic_unverified']),21)
        self.assertEqual(old['training_eligible_new'],[])
        self.assertFalse(self.manifest['source_annotations_or_partitions_modified'])

    def test_all_new_manifest_tiers_disjoint_and_complete(self):
        groups=[self.manifest[k] for k in ('trusted_negative_ids','unverified_candidate_ids','protected_unsuitable_ids')]
        ids=[v for g in groups for v in g]
        self.assertEqual([len(g) for g in groups],[0,5,16]);self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{r['candidate_id'] for r in self.rows})

    def test_actual_inspection_not_human_or_training_activation(self):
        for r in self.rows:
            self.assertTrue(r['assistant_actual_visual_inspection'])
            self.assertFalse(r['human_reviewed']);self.assertFalse(r['human_confirmed'])
            self.assertFalse(r['training_activated']);self.assertFalse(r['training_authorized'])
            self.assertEqual(sha(ROOT/r['evidence_plate']),r['evidence_plate_sha256'])

    def test_bbox_labels_are_not_visible_mask_identity(self):
        r=next(r for r in self.rows if r['sample_id']=='E7V02')
        self.assertIn(r['native_label'],('laptop','laptop_computer'))
        self.assertEqual(r['native_mask_hits'][0]['native_label'],'handbag')
        self.assertNotEqual(r['decision'],'trusted_negative')

    def test_handbag_partial_queries_are_not_independent_objects(self):
        summary=read(REPORT/'qualification-summary.json');g=summary['repeated_source_instance_groups']
        self.assertEqual(len(g),1);self.assertEqual(len(g[0]['candidate_ids']),5);self.assertEqual(g[0]['extra_observations'],4)
        self.assertEqual(g[0]['annotation_key'],'COCO:2078705')
        rep=next(r for r in self.rows if r['candidate_id']==g[0]['representative'])
        self.assertEqual(rep['sample_id'],'E7V02')

    def test_coverage_is_real_zero_not_synthetic_probe_count(self):
        r=read(DATA/'actual-query-coverage.json');self.assertEqual(r['actual_saved_E5_queries_checked'],21)
        self.assertEqual(r['additional_valid_high_confidence_Chair_negative_queries'],0)
        self.assertTrue(r['no_synthetic_queries_counted'])
        self.assertTrue(all(not x['accepted_new_high_confidence_Chair_query'] for x in r['records']))

    def test_existing_source_instance_not_counted_as_new_hard_object(self):
        d=read(DATA/'historical-evidence-deduplication-v2.json')
        self.assertEqual(d['strong_existing_handbag_instance_matches'],5)
        self.assertEqual(d['exact_native_annotation_reference_matches_existing41'],9)
        self.assertEqual(d['previously_seen_E6_footprints'],13)
        self.assertEqual(d['old_region_full_containment_queries'],0)
        self.assertEqual(d['new_trusted_source_instances'],0)

    def test_unverified_rights_candidates_never_cleared_automatically(self):
        m=read(DATA/'licensed-training-rights-candidates-v1.json');self.assertEqual(len(m['records']),64)
        for r in m['records']:
            self.assertFalse(r['license_verified_this_phase']);self.assertFalse(r['photo_acquisition_authorized'])
            self.assertEqual(r['rights_status'],'individually_unverified')
        plan=read(REPORT/'licensed-training-acquisition-proposal.json')
        self.assertFalse(plan['approved']);self.assertFalse(plan['photo_acquisition_ready'])
        self.assertEqual(plan['total_native_furniture_boxes'],268)
        self.assertEqual(plan['categories']['desk']['native_boxes'],0)

    def test_no_inference_training_or_downloader_entrypoints(self):
        for p in HERE.glob('*.py'):
            if p.name.startswith('test_'):continue
            tree=ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                    self.assertNotIn(n.func.attr,('from_pretrained','backward','step','urlopen','urlretrieve','optimizer'))

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload,cls.sources=load_dashboard();cls.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler_for(cls.payload,cls.sources))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2)
    def request(self,path,method='GET',headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5);c.request(method,path,headers=headers or {});r=c.getresponse();status=r.status;h=dict(r.getheaders());data=r.read();c.close();return status,h,data

    def test_dashboard_counts_and_all21_rows(self):
        status,h,data=self.request('/api/data');self.assertEqual(status,200)
        p=json.loads(data);self.assertEqual(len(p['records']),21);self.assertEqual(p['summary']['decisions'],dict(unverified_candidate=5,protected_unsuitable=16))
        self.assertIn("default-src 'none'",h['Content-Security-Policy'])

    def test_only_training_sources_served(self):
        for ident in self.sources:
            status,_,data=self.request('/sources/'+urllib.parse.quote(ident,safe=''))
            self.assertEqual(status,200);self.assertEqual(hashlib.sha256(data).hexdigest(),self.sources[ident][1])
        self.assertEqual(self.request('/sources/reserved-evaluation')[0],404)

    def test_every_plate_and_static_asset_loads(self):
        for r in self.payload['records']:self.assertEqual(self.request('/assets/'+r['sample_id']+'.jpg')[0],200)
        for path in ('/','/app.js','/style.css'):
            status,_,data=self.request(path);self.assertEqual(status,200);self.assertTrue(data)

    def test_mutations_and_path_escape_rejected(self):
        for method in ('POST','PUT','DELETE'):self.assertEqual(self.request('/api/data',method)[0],405)
        for path in ('/../../models/model.safetensors','/assets/%2e%2e/input-binding.json','/api/data?file=secrets','/sources/unknown'):
            self.assertEqual(self.request(path)[0],404)
        self.assertEqual(self.request('/api/data',headers={'Host':'untrusted.invalid'})[0],403)

    def test_csv_has21_actual_decisions_and_no_human_flags(self):
        status,_,data=self.request('/decisions.csv');self.assertEqual(status,200)
        rows=list(csv.DictReader(io.StringIO(data.decode())));self.assertEqual(len(rows),21)
        self.assertTrue(all(r['human_reviewed']=='False' and r['training_activated']=='False' for r in rows))

if __name__=='__main__':unittest.main()
