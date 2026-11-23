"""CPU-only tests of actual completed evidence;all source network access closed."""
import ast,gzip,urllib.robotparser,unittest
from common9 import *
from transport9 import Transport
class CompletedTests(unittest.TestCase):
    def test_actual_provider_rule_denies_all64without_identity_rotation(self):
        n=read(OUT/'network-final-ledger.json');r=n['requests'][0];body=(OUT/r['file']).read_bytes()
        self.assertEqual(sha(OUT/r['file']),r['sha256'])
        if next((v for k,v in r['headers'].items() if k.lower()=='content-encoding'),None)=='gzip':body=gzip.decompress(body)
        p=urllib.robotparser.RobotFileParser();p.parse(body.decode().splitlines())
        self.assertTrue(all(not p.can_fetch(UA,row['original_source_page_resolver']) for row in manifest()['records']))
        self.assertEqual(len(n['requests']),1);self.assertEqual(n['candidate_evidence_attempts'],[])
    def test_closed_audit_cannot_resume_network(self):
        with self.assertRaises(PermissionError):Transport(resume=True)
    def test_exact64_dispositions_and_no_false_verification(self):
        records=read(OUT/'license-attribution-ledger.json')['records']
        self.assertEqual({r['image_id'] for r in records},{r['image_id'] for r in manifest()['records']})
        self.assertEqual(len(records),64)
        self.assertTrue(all(r['rights_status']==STATUS_B and not r['individual_source_attempted'] and r['rights_verified_utc'] is None and r['photograph_license_verified'] is None for r in records))
    def test_access_denial_not_misreported_as_copyright_restriction(self):
        r=read(REPORT/'audit-summary.json');self.assertEqual([r['rights_supported'],r['unresolved'],r['unsuitable']],[0,64,0])
        self.assertEqual(read(REPORT/'source-access-evidence.json')['HTTP_status'],200)
    def test_no_image_bodies_heads_or_gpu_calls(self):
        r=read(REPORT/'audit-summary.json')
        for k in ('photograph_bodies','photo_object_HEADs','individual_rights_IDs_attempted','GPU_forwards','new_training_labels','partition_assignments'):self.assertEqual(r[k],0)
    def test_transfer_reconciles_measured_TLS_and_conservative_allowance(self):
        n=read(OUT/'network-final-ledger.json')
        self.assertEqual(sum(r['actual_body_bytes'] for r in n['requests']),n['actual_response_body_bytes'])
        self.assertEqual(sum(r['sent_TLS_bytes'] for r in n['requests']),n['measured_TLS_sent_bytes'])
        self.assertEqual(sum(r['received_TLS_bytes'] for r in n['requests']),n['measured_TLS_received_bytes'])
        self.assertEqual(n['conservative_transfer_charge'],sum(r['conservative_charge'] for r in n['requests']))
        self.assertLessEqual(n['conservative_transfer_charge'],CAP);self.assertLess(allocated(),DISK)
    def test_no_rights_supported_acquisition_list_is_fabricated(self):
        self.assertEqual(read(OUT/'proposed-training-acquisition.json')['records'],[])
        self.assertEqual(read(OUT/'proposed-independent-evaluation-acquisition.json')['records'],[])
        self.assertTrue(all(r['final_partition'] is None and not r['eligible_for_photograph_acquisition'] for r in read(OUT/'license-attribution-ledger.json')['records']))
    def test_native_annotations_and_context_remain_provisional(self):
        r=read(OUT/'local-candidate-audit.json')['records'];self.assertEqual(sum(len(x['native_annotations']) for x in r),786)
        self.assertEqual(sum(len(x['native_annotation_ids']) for x in r),268)
        self.assertTrue(all(not x['Chair_absence_inferred'] and not x['visual_content_verified'] and not x['annotation_completeness_verified'] for x in r))
    def test_identity_unknown_does_not_become_independent_group(self):
        r=read(OUT/'identity-audit.json');self.assertFalse(r['physical_scene_or_object_independence_verified'])
        self.assertTrue(all(not x['group_verified'] and not x['partition_assigned'] for x in r['records']))
    def test_LVIS_native_splits_are_not_project_partitions_or_rights(self):
        s=read(REPORT/'LVIS-existing-annotation-summary.json');v=read(OUT/'LVIS-existing-annotation-audit.json')
        self.assertEqual(s['matched_same_COCO_photographs'],64);self.assertEqual(s['native_split_photographs'],{'train':54,'val':10})
        self.assertEqual(s['core_instances'],120);self.assertEqual(s['total_native_instances'],159)
        self.assertEqual(s['core_categories']['cabinet']['instances'],48);self.assertEqual(s['core_categories']['desk']['instances'],2)
        self.assertTrue(all(r['photographic_rights']==STATUS_B and r['new_training_labels']==0 and r['project_Chair_absence_not_inferred'] for r in v['records']))
        self.assertTrue(all(not a['training_activated'] and not a['rights_supported'] and digest(a['original_annotation'])==a['sha256'] for a in v['annotations']))
    def test_LVIS_index_flags_are_only_original_selected_category_projection(self):
        from lvis9 import INDEX_RELEVANT
        tree=ast.parse((ROOT/'tools/phase7-acquisition-v1/core.py').read_text());parts={}
        for n in tree.body:
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name) and target.id in ('CORE','SUBTYPES'):parts[target.id]=ast.literal_eval(n.value)
        self.assertEqual(INDEX_RELEVANT,set(parts['CORE'])|set(parts['SUBTYPES']))
        self.assertEqual(len(INDEX_RELEVANT),23)
if __name__=='__main__':unittest.main()
