import copy
import hashlib
from pathlib import Path
import sys
import unittest
import prepare_pilot as p

class Tests(unittest.TestCase):
    def test_full_source_group_bridge_includes_excluded_photos(self):
        rows=[dict(image_id=str(i),source_identity='photo'+str(i),photographer_identity=a,metadata_original_md5=m) for i,a,m in [(1,'authorA','one'),(2,'authorA','shared'),(3,'authorB','shared'),(4,'authorB','four')]]
        actual=p.grouped(copy.deepcopy(rows))
        self.assertEqual(actual[0]['identity_group'],actual[3]['identity_group'])
        subset=p.grouped([copy.deepcopy(rows[0]),copy.deepcopy(rows[3])])
        self.assertNotEqual(subset[0]['identity_group'],subset[1]['identity_group'])

    def rows(self):
        rows=[]
        for i in range(210):
            c=p.d.CLASSES[i%7];ident=f'{i:016x}'
            rows.append(dict(image_id=ident,source_partition='test' if i<90 else 'validation',source_identity='photo'+ident,metadata_original_md5='hash'+ident,identity_group='group'+ident,eligible_box_counts={c:1},native_box_counts={c:1},ordinary_box_counts={c:1},indoor_evidence_rank=1,complete_visible_boxes=1,indoor_title_matches=[],indoor_context_objects=[]))
        return rows

    def test_deterministic_selection_and_panel_coverage(self):
        rows=self.rows();a,sa=p.choose(rows);b,sb=p.choose(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(sa,sb)
        self.assertEqual({k:v['images'] for k,v in sa.items()},dict(train=60,development=30,frozen_evaluation=60))
        self.assertTrue(all(sa[k]['category_images'][c]>0 for k in sa for c in p.d.CLASSES))
        p.assert_proposal_isolation(a)

    def test_unknown_indoor_excluded_and_no_fabricated_verification(self):
        rows=self.rows();rows[0]['indoor_evidence_rank']=0
        chosen,_=p.choose(rows)
        self.assertNotIn(rows[0]['image_id'],{r['image_id'] for r in chosen})
        self.assertTrue(all(r['image_sha256'] is None and r['indoor_verified'] is None and r['license_verified'] is False for r in chosen))
        chosen[0]['license_verified']=True
        with self.assertRaises(ValueError):p.assert_proposal_isolation(chosen)

    def test_duplicate_and_source_partition_guards(self):
        rows,_=p.choose(self.rows())
        with self.assertRaises(ValueError):p.assert_proposal_isolation([rows[0],rows[0]])
        row=copy.deepcopy(rows[0]);row['proposed_split']='train'
        with self.assertRaises(ValueError):p.assert_proposal_isolation([row])
        a,b=copy.deepcopy(rows[:2]);b['metadata_original_md5']=a['metadata_original_md5']
        with self.assertRaises(ValueError):p.assert_proposal_isolation([a,b])

    def test_v7_dictionary_header_and_ids(self):
        from metadata_dictionary import read_dictionary
        names=read_dictionary(p.META/'oidv7-class-descriptions-boxable.csv')
        self.assertEqual(len(names),601);self.assertNotIn('LabelName',names)
        self.assertEqual(names['/m/0dtln'],'Lamp')

    def test_real_proposal_rebuild_is_identical(self):
        import contextlib,io,tempfile
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'rebuilt'
            with contextlib.redirect_stdout(io.StringIO()):p.build(p.META,out)
            self.assertEqual(p.d.read(out/'proposed-image-acquisition.json'),p.d.read(p.META/'pilot-proposal-v4/proposed-image-acquisition.json'))
            self.assertEqual(p.d.read(out/'eligibility-report.json'),p.d.read(p.META/'pilot-proposal-v4/eligibility-report.json'))

    def test_real_source_integrity_and_approved_budgets(self):
        ledger=p.d.read(p.META/'transfer-ledger.json')
        self.assertTrue(ledger['complete']);self.assertEqual(ledger['actual_body_bytes'],205865596)
        self.assertEqual(ledger['charged_bytes'],205865596);self.assertEqual(len(ledger['objects']),8)
        self.assertTrue(all(r['attempts']==1 for r in ledger['objects'].values()))
        for name,r in ledger['objects'].items():
            path=p.META/name
            self.assertEqual(path.stat().st_size,r['bytes']);self.assertEqual(p.d.sha(path),r['sha256'])
            h=hashlib.md5()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            self.assertEqual(h.hexdigest(),r['md5_hex'])
        used=sum(path.stat().st_blocks*512 for path in [p.META,*p.META.rglob('*')])
        self.assertLess(used,1073741824)

    def test_real_manifest_native_classes_and_isolation(self):
        actual=p.d.read(p.META/'pilot-proposal-v4/proposed-image-acquisition.json')
        self.assertEqual(actual['image_count'],150);p.assert_proposal_isolation(actual['images'])
        self.assertEqual(set(actual['native_mapping']['mapping'].values()),set(p.d.CLASSES))
        self.assertEqual(actual['selection_implementation_sha256'],p.d.sha(p.__file__))
        self.assertTrue(all(r['indoor_evidence_rank']>=1 for r in actual['images']))
        for r in actual['images']:
            for g in r['annotations']:self.assertEqual(actual['native_mapping']['mapping'][g['source_mid']],g['label'])
        self.assertEqual(actual['summary']['development']['category_eligible_boxes']['lamp'],3)
        self.assertTrue(all(v>=10 for v in actual['summary']['frozen_evaluation']['category_eligible_boxes'].values()))
        # An acquisition proposal must not be passed off as a verified evaluation bundle.
        with self.assertRaises(ValueError):p.d.validate_public_bundle(p.META/'pilot-proposal-v4/proposed-image-acquisition.json')

if __name__=='__main__':unittest.main()
