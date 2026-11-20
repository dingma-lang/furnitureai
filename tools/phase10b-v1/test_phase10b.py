import ast
import collections
import copy
import http.client
import importlib.util
import tempfile
import threading
import unittest
from unittest.mock import patch
from p10 import *
from state import Ledger,StopRun
from worker import validate,grouped_epoch,resource_limits
from tracking import record,score
from review_server import make_server,validate_submission

class PreparationTests(unittest.TestCase):
    def test_current_eligibility_and_old_manifest_unchanged(self):
        m=read(RAW/'eligibility-manifest.json');old=read(OLD/'prepared-manifest.json')
        self.assertEqual(m['rows'],old['rows']);self.assertEqual(m['eligible_boxes'],243)
        self.assertEqual((len(m['rows']),m['eligible_groups']),(78,75))
        self.assertEqual(m['annotation_recovery_activated'],0);self.assertFalse(m['training_authorized'])
        self.assertEqual(m['unreviewed_overlay_annotations_still_inactive'],33)
    def test_all130_visual_review_records_no_human_flags(self):
        q=read(RAW/'review/queue.json')['records'];t=read(RAW/'review/assistant-triage.json')['records']
        self.assertEqual(len(q),130);self.assertEqual(len({r['annotation_id'] for r in q}),130)
        self.assertEqual({r['id'] for r in q},{r['id'] for r in t})
        self.assertTrue(all(not r['human_reviewed'] and not r['training_active'] for r in t))
        self.assertEqual(len(list((RAW/'review/contacts').glob('*.jpg'))),11)
    def test_frozen_and_forged_rows_rejected_before_read(self):
        m=read(RAW/'eligibility-manifest.json');r=m['rows'][0]
        with patch('p10.sha',side_effect=AssertionError('Must deny before opening')):
            with self.assertRaises(PermissionError):image_path(r,'frozen_evaluation',m)
            q=dict(r,path='data/frozen.jpg')
            with self.assertRaises(PermissionError):image_path(q,q['internal_partition'],m)
            q=next(x for x in m['rows'] if x['internal_partition']=='inner_development')
            with self.assertRaises(PermissionError):image_path(q,'optimization',m)
    def test_source_groups_no_leakage(self):
        m=read(RAW/'eligibility-manifest.json');rows=m['rows']
        a={r['identity_group'] for r in rows if r['internal_partition']=='optimization'}
        b={r['identity_group'] for r in rows if r['internal_partition']=='inner_development'}
        self.assertFalse(a&b);self.assertEqual((len(a),len(b)),(56,19))
    def test_grouped_epoch_last_batch_and_determinism(self):
        rows=[r for r in read(RAW/'eligibility-manifest.json')['rows'] if r['internal_partition']=='optimization']
        g=grouped_epoch(rows,20261118,0);self.assertEqual(len(g),15);self.assertEqual(len(g[-1]),3)
        self.assertEqual(g,grouped_epoch(rows,20261118,0));self.assertNotEqual(g,grouped_epoch(rows,20261118,1))
        self.assertEqual(sum(len(grouped_epoch(rows,20261118,e)) for e in range(5)),75)
    def test_preapproval_gate_precedes_any_runtime(self):
        with patch('worker.RUN',RAW/'must-not-exist'):
            with self.assertRaises(PermissionError):validate(RAW/'authorization-proposal.json')
            self.assertFalse((RAW/'must-not-exist').exists())
    def test_gpu_smoke_is_weight_preserving_and_gates_optimizer(self):
        text=(HERE/'worker.py').read_text()
        self.assertIn('if smoke_only:',text);self.assertIn('frozen_and_head_weights_unchanged=True',text)
        self.assertLess(text.index("elif not smoke_only:"),text.index('import torch\n'))
        self.assertLess(text.index('if smoke_only:\n                            if cursor'),text.index('scaler.step(optimizer)'))
    def test_authorization_binding_rejects_unapproved_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';save(p,dict(approved=True,model_updates_authorized=True,approved_by='fixture',approval_reference='fixture'))
            with self.assertRaises(ValueError):validate(p)
    def test_model_imports_only_inside_authorized_run(self):
        tree=ast.parse((HERE/'worker.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertEqual(run.body[0].value.func.id,'validate')
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'torch' in ast.unparse(n) for n in tree.body))
        self.assertIn('local_files_only=True',(HERE/'worker.py').read_text())
    def test_vram_disk_and_storage_safeguards_without_gpu(self):
        resource_limits(2*2**30,3*2**30,[10*2**30,10*2**30],0)
        for args in [(7*2**30,7*2**30,[10*2**30]*2,0),(0,0,[4*2**30,10*2**30],0),(0,0,[10*2**30]*2,8*2**30)]:
            with self.assertRaises(StopRun):resource_limits(*args)
    def test_history_preserves_required_scope(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):record(d,{})
            r=dict(experiment_id='fixture',dataset_version='x',dataset_sha256='a'*64,git_commit='x',checkpoint_sha256=None,
                   settings={},metrics=None,resources={},decision='investigate',next_experiment='review',evaluation_scope='not_evaluated')
            record(d,r);record(d,r);self.assertEqual(len(list(Path(d).glob('*.json'))),2)
            with self.assertRaises(ValueError):record(d,dict(r,evaluation_scope='independent_verified'))

class StateTests(unittest.TestCase):
    def ledger(self,d,clock=None):
        b=Ledger(d,{'config':'fixture'},clock=clock or (lambda:0));b.start('heads-original-lr1e-5');return b
    def test_no_model_weights_updated_during_snapshot_recovery(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d);before=torch.ones(2);state=dict(cursor=0,weights={'fixture':before.clone()},optimizer={},rng={},binding=b.binding)
            r=b.checkpoint(state);b.committed(r);p=b.resume();loaded=torch.load(p,weights_only=True)
            torch.testing.assert_close(loaded['weights']['fixture'],before);torch.testing.assert_close(before,torch.ones(2))
    def test_uncertain_call_never_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d);b.reserve(0)
            with self.assertRaises(RuntimeError):b.resume()
            with self.assertRaises(RuntimeError):b.reserve(0)
    def test_receipt_hash_and_binding_checked(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d);r=b.checkpoint(dict(cursor=0));b.committed(r);p=b.resume();p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):b.resume()
            c=Ledger(d,{'config':'changed'})
            with self.assertRaises(ValueError):c.start('heads-original-lr1e-5')
    def test_budget_caps_without_optimizer_calls(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d)
            # Bookkeeping fixtures represent calls; no optimizer/model is constructed.
            for fit in b.data['fits']:
                b.start(fit)
                for cursor in range(150):
                    b.reserve(cursor);rec=b.checkpoint(dict(cursor=cursor+1));b.committed(rec)
                with self.assertRaises(StopRun):b.reserve(150)
            self.assertEqual(sum(x['applied'] for x in b.data['fits'].values()),300)
    def test_crashes_do_not_refund_time_and_per_fit_bound(self):
        with tempfile.TemporaryDirectory() as d:
            clock=[0];b=self.ledger(d,lambda:clock[0]);self.assertEqual(b.data['charged_seconds'],30)
            clock[0]=60;b.tick();self.assertEqual(b.data['charged_seconds'],90)
            c=Ledger(d,b.binding,clock=lambda:clock[0]);c.start('heads-original-lr1e-5');self.assertEqual(c.data['charged_seconds'],120)
            clock[0]=1800
            with self.assertRaises(StopRun):c.tick()
    def test_snapshot_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d);r1=b.checkpoint(dict(cursor=0));r2=b.checkpoint(dict(cursor=0))
            self.assertNotEqual(r1['path'],r2['path']);self.assertTrue((Path(d)/r1['path']).exists())
    def test_outside_snapshot_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.ledger(d)
            with self.assertRaises(ValueError):b.committed(dict(path='../outside.pt',sha256='x',cursor=0))

class ReviewTests(unittest.TestCase):
    def test_confirmation_evidence_and_exact_membership_required(self):
        row=read(RAW/'review/queue.json')['records'][0];known={row['id']:row}
        r=dict(id=row['id'],decision='visually_ambiguous',reviewer='fixture',evidence='Test fixture only, not submitted.',confirmed=True)
        self.assertFalse(validate_submission(r,known)['training_active'])
        for bad in [dict(r,confirmed=False),dict(r,reviewer=''),dict(r,evidence=''),dict(r,id='unknown')]:
            with self.assertRaises(ValueError):validate_submission(bad,known)
    def test_loopback_ui_origin_and_traversal_guards(self):
        server=make_server(0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET','/api/queue');r=c.getresponse();payload=json.loads(r.read());self.assertEqual(len(payload['records']),130)
            c.request('GET','/photo/../../etc/passwd');r=c.getresponse();self.assertEqual(r.status,404);r.read()
            c.request('POST','/api/review','{}',{'Content-Type':'application/json','Origin':'https://example.com'});r=c.getresponse();self.assertEqual(r.status,403);r.read()
            c.close()
        finally:server.shutdown();server.server_close();thread.join()

class MetricTests(unittest.TestCase):
    def test_internal_native_metric_and_unknown_fp_scopes(self):
        from internal import evaluate
        row=copy.deepcopy(next(r for r in read(RAW/'eligibility-manifest.json')['rows'] if r['internal_partition']=='inner_development' and r['source']=='phase6d_coco'))
        valid=[a for a in row['annotations'] if not a['ignore']]
        pred=[dict(label=a['label'],box=a['box'],score=.9) for a in valid]
        pred.append(dict(label='lamp',box=[0,0,.05,.05],score=.8))
        result=evaluate([row],{row['image_id']:pred});self.assertEqual(result['tp'],len(valid));self.assertEqual(result['fp'],0)
        row['historical_partition']='frozen_evaluation'
        with self.assertRaises(PermissionError):evaluate([row],{row['image_id']:pred})
    def test_matched_recall_cannot_reward_suppression(self):
        cats={c:dict(tp=1,ap75=.5) for c in ('sofa','chair','bed')}
        base=dict(recall=.9,ap75=.5,f1=.6,categories=cats)
        m=dict(base,pr_curve=[dict(recall=.5,precision=.99)])
        self.assertEqual(score(m,base)[0],0)

if __name__=='__main__':unittest.main()
