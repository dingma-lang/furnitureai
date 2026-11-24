"""CPU-only end-to-end tests. Every photograph, query and checkpoint is synthetic."""
import copy
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from furniture_bench.cache import make_contract, read_prediction_cache
from furniture_bench.cached_pipeline import (CacheConflictError, _prepare, _publish,
    load_completed_run, run_cached_detection, stage_keys)
from furniture_bench.data import digest, read_json, write_json
from furniture_bench.persistence import artifact_lock
from furniture_bench.postprocessing import FilterConfig
from synthetic_pipeline_fixture import create_fixture, SyntheticDetector, semantic_result


def run_worker(bundle, output, log, queue, start=None, fail_stage=None, ready=None, release=None):
    """Fresh process, cooperative locks, optional abrupt death inside an atomic write."""
    from furniture_bench import persistence
    manifest, queries, spec, weights, processor = bundle
    class CountingDetector(SyntheticDetector):
        def detect(self, image):
            fd=os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:os.write(fd,b'synthetic-call\n')
            finally:os.close(fd)
            if fail_stage == 'second-forward' and self.calls == 1:os._exit(24)
            return super().detect(image)
    detector=CountingDetector(queries,spec)
    original=persistence.atomic_write
    def interrupted(path, writer, **kwargs):
        path=Path(path)
        selected=((fail_stage == 'raw' and path.parent.name == 'raw')
                  or (fail_stage == 'adapter.json' and path.name == 'adapter.json')
                  or (path.name == fail_stage and 'runs' in path.parts))
        if not selected:return original(path,writer,**kwargs)
        def partial(stream):
            stream.write(b'{"synthetic-partial":');stream.flush();os.fsync(stream.fileno())
            if ready is not None:ready.set()
            if release is not None and not release.wait(15):os._exit(71)
            os._exit(23)
        return original(path,partial,**kwargs)
    try:
        if start is not None and not start.wait(15):raise TimeoutError('Test barrier timed out')
        with patch.object(persistence,'atomic_write',side_effect=interrupted):
            result=run_cached_detection(manifest,output,detector,spec=spec,model_files=weights,processor_files=processor)
        queue.put(dict(status='complete',run_key=result['run_key'],calls=detector.calls,
                       counts=result['artifacts']['evaluation.json']['detection']))
    except Exception as exc:
        queue.put(dict(status='error',type=type(exc).__name__,message=str(exc)))


def publish_worker(path, value, queue, start):
    if not start.wait(15):raise TimeoutError('Test barrier timed out')
    try:_publish(path,value);queue.put('committed')
    except CacheConflictError:queue.put('conflict')


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='E14-synthetic-test-')
        self.root=Path(self.temp.name)
        self.bundle=create_fixture(self.root/'inputs')
        self.manifest,self.queries,self.spec,self.weights,self.processor=self.bundle
        self.output=self.root/'cache'
        self.detector=SyntheticDetector(self.queries,self.spec)

    def tearDown(self):self.temp.cleanup()

    def run_pipeline(self, **kwargs):
        return run_cached_detection(self.manifest,self.output,self.detector,
            spec=self.spec,model_files=self.weights,processor_files=self.processor,**kwargs)

    def change(self, callback):
        value=read_json(self.manifest);callback(value['images']);write_json(self.manifest,value)


class EndToEndTests(PipelineFixture):
    def test_exact_prechange_baseline_and_verified_reload(self):
        baseline=read_json(Path(__file__).resolve().parents[1]/'reports/phase10e14-v1/baseline-before.json')
        first=self.run_pipeline();docs=first['artifacts']
        self.assertEqual(semantic_result(docs['predictions.json'],docs['evaluation.json']),baseline['semantic'])
        self.assertEqual(self.detector.calls,3)
        second=self.run_pipeline()
        self.assertEqual(first,second);self.assertEqual(self.detector.calls,3)
        self.assertEqual(docs['predictions.json']['images'][1]['width'],80)
        self.assertEqual(docs['predictions.json']['images'][1]['height'],120)
        self.assertEqual(set(docs['evaluation.json']['per_category']),set(self.spec['categories']))
        self.assertTrue(all(r['recall']==1. for r in docs['evaluation.json']['per_category'].values()))

    def test_complete_queries_scores_logits_and_boxes_survive_raw_reload(self):
        result=self.run_pipeline()
        for path in (self.output/'raw').glob('*.json'):
            value=read_json(path);payload=value['payload']
            self.assertEqual(payload['detections'],self.queries[payload['image_id']])
            self.assertEqual(payload['box_frame'],'oriented-xyxy-pixels')
            self.assertEqual(len(payload['detections'][0]['logits']),7)
        self.assertEqual(len(result['artifacts']['postprocessed.json']['images']),3)

    def test_annotation_changes_reuse_raw_and_recompute_metrics(self):
        old=self.run_pipeline()
        self.change(lambda rows:rows[0]['objects'][0].update(box=[2,25,16,39]))
        new=self.run_pipeline()
        self.assertEqual(self.detector.calls,3)
        self.assertNotEqual(old['run_key'],new['run_key'])
        self.assertEqual(new['artifacts']['evaluation.json']['detection']['false_negatives'],1)
        self.assertEqual(new['artifacts']['evaluation.json']['detection']['false_positives'],1)
        self.assertTrue((Path(old['folder'])/'completed.json').exists())

    def test_incomplete_or_protected_regions_do_not_become_negatives(self):
        self.change(lambda rows:rows[0].update(completeness={c:'partial' for c in self.spec['categories']},
                                             ignore_regions=[],protected_regions=[]))
        result=self.run_pipeline();metrics=result['artifacts']['evaluation.json']
        self.assertEqual(metrics['detection']['true_positives'],8)
        self.assertEqual(metrics['detection']['false_positives'],0)
        self.assertGreater(metrics['ignored_predictions'],0)
        context=result['artifacts']['supervision.json']['images'][0]['context']
        self.assertEqual(context['completeness']['chair'],'partial')
        self.assertNotIn('negative_targets',context)

    def test_missing_optional_ignore_list_supported(self):
        self.change(lambda rows:rows[2].pop('ignore_regions'))
        result=self.run_pipeline()
        self.assertEqual(result['artifacts']['evaluation.json']['detection']['false_positives'],0)

    def test_unannotated_is_not_an_exhaustive_empty_annotation(self):
        self.change(lambda rows:rows[0].update(objects=None,completeness={c:'unknown' for c in self.spec['categories']}))
        result=self.run_pipeline();metrics=result['artifacts']['evaluation.json']
        self.assertEqual(metrics['unannotated_images'],1)
        self.assertEqual(metrics['detection']['true_positives'],5)

    def test_different_postprocessing_reuses_raw_without_lowering_acquisition_floor(self):
        a=self.run_pipeline();b=self.run_pipeline(policy=FilterConfig(threshold=.95))
        self.assertEqual(self.detector.calls,3);self.assertNotEqual(a['run_key'],b['run_key'])
        self.assertEqual(b['artifacts']['evaluation.json']['detection']['true_positives'],0)
        self.assertEqual(b['artifacts']['evaluation.json']['detection']['false_negatives'],8)

    def test_schema_or_corrupt_raw_reuse_is_rejected_and_preserved(self):
        self.run_pipeline();p=next((self.output/'raw').glob('*.json'));value=read_json(p)
        value['schema_version']=1;write_json(p,value);before=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'v2 prediction cache'):self.run_pipeline()
        self.assertEqual(p.read_bytes(),before);self.assertEqual(self.detector.calls,3)

    def test_committed_artifact_corruption_is_not_repaired_silently(self):
        result=self.run_pipeline();p=Path(result['folder'])/'evaluation.json';p.write_text('{"partial":')
        with self.assertRaisesRegex(ValueError,'integrity'):self.run_pipeline()
        self.assertEqual(p.read_text(),'{"partial":')

    def test_stale_completion_request_is_rejected(self):
        result=self.run_pipeline();p=Path(result['folder'])/'completed.json';value=read_json(p)
        value['request']['input_sha256']='f'*64;write_json(p,value)
        with self.assertRaisesRegex(ValueError,'Stale'):self.run_pipeline()

    def test_input_identity_changes_midrun_withhold_completion(self):
        real=self.detector.detect
        def changed(image):
            result=real(image)
            if self.detector.calls==1:self.change(lambda rows:rows[0].update(source_group_id='edited-during-run'))
            return result
        self.detector.detect=changed
        with self.assertRaisesRegex(ValueError,'manifest changed during processing'):self.run_pipeline()
        self.assertFalse(list((self.output/'runs').glob('*/completed.json')))

    def test_partition_group_and_duplicate_content_isolation(self):
        self.change(lambda rows:rows[2].update(source_group_id=rows[0]['source_group_id']))
        with self.assertRaisesRegex(ValueError,'group spans'):self.run_pipeline()
        self.assertEqual(self.detector.calls,0)
        self.bundle=create_fixture(self.root/'inputs')
        self.change(lambda rows:rows[2].update(file=rows[0]['file'],image_sha256=rows[0]['image_sha256']))
        with self.assertRaisesRegex(ValueError,'Duplicate image content'):self.run_pipeline()

    def test_crossrun_partition_reassignment_is_rejected(self):
        self.run_pipeline()
        self.change(lambda rows:rows[0].update(partition='development'))
        with self.assertRaisesRegex(ValueError,'partition assignment'):self.run_pipeline()

    def test_crossrun_source_group_conflicts_are_rejected(self):
        self.run_pipeline()
        value=read_json(self.manifest);row=value['images'][2]
        row['source_group_id']=value['images'][0]['source_group_id']
        write_json(self.manifest,dict(schema_version=1,images=[row]))
        with self.assertRaisesRegex(ValueError,'partition assignment'):self.run_pipeline()

    def test_known_family_cross_partition_is_rejected(self):
        def edited(rows):
            rows[0]['identity_family_group_ids']=['known-synthetic-family']
            rows[2]['identity_family_group_ids']=['known-synthetic-family']
        self.change(edited)
        with self.assertRaisesRegex(ValueError,'known identity families'):self.run_pipeline()
        self.assertEqual(self.detector.calls,0)

    def test_model_file_and_prompt_identity_are_checked_before_processing(self):
        self.spec['prompts']=list(reversed(self.spec['prompts']))
        with self.assertRaisesRegex(ValueError,'query configuration'):self.run_pipeline()
        self.spec['prompts']=list(reversed(self.spec['prompts']))
        self.weights['weights'].write_text('changed synthetic weights')
        with self.assertRaisesRegex(ValueError,'checkpoint identity'):self.run_pipeline()
        self.assertEqual(self.detector.calls,0)

    def test_inconsistent_logits_are_not_saved_under_an_incorrect_query_order(self):
        self.detector.queries['SYNTHETIC-ONLY-0'][0]['logits']=list(reversed(self.detector.queries['SYNTHETIC-ONLY-0'][0]['logits']))
        with self.assertRaisesRegex(ValueError,'ordered query logits'):self.run_pipeline()
        self.assertFalse(list((self.output/'raw').glob('*.json')))

    def test_changed_image_bytes_recompute_only_the_affected_raw_entry(self):
        first=self.run_pipeline()
        value=read_json(self.manifest);path=self.manifest.parent/value['images'][0]['file']
        path.write_bytes(path.read_bytes()+b'SYNTHETIC trailing PNG bytes')
        self.change(lambda rows:rows[0].update(image_sha256=digest(path)))
        second=self.run_pipeline()
        self.assertEqual(self.detector.calls,4)
        self.assertNotEqual(first['run_key'],second['run_key'])
        self.assertEqual(len(list((self.output/'raw').glob('*.json'))),4)

    def test_unsupported_orientation_does_not_silently_reuse_pixels(self):
        self.run_pipeline();self.spec['preprocessing']['orientation']='ignore-EXIF'
        with self.assertRaisesRegex(ValueError,'EXIF-transposed'):self.run_pipeline()
        self.assertEqual(self.detector.calls,3)

    def test_caller_mutation_cannot_change_an_inflight_request(self):
        policy=FilterConfig();real=self.detector.detect
        original_taxonomy=self.spec['taxonomy_version']
        def changed(image):
            result=real(image)
            policy.threshold=.95;self.spec['taxonomy_version']='externally-mutated'
            return result
        self.detector.detect=changed
        result=self.run_pipeline(policy=policy)
        self.assertEqual(result['artifacts']['evaluation.json']['detection']['true_positives'],8)
        self.assertEqual(result['artifacts']['predictions.json']['detector']['taxonomy_version'],original_taxonomy)

    def test_new_checkpoint_identity_uses_new_cache_without_rewriting_old_cache(self):
        self.run_pipeline();old={p:digest(p) for p in (self.output/'raw').glob('*.json')}
        from furniture_bench.cache import file_identity
        self.weights['weights'].write_text('new explicitly synthetic model contents')
        self.spec['model']['checkpoint_sha256']=file_identity(self.weights)['sha256']
        self.detector.spec=copy.deepcopy(self.spec)
        self.run_pipeline()
        self.assertEqual(self.detector.calls,6)
        self.assertEqual({p:digest(p) for p in old},old)

    def test_new_schema_is_rejected_not_migrated(self):
        self.run_pipeline();path=next((self.output/'raw').glob('*.json'))
        value=read_json(path);value['schema_version']=3;write_json(path,value)
        with self.assertRaisesRegex(ValueError,'v2 prediction cache'):self.run_pipeline()

    def test_raw_payload_corruption_is_detected_without_reprocessing(self):
        self.run_pipeline();path=next((self.output/'raw').glob('*.json'))
        value=read_json(path);value['payload']['detections'][0]['score']=.01;write_json(path,value)
        before=digest(path)
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):self.run_pipeline()
        self.assertEqual(self.detector.calls,3);self.assertEqual(digest(path),before)

    def test_legacy_outputs_are_unchanged_and_not_implicitly_migrated(self):
        self.output.mkdir();old=self.output/'predictions.json';old.write_text('{"schema_version":1}')
        before=digest(old)
        with self.assertRaisesRegex(ValueError,'legacy/unmarked'):self.run_pipeline()
        self.assertEqual(digest(old),before);self.assertFalse((self.output/'adapter.json').exists())

    def test_candidate_floor_enforced_and_filtered_only_detectors_rejected(self):
        self.spec['settings']['candidate_floor']=.2
        with self.assertRaisesRegex(ValueError,'below'):self.run_pipeline()
        self.spec['settings']['candidate_floor']=0.
        def filtered(_):self.detector.last_candidates=None;return [],0.
        self.detector.detect=filtered
        with self.assertRaisesRegex(ValueError,'no pre-NMS'):self.run_pipeline()


class InvalidationTests(PipelineFixture):
    def test_precise_stage_invalidation_matrix(self):
        _,prepared=_prepare(self.manifest,self.spec);contract=prepared[0][1]
        policy=FilterConfig();protocol=dict(iou_threshold=.5, matching='existing')
        original=stage_keys(contract,policy,protocol)
        cases={
            'image_bytes':('inference','image_sha256','e'*64,{'raw','postprocessed','evaluation','supervision'}),
            'preprocessing':('inference','preprocessing',dict(contract['inference']['preprocessing'],resize=[224,224]),{'raw','postprocessed','evaluation','supervision'}),
            'checkpoint':('inference','model',dict(self.spec['model'],checkpoint_sha256='e'*64),{'raw','postprocessed','evaluation'}),
            'prompt_order':('inference','prompts',list(reversed(self.spec['prompts'])),{'raw','postprocessed','evaluation'}),
            'category_order':('inference','categories',list(reversed(self.spec['categories'])),{'raw','postprocessed','evaluation'}),
            'taxonomy':('supervision','taxonomy_version','new-description-only',{'evaluation','supervision'}),
            'ground_truth':('supervision','annotations',[],{'evaluation','supervision'}),
            'completeness':('supervision','completeness',{c:'unknown' for c in self.spec['categories']},{'evaluation','supervision'}),
            'ignore_regions':('supervision','ignore_regions',[],{'evaluation','supervision'}),
            'protected_regions':('supervision','protected_regions',[],{'evaluation','supervision'}),
            'source_group':('supervision','source_group_id','new-group',{'evaluation','supervision'}),
        }
        for name,(section,key,value,expected) in cases.items():
            with self.subTest(name=name):
                c=copy.deepcopy(contract);c[section][key]=value;i=c['inference']
                changed=make_contract(image_id=i['image_id'],image_sha256=i['image_sha256'],model=i['model'],
                    preprocessing=i['preprocessing'],categories=i['categories'],prompts=i['prompts'],
                    query_categories=i['query_categories'],inference_settings=i['settings'],supervision=c['supervision'])
                keys=stage_keys(changed,policy,protocol)
                self.assertEqual({k for k in keys if keys[k]!=original[k]},expected)
        for name,p in [('NMS',FilterConfig(nms_iou=.9)),('confidence',FilterConfig(threshold=.5))]:
            keys=stage_keys(contract,p,protocol)
            self.assertEqual({k for k in keys if keys[k]!=original[k]},{'postprocessed','evaluation'},name)
        keys=stage_keys(contract,policy,dict(protocol,iou_threshold=.75))
        self.assertEqual({k for k in keys if keys[k]!=original[k]},{'evaluation'})


class ProcessTests(PipelineFixture):
    def processes(self, processes):
        for p in processes:p.join(20)
        for p in processes:
            if p.is_alive():p.terminate();p.join(5);self.fail('Synthetic worker failed to finish')

    def test_same_identity_concurrent_run_is_once_and_not_double_counted(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();start=ctx.Event();log=self.root/'calls'
        ps=[ctx.Process(target=run_worker,args=(self.bundle,self.output,log,q,start)) for _ in range(2)]
        for p in ps:p.start()
        start.set();self.processes(ps)
        results=[q.get(timeout=2) for _ in ps]
        self.assertTrue(all(r['status']=='complete' for r in results),results)
        self.assertEqual(len(log.read_text().splitlines()),3)
        self.assertEqual(results[0]['run_key'],results[1]['run_key'])
        self.assertEqual(results[0]['counts']['true_positives'],8)
        self.assertEqual(len(list((self.output/'runs').glob('*/completed.json'))),1)

    def test_different_identity_writes_and_explicit_conflicts(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();start=ctx.Event()
        ps=[ctx.Process(target=publish_worker,args=(self.root/f'{i}.json',{'identity':i},q,start)) for i in range(3)]
        for p in ps:p.start()
        start.set();self.processes(ps)
        self.assertEqual([q.get(timeout=2) for _ in ps],['committed']*3)
        for i in range(3):self.assertEqual(read_json(self.root/f'{i}.json'),{'identity':i})
        q=ctx.Queue();start=ctx.Event();path=self.root/'conflict.json'
        ps=[ctx.Process(target=publish_worker,args=(path,{'candidate':i},q,start)) for i in range(2)]
        for p in ps:p.start()
        start.set();self.processes(ps)
        self.assertEqual(sorted(q.get(timeout=2) for _ in ps),['committed','conflict'])
        self.assertIn(read_json(path)['candidate'],[0,1])

    def test_abrupt_cache_writer_and_reader_then_verified_resume(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();ready=ctx.Event();release=ctx.Event();log=self.root/'calls'
        p=ctx.Process(target=run_worker,args=(self.bundle,self.output,log,q,None,'raw',ready,release));p.start()
        try:
            self.assertTrue(ready.wait(15));self.assertFalse(list((self.output/'raw').glob('*.json')))
            self.assertTrue(list((self.output/'raw').glob('*.pending')))
            _,prepared=_prepare(self.manifest,self.spec);contract=prepared[0][1]
            with self.assertRaises(FileNotFoundError):
                read_prediction_cache(self.output/'raw'/(contract['inference_key']+'.json'),contract)
            folder=next((self.output/'runs').iterdir())
            with self.assertRaises(FileNotFoundError):load_completed_run(folder)
        finally:release.set();self.processes([p])
        self.assertEqual(p.exitcode,23)
        result=self.run_pipeline()
        self.assertEqual(result['artifacts']['evaluation.json']['detection']['true_positives'],8)
        self.assertEqual(self.detector.calls,3)
        self.assertTrue(list((self.output/'raw').glob('*.pending')))  # Never delete crash evidence.

    def test_resume_reuses_earlier_photo_without_double_counting(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();log=self.root/'calls'
        p=ctx.Process(target=run_worker,args=(self.bundle,self.output,log,q,None,'second-forward'))
        p.start();self.processes([p]);self.assertEqual(p.exitcode,24)
        self.assertEqual(len(list((self.output/'raw').glob('*.json'))),1)
        result=self.run_pipeline()
        self.assertEqual(self.detector.calls,2)
        self.assertEqual(len(result['artifacts']['predictions.json']['images']),3)
        self.assertEqual(result['artifacts']['evaluation.json']['detection']['true_positives'],8)

    def test_distinct_image_identities_can_publish_concurrent_runs(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();start=ctx.Event();log=self.root/'calls'
        document=read_json(self.manifest);bundles=[]
        for i in (0,2):
            manifest=self.manifest.parent/f'subset-{i}.json'
            write_json(manifest,dict(schema_version=1,images=[document['images'][i]]))
            bundles.append((manifest,*self.bundle[1:]))
        ps=[ctx.Process(target=run_worker,args=(b,self.output,log,q,start)) for b in bundles]
        for p in ps:p.start()
        start.set();self.processes(ps);results=[q.get(timeout=2) for _ in ps]
        self.assertTrue(all(r['status']=='complete' for r in results),results)
        self.assertEqual(len(log.read_text().splitlines()),2)
        self.assertEqual(len(list((self.output/'runs').glob('*/completed.json'))),2)

    def test_interrupted_manifests_evaluation_and_final_marker(self):
        ctx=mp.get_context('spawn')
        for stage in ('input.json','evaluation.json','completed.json'):
            with self.subTest(stage=stage):
                root=self.root/stage;q=ctx.Queue();log=self.root/(stage+'.calls')
                p=ctx.Process(target=run_worker,args=(self.bundle,root,log,q,None,stage));p.start();self.processes([p])
                self.assertEqual(p.exitcode,23)
                folder=next((root/'runs').iterdir())
                with self.assertRaises(FileNotFoundError):load_completed_run(folder)
                old={p.name:digest(p) for p in folder.glob('*.json')}
                detector=SyntheticDetector(self.queries,self.spec)
                result=run_cached_detection(self.manifest,root,detector,spec=self.spec,
                                            model_files=self.weights,processor_files=self.processor)
                self.assertEqual(detector.calls,0)
                self.assertEqual(result['artifacts']['evaluation.json']['detection']['true_positives'],8)
                self.assertEqual(old,{name:digest(folder/name) for name in old})

    def test_interrupted_initial_adapter_manifest_can_resume_without_deleting_orphans(self):
        ctx=mp.get_context('spawn');q=ctx.Queue();log=self.root/'calls'
        p=ctx.Process(target=run_worker,args=(self.bundle,self.output,log,q,None,'adapter.json'))
        p.start();self.processes([p]);self.assertEqual(p.exitcode,23)
        self.assertFalse((self.output/'adapter.json').exists())
        orphans=list(self.output.glob('.adapter.json.*.pending'));self.assertTrue(orphans)
        result=self.run_pipeline()
        self.assertEqual(result['artifacts']['evaluation.json']['detection']['true_positives'],8)
        self.assertTrue(all(p.exists() for p in orphans))

    def test_lock_timeout_is_explicit(self):
        path=self.root/'lock'
        with artifact_lock(path):
            with self.assertRaises(TimeoutError):
                with artifact_lock(path,timeout=0.):pass


if __name__=='__main__':unittest.main()
