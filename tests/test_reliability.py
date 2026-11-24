"""Synthetic reliability tests: no real model outputs, external photos or network."""
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from furniture_bench.cache import file_identity, make_contract, validate_contract
from furniture_bench.catalog_expansion import incremental_embeddings
from furniture_bench.data import digest, load_image, load_image_with_digest, write_json
from furniture_bench.persistence import AtomicCommitError, atomic_json, atomic_write
from furniture_bench.retrieval import CatalogIndex


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="furniture-reliability-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class AtomicPersistenceTests(TemporaryFiles):
    def test_json_bytes_match_existing_serialization(self):
        path = self.root / "value.json"
        value = {"text": "家具", "values": [1, 0.5, None, True]}
        write_json(path, value)
        self.assertEqual(path.read_bytes(), (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode())

    def test_failure_mid_write_preserves_existing_target_and_cleans_own_temp(self):
        path = self.root / "result.json"
        path.write_bytes(b'{"old":true}')
        def fail(stream):
            stream.write(b'{"partial":')
            raise OSError("Synthetic interruption")
        with self.assertRaises(OSError):
            atomic_write(path, fail)
        self.assertEqual(path.read_bytes(), b'{"old":true}')
        self.assertEqual(list(self.root.glob('*.pending')), [])
        atomic_json(path, {"recovered": True})
        self.assertEqual(json.loads(path.read_text()), {"recovered": True})

    def test_failed_new_write_does_not_publish_target(self):
        path = self.root / "result.json"
        with patch('furniture_bench.persistence.os.fsync', side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError):
                atomic_json(path, {"ok": True})
        self.assertFalse(path.exists())

    def test_validator_failure_preserves_previous_result(self):
        path = self.root / 'result';path.write_bytes(b'old')
        def reject(_):raise ValueError('Synthetic validation rejection')
        with self.assertRaises(ValueError):
            atomic_write(path, lambda f:f.write(b'new'), validator=reject)
        self.assertEqual(path.read_bytes(), b'old')

    def test_replace_failure_preserves_previous_result(self):
        path=self.root/'result';path.write_bytes(b'old')
        with patch('furniture_bench.persistence.os.replace', side_effect=OSError('Synthetic rename failure')):
            with self.assertRaises(OSError):atomic_json(path, {'new':True})
        self.assertEqual(path.read_bytes(),b'old')

    def test_no_clobber_uses_atomic_publication(self):
        path=self.root/'result';path.write_bytes(b'old')
        with self.assertRaises(FileExistsError):atomic_json(path, {'new':True}, overwrite=False)
        self.assertEqual(path.read_bytes(),b'old')

    def test_nonfinite_or_unserializable_values_do_not_touch_destination(self):
        path=self.root/'result';path.write_bytes(b'old')
        for bad in (float('nan'),float('inf'),object()):
            with self.assertRaises((ValueError,TypeError)):atomic_json(path, {'bad':bad})
            self.assertEqual(path.read_bytes(),b'old')

    def test_unique_tempnames_and_stale_orphans_not_treated_as_results(self):
        path=self.root/'result';orphan=self.root/'.result.stale.pending';orphan.write_bytes(b'{broken')
        atomic_json(path, {'complete':True})
        self.assertEqual(orphan.read_bytes(),b'{broken')
        self.assertEqual(json.loads(path.read_text()),{'complete':True})

    def test_directory_sync_failure_reports_committed_state(self):
        path=self.root/'result';path.write_bytes(b'old')
        with patch('furniture_bench.persistence._sync_directory',side_effect=OSError('Synthetic directory fsync failure')):
            with self.assertRaises(AtomicCommitError) as caught:atomic_json(path,{'complete':True})
        self.assertTrue(caught.exception.committed)
        self.assertEqual(json.loads(path.read_text()),{'complete':True})

    def test_existing_permissions_retained(self):
        path=self.root/'result';path.write_bytes(b'old');path.chmod(0o640)
        atomic_json(path,{'new':True});self.assertEqual(path.stat().st_mode & 0o777,0o640)

    def test_symlink_destination_rejected_without_changing_target(self):
        target=self.root/'target';target.write_bytes(b'old');link=self.root/'link';link.symlink_to(target)
        with self.assertRaises(ValueError):atomic_json(link,{'new':True})
        self.assertEqual(target.read_bytes(),b'old')


class ContractFixtures(TemporaryFiles):
    def inputs(self):
        return dict(image_id='synthetic-image',image_sha256='a'*64,
                    model=dict(model_id='synthetic-no-model',checkpoint_sha256='b'*64,processor_sha256='c'*64),
                    preprocessing=dict(orientation='EXIF-RGB',crop=None,size=[224,224],resample='model-default'),
                    categories=['chair','bed'],prompts=['a chair','a bed'],inference_settings=dict(dtype='float32',threshold=.15,decoder='synthetic-v1'),
                    supervision=dict(annotations=[],ignore_regions=[],taxonomy_version='synthetic-v1',completeness={'chair':'unknown'},source_group_id='synthetic-group'))


class ContractTests(ContractFixtures):
    def test_model_file_content_not_mtime_controls_identity(self):
        p=self.root/'weights';p.write_bytes(b'abcd');stat=p.stat();old=file_identity({'weights':p})
        p.write_bytes(b'efgh');os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns));new=file_identity({'weights':p})
        self.assertNotEqual(old['sha256'],new['sha256'])

    def test_file_paths_and_mapping_order_are_not_fingerprints(self):
        a=self.root/'a';b=self.root/'b';a.write_bytes(b'equal');b.write_bytes(b'equal')
        self.assertEqual(file_identity({'weights':a}),file_identity({'weights':b}))

    def test_inference_dependencies_each_invalidate(self):
        original=self.inputs();base=make_contract(**original)
        cases=[]
        for key,val in [('image_id','different'),('image_sha256','f'*64),('prompts',['a bed','a chair']),('categories',['bed','chair'])]:
            x=copy.deepcopy(original);x[key]=val;cases.append(x)
        for key in ('checkpoint_sha256','processor_sha256'):
            x=copy.deepcopy(original);x['model'][key]='f'*64;cases.append(x)
        for group,key,val in [('preprocessing','size',[384,384]),('preprocessing','crop',[1,2,9,10]),('inference_settings','threshold',.5),('inference_settings','dtype','float16')]:
            x=copy.deepcopy(original);x[group][key]=val;cases.append(x)
        for x in cases:self.assertNotEqual(base['inference_key'],make_contract(**x)['inference_key'])

    def test_supervision_dependencies_invalidate_context_not_raw_inference(self):
        original=self.inputs();base=make_contract(**original)
        for key,val in [('annotations',[{'synthetic':True}]),('ignore_regions',[[0,0,1,1]]),('taxonomy_version','new'),('completeness',{'chair':'review-required'}),('source_group_id','different-group')]:
            x=copy.deepcopy(original);x['supervision'][key]=val;new=make_contract(**x)
            self.assertEqual(base['inference_key'],new['inference_key']);self.assertNotEqual(base['full_key'],new['full_key'])

    def test_returned_contract_does_not_alias_mutable_inputs(self):
        inputs=self.inputs();value=make_contract(**inputs);inputs['preprocessing']['size'][0]=999
        self.assertEqual(value['inference']['preprocessing']['size'],[224,224])

    def test_unknown_schema_tampering_and_nonfinite_values_rejected(self):
        c=make_contract(**self.inputs());c['schema_version']=3
        with self.assertRaises(ValueError):validate_contract(c)
        c=make_contract(**self.inputs());c['inference']['settings']['threshold']=.9
        with self.assertRaises(ValueError):validate_contract(c)
        inputs=self.inputs();inputs['inference_settings']['threshold']=float('nan')
        with self.assertRaises(ValueError):make_contract(**inputs)

    def test_image_binding_and_required_identity_not_optional(self):
        c=make_contract(**self.inputs())
        with self.assertRaises(ValueError):validate_contract(c,image_id='different')
        inputs=self.inputs();inputs['model'].pop('checkpoint_sha256')
        with self.assertRaises(ValueError):make_contract(**inputs)

    def test_order_of_dictionary_keys_does_not_change_identity(self):
        x=self.inputs();y=copy.deepcopy(x);y['model']=dict(reversed(list(y['model'].items())))
        self.assertEqual(make_contract(**x),make_contract(**y))

    def test_catalogue_strong_provenance_requires_both_sides(self):
        legacy=dict(model_id='synthetic',requested_revision='main',feature='unit',preprocessing='test')
        index=CatalogIndex(None,{'embedding':legacy});index.assert_compatible(legacy)
        strong=dict(legacy,model_fingerprint='b'*64,preprocessing_fingerprint='c'*64)
        with self.assertRaises(ValueError):index.assert_compatible(strong)
        index=CatalogIndex(None,{'embedding':strong});index.assert_compatible(strong)
        with self.assertRaises(ValueError):index.assert_compatible(dict(strong,model_fingerprint='d'*64))


class FeatureCacheTests(ContractFixtures):
    def inputs(self):
        value = super().inputs()
        value.update(task="image_embedding", categories=[], prompts=[])
        return value

    def setup_cache(self):
        path=self.root/'input.bin';path.write_bytes(b'synthetic-encoder-input-no-photo')
        row=dict(image_id='synthetic-image',file=str(path),sha256=digest(path))
        inputs=self.inputs();inputs['image_sha256']=row['sha256'];contract=make_contract(**inputs)
        return row,inputs['model'],contract

    def encoder(self,paths):
        return np.array([[1.,0.,0.] for _ in paths],dtype=np.float32),0.

    def test_v2_reuse_and_label_only_changes_without_reencoding(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'
        a,r=incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c})
        changed=copy.deepcopy(c);args=self.inputs();args['image_sha256']=row['sha256'];args['supervision']['source_group_id']='different';new=make_contract(**args)
        def forbidden(_):raise AssertionError('Label-only changes must not rerun encoder')
        b,s=incremental_embeddings([row],folder,model,forbidden,cache_contracts={row['image_id']:new})
        np.testing.assert_array_equal(a,b);self.assertEqual(s['reused'],1);self.assertNotEqual(r['context_keys'],s['context_keys'])

    def test_legacy_filename_and_npy_bytes_preserved(self):
        row,model,_=self.setup_cache();folder=self.root/'legacy'
        vector,report=incremental_embeddings([row],folder,model,self.encoder)
        key=hashlib.sha256(json.dumps([row['image_id'],row['sha256'],model],sort_keys=True).encode()).hexdigest()
        raw=io.BytesIO();np.save(raw,vector[0])
        self.assertEqual((folder/(key+'.npy')).read_bytes(),raw.getvalue());self.assertNotIn('cache_schema_version',report)

    def test_v2_does_not_rewrite_legacy_or_migrate_it(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'
        incremental_embeddings([row],folder,model,self.encoder);legacy=next(folder.glob('*.npy'));before=legacy.read_bytes()
        _,r=incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c})
        self.assertEqual(r['new_encodings'],1);self.assertEqual(legacy.read_bytes(),before)

    def test_corrupt_cache_is_rejected_without_encoder_or_replacement(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'
        incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c})
        path=next(folder.rglob('*.npz'));path.write_bytes(b'broken')
        with self.assertRaises((ValueError,OSError)):
            incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c})
        self.assertEqual(path.read_bytes(),b'broken')

    def test_incomplete_npz_not_accepted_as_valid_features(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'/'features-v2';folder.mkdir(parents=True)
        with (folder/(c['inference_key']+'.npz')).open('wb') as stream:np.savez(stream,vector=np.array([1.,0.]))
        with self.assertRaises(ValueError):incremental_embeddings([row],folder.parent,model,self.encoder,cache_contracts={row['image_id']:c})

    def test_mid_save_interruption_then_clean_retry(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'
        def broken(stream,**_):stream.write(b'partial');raise OSError('Synthetic zip interruption')
        with patch('furniture_bench.catalog_expansion.np.savez',side_effect=broken),self.assertRaises(OSError):
            incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c})
        self.assertEqual(list(folder.rglob('*.npz')),[])
        _,report=incremental_embeddings([row],folder,model,self.encoder,cache_contracts={row['image_id']:c});self.assertEqual(report['new_encodings'],1)

    def test_changed_source_during_callback_not_published(self):
        row,model,c=self.setup_cache();folder=self.root/'cache'
        def mutate(paths):Path(paths[0]).write_bytes(b'changed');return self.encoder(paths)
        with self.assertRaises(ValueError):incremental_embeddings([row],folder,model,mutate,cache_contracts={row['image_id']:c})
        self.assertEqual(list(folder.rglob('*.npz')),[])


class ImageSnapshotTests(TemporaryFiles):
    def image(self):
        path=self.root/'image.jpg';exif=Image.Exif();exif[274]=6
        Image.new('RGB',(40,30),'blue').save(path,exif=exif);return path

    def test_exif_rgb_and_digest_are_same_snapshot(self):
        path=self.image();image,h=load_image_with_digest(path,digest(path));original=load_image(path)
        self.assertEqual(image.size,(30,40));self.assertEqual(image.tobytes(),original.tobytes());self.assertEqual(h,digest(path))

    def test_replaced_file_after_manifest_validation_rejected(self):
        path=self.image();expected=digest(path);Image.new('RGB',(40,30),'red').save(path)
        with self.assertRaises(ValueError):load_image_with_digest(path,expected)

    def test_file_replacement_after_read_does_not_mislabel_decoded_pixels(self):
        path=self.image();payload=path.read_bytes();actual_read=Path.read_bytes
        def replace(p):
            data=actual_read(p);p.write_bytes(b'changed-after-snapshot');return data
        with patch.object(Path,'read_bytes',replace):image,h=load_image_with_digest(path)
        self.assertEqual(h,hashlib.sha256(payload).hexdigest());self.assertEqual(image.size,(30,40));self.assertNotEqual(h,digest(path))


class PredictionCacheTests(ContractFixtures):
    def setup_prediction(self):
        from furniture_bench.cache import write_prediction_cache
        inputs=self.inputs();inputs['preprocessing']['box_frame']='EXIF-original-pixel-xyxy';c=make_contract(**inputs)
        p=dict(image_id='synthetic-image',box_frame='EXIF-original-pixel-xyxy',detections=[dict(label='chair',score=.5,box=[1,2,10,20],query_id=7)])
        path=self.root/'prediction.json';write_prediction_cache(path,c,p)
        return path,inputs,c,p

    def test_roundtrip_and_context_only_reevaluation(self):
        from furniture_bench.cache import read_prediction_cache
        path,inputs,c,p=self.setup_prediction();inputs['supervision']['ignore_regions']=[[1,2,3,4]]
        data,meta=read_prediction_cache(path,make_contract(**inputs));self.assertEqual(data,p);self.assertTrue(meta['evaluation_recomputation_required'])
        self.assertTrue(meta['inference_reused'])

    def test_changed_prompt_or_model_rejects_saved_predictions(self):
        from furniture_bench.cache import read_prediction_cache
        path,inputs,c,p=self.setup_prediction();inputs['prompts'].reverse()
        with self.assertRaises(ValueError):read_prediction_cache(path,make_contract(**inputs))

    def test_corrupted_payload_or_partial_json_rejected(self):
        from furniture_bench.cache import read_prediction_cache
        path,inputs,c,p=self.setup_prediction();v=json.loads(path.read_text());v['payload']['detections'][0]['score']=.9;path.write_text(json.dumps(v))
        with self.assertRaises(ValueError):read_prediction_cache(path,c)
        path.write_text('{"schema_version":2,')
        with self.assertRaises(ValueError):read_prediction_cache(path,c)

    def test_existing_predictions_never_overwritten(self):
        from furniture_bench.cache import write_prediction_cache
        path,inputs,c,p=self.setup_prediction();old=path.read_bytes()
        with self.assertRaises(FileExistsError):write_prediction_cache(path,c,p)
        self.assertEqual(path.read_bytes(),old)

    def test_coordinate_frame_is_part_of_identity(self):
        from furniture_bench.cache import write_prediction_cache
        path,inputs,c,p=self.setup_prediction();p['box_frame']='model-square-cxcywh'
        with self.assertRaises(ValueError):write_prediction_cache(self.root/'other.json',c,p)

    def test_prompt_to_category_map_is_fingerprinted(self):
        x=self.inputs();x['prompts']=['chair view 1','chair view 2','bed'];x['query_categories']=['chair','chair','bed'];a=make_contract(**x)
        x['query_categories']=['chair','bed','bed'];self.assertNotEqual(a['inference_key'],make_contract(**x)['inference_key'])
        del x['query_categories']
        with self.assertRaises(ValueError):make_contract(**x)

class AdditionalSafetyTests(ContractFixtures):
    def test_abrupt_process_exit_leaves_old_target_and_retry_ignores_orphan(self):
        import subprocess,sys
        path=self.root/'result.json';path.write_text('{"old":true}')
        script="import os,sys;from furniture_bench.persistence import atomic_write\ndef stop(f):\n f.write(b'{partial');f.flush();os._exit(17)\natomic_write(sys.argv[1],stop)"
        result=subprocess.run([sys.executable,'-B','-c',script,str(path)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,17,result.stderr)
        self.assertEqual(json.loads(path.read_text()),{'old':True});self.assertTrue(list(self.root.glob('*.pending')))
        atomic_json(path,{'recovered':True});self.assertEqual(json.loads(path.read_text()),{'recovered':True})

    def test_metric_rules_and_context_version_bind_evaluation(self):
        from furniture_bench.cache import evaluation_fingerprint
        c=make_contract(**self.inputs())
        self.assertNotEqual(evaluation_fingerprint(c,{'IoU':.5,'version':'synthetic'}),evaluation_fingerprint(c,{'IoU':.75,'version':'synthetic'}))

    def test_empty_model_or_malformed_context_rejected(self):
        inputs=self.inputs();inputs['supervision']['annotations']=None
        with self.assertRaises(ValueError):make_contract(**inputs)

    def test_degenerate_geometry_rejected_before_suppression_and_ignore_math(self):
        from furniture_bench.geometry import clip_box,iou,validate_box
        from furniture_bench.postprocessing import filter_detections,FilterConfig
        box=[0,0,0,10]
        self.assertEqual(iou(box,box),0.);self.assertIsNone(clip_box(box,100,100))
        with self.assertRaises(ValueError):validate_box(box)
        with self.assertRaises(ValueError):filter_detections([dict(label='chair',score=.5,box=box)],FilterConfig())

    def test_pipeline_rechecks_validated_image_before_any_model_call(self):
        from furniture_bench.pipeline import run_experiment
        from furniture_bench.data import load_manifest
        path=self.root/'photo.png';Image.new('RGB',(10,10),'blue').save(path)
        manifest=self.root/'manifest.json'
        write_json(manifest,dict(schema_version=1,images=[dict(image_id='synthetic',file='photo.png',image_sha256=digest(path),objects=None,rights=dict(source='synthetic',license='CC0',allowed_for_benchmark=True))]))
        rows=load_manifest(manifest,'rooms');Image.new('RGB',(10,10),'red').save(path)
        class ForbiddenDetector:
            def detect(self,*_):raise AssertionError('Must reject before model call')
            def metadata(self):return {'model':'synthetic-no-model'}
        with patch('furniture_bench.pipeline.load_manifest',return_value=rows):
            run=run_experiment(manifest,self.root/'run',ForbiddenDetector())
        self.assertFalse(run['images'][0]['detection_completed']);self.assertEqual(run['images'][0]['errors'][0]['stage'],'image_load')

class CatalogInputBindingTests(TemporaryFiles):
    def test_catalog_encoder_cannot_publish_hash_from_changed_input(self):
        from furniture_bench.retrieval import build_catalog
        image=self.root/'catalog.png';Image.new('RGB',(10,10),'blue').save(image)
        manifest=self.root/'catalog.json'
        write_json(manifest,dict(schema_version=1,products=[dict(product_id='synthetic',label='chair',file='catalog.png',rights=dict(source='synthetic',license='CC0',allowed_for_benchmark=True))]))
        class SyntheticEncoder:
            batch_size=1
            def encode(self,images):
                image.write_bytes(b'changed-during-callback')
                return np.array([[1.,0.]],dtype=np.float32),0.
        with patch('furniture_bench.retrieval.CatalogIndex.build',side_effect=AssertionError('Must fail before index creation')) as build:
            with self.assertRaisesRegex(ValueError,'changed during embedding'):build_catalog(manifest,self.root/'index',SyntheticEncoder())
            build.assert_not_called()

    def test_schema_is_an_integer_and_context_files_can_be_content_bound(self):
        from furniture_bench.cache import file_identity
        fixture=ContractFixtures();inputs=fixture.inputs();c=make_contract(**inputs);c['schema_version']=2.0
        with self.assertRaises(ValueError):validate_contract(c)
        annotation=self.root/'synthetic-mask-metadata';annotation.write_bytes(b'old-mask-fixture')
        inputs['supervision']['annotation_files']=file_identity({'mask':annotation});before=make_contract(**inputs)
        annotation.write_bytes(b'changed-mask-fixture');inputs['supervision']['annotation_files']=file_identity({'mask':annotation});after=make_contract(**inputs)
        self.assertEqual(before['inference_key'],after['inference_key']);self.assertNotEqual(before['context_key'],after['context_key'])

class MalformedArtifactTests(ContractFixtures):
    def test_legacy_npz_disguised_as_npy_is_rejected(self):
        from furniture_bench.catalog_expansion import _read_feature
        path=self.root/'legacy.npy'
        with path.open('wb') as stream:np.savez(stream,vector=np.array([1.,0.]))
        with self.assertRaisesRegex(ValueError,'single NumPy array'):_read_feature(path)

    def test_nonobject_prediction_or_inference_metadata_rejected(self):
        from furniture_bench.cache import read_prediction_cache
        contract=make_contract(**self.inputs());path=self.root/'prediction.json';path.write_text('[]')
        with self.assertRaises(ValueError):read_prediction_cache(path,contract)
        contract['inference']=[]
        with self.assertRaises(ValueError):validate_contract(contract)

if __name__=='__main__':unittest.main()
