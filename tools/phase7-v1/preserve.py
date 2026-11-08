"""Read-only seals: historical hashes, model bytes, frozen membership; never decode holdouts."""
import argparse
import importlib.util
import sys
from contracts import ROOT, read, sha, write_new

BASES = [
    'data/open-images/metadata/phase5-v1',
    'data/open-images/image-pilots/phase5-v1',
    'data/open-images/experiments/phase5b-v1',
    'data/open-images/experiments/phase5c-v1',
    'data/open-images/preparation/phase6a-v1',
    'data/open-images/image-pilots/phase6a-v1',
    'data/coco/metadata/phase6b-v1',
    'data/coco/rights-audits/phase6c-v1',
    'data/coco/image-pilots/phase6d-v1',
]

def verify():
    seals={};implementations=0
    for relative in BASES:
        base=ROOT/relative;n=0
        for line in (base/'ARTIFACT-SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1);path=(base/name).resolve()
            if not path.is_relative_to(base.resolve()) or sha(path) != expected:
                raise ValueError('Historical seal failed '+str(path))
            n+=1
        seals[relative]=n
        manifest=base/'implementation-sha256.json'
        if manifest.exists():
            for filename,expected in read(manifest).items():
                if sha(ROOT/filename) != expected:raise ValueError('Historical implementation changed '+filename)
                implementations+=1
    expected_models={
        'models/owlv2-original-baseline-20261103/model.safetensors':'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7',
        'models/siglip2-base-patch16-224-phase2a-v1/model.safetensors':'612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b',
    }
    registry=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json')
    expected_models[registry['checkpoint']]=registry['sha256']
    for filename,expected in expected_models.items():
        if sha(ROOT/filename) != expected:raise ValueError('Model preservation failed '+filename)
    # Existing verifier uses stat-only checks for official ABO held-out images/features.
    sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
    spec=importlib.util.spec_from_file_location('phase7_historical_verifier',ROOT/'tools/phase5-metadata-v1/verify_metadata.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    historical=module.historical(ROOT/'data/open-images/metadata/phase5-v1')
    lock=read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')
    slots=read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']
    return dict(status='PASS',historical_artifact_ledgers=seals,historical_implementation_files=implementations,
                model_hashes=expected_models,previous_historical_checks=historical,
                original_openimages_frozen_slots=sum(r['proposed_split']=='frozen_evaluation' for r in slots),
                eligible_reserved_frozen_references=len(lock['frozen_references']),
                frozen_pixels_decoded=0,frozen_scoring=0,model_training_or_inference=0,mac_project_accessed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();write_new(args.output,verify());print('Historical preservation PASS')
