"""Separately gated metadata proposals, backed by body-free official HEADs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *

def run():
 root=DEST/'expansion-proposal-v1';heads=read(root/'official-source-heads.json')
 byurl={r['url']:r for r in heads['records']}
 coco='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
 expected_sha='113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
 record=byurl[coco];assert record['status']==200 and record['tls_verified'] and record['size']==252907541
 common=dict(approval_granted=False,photographs_authorized=0,training_authorized=False,model_weights_authorized=False,paid_services=False,created_utc=now(),max_attempts_per_object=2,
  stop_conditions=['Unapproved source or redirect','Changed Content-Length or ETag','TLS failure','Integrity mismatch','Unsafe archive member or expansion limit','Resource limit','Unexpected annotation schema'],
  licensing='Annotations only: CC BY4.0. This does not license individual images; all later photographs require matching creator-page rights, attribution and commercial-use allowlist before a separate exact image approval.')
 coco_proposal=dict(**common,schema='phase6b-coco-metadata-only-proposal-v1',objects=[dict(url=coco,official_published_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',bytes=252907541,etag=record['etag'],sha256=expected_sha,
  checksum_source='https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/coco.txt',license_url='https://creativecommons.org/licenses/by/4.0/',annotation_creator='COCO Consortium')],
  expected_transfer_bytes=252907541,maximum_two_attempt_bytes=505815082,max_cumulative_transfer=520000000,max_new_allocated_storage=1073741824,
  destination=str(ROOT/'data/coco/metadata/phase6b-v1'),extract_only=['annotations/instances_train2017.json','annotations/instances_val2017.json'],
  maximum_combined_extracted_bytes=650000000,max_derived_artifact_bytes=100000000,archive_policy='Read and validate central directory first; exact whitelisted regular JSON members only; reject duplicate names, symlinks, absolute/parent paths, encrypted entries, excessive expansion; stream extract with byte cap and CRC validation. Do not extract captions, keypoints or images.',
  secure_source_basis='Observed official images.cocodataset.org DNS aliases images.cocodataset.org.s3.amazonaws.com. AWS documents same-bucket path-style requests and dotted-bucket wildcard limitations. HEAD on exact path-style object returned 200 under default TLS verification; no certificate bypass. Expected SHA pinned independently from Google TFDS.',
  official_source_reference='https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm',aws_reference='https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html',
  priorities=['sofa/couch','bed','chair','dining table subtype'],unsupported_categories=['desk','cabinet','lamp'],
  outcome='Parse official metadata only; report available native counts and source identities; prepare a later rights-verified exact photograph proposal. No image count or licence acceptance count is fabricated before parsing.')
 write_new(root/'coco-metadata-acquisition-proposal.json',coco_proposal)
 compression=read(root/'openimages-compression-heads.json');oi=[]
 for r in compression:
  assert r['status']==200 and r['encoding'] is None
  oi.append(dict(url=r['url'],bytes=int(r['bytes']),etag=r['etag'],annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_creator='Google LLC'))
 total=sum(r['bytes'] for r in oi);assert total==3273620121
 oi_proposal=dict(**common,schema='phase6b-openimages-training-metadata-proposal-v1',objects=oi,expected_transfer_bytes=total,maximum_two_attempt_bytes=2*total,
  max_cumulative_transfer=6600000000,max_new_allocated_storage=6442450944,destination=str(ROOT/'data/open-images/metadata/phase6b-train-v1'),
  exceeds_current_phase6a_approval=True,budget_increase_requires_explicit_approval=True,official_source_reference='https://storage.googleapis.com/openimages/web/download_v7.html',
  priorities=['native Desk and Lamp boxes unavailable in COCO','Couch/Bed coverage','genuine indoor context','source-group novelty'],
  streaming_policy='Stream CSV records and keep selected furniture/native-label evidence plus on-disk identity index; never load all training annotations as Python objects. Reuse pinned local class dictionary/hierarchy. Retain original source rows and hashes; no class substitution or photograph download.',
  caveat='Desk native semantic scope is not fixed by more examples; select comparable units and quarantine uncertain examples. Avoid Cabinetry optimization until unit policy is resolved. Unannotated classes remain unknown, not negatives.')
 write_new(root/'openimages-training-metadata-proposal.json',oi_proposal)
 write_new(root/'future-image-pilot-targets.json',dict(approved=False,stage='Only after separately approved metadata and per-photo rights audits',target_max_new_photos=400,targets_not_promised_counts=dict(sofa=100,desk=60,lamp=80,bed=60,chair=50,table=50,cabinet=0),
  split_policy='New 80/20 group-stratified train/development proposal; preserve every prior frozen group; no image replacement after approval',
  expected_photo_budget_scenario='400 * 250000 bytes =100000000 photo bytes; scenario only, not HEAD-verified. Exact accepted counts/bytes must be pinned before photo approval.',
  suggested_later_photo_caps=dict(cumulative_transfer=1073741824,allocated_storage=2147483648,per_image_bytes=3145728,max_attempts=2),
  confidence_policy='Source-supported provisional categories and source-native units; individual copyright licensing verified separately; no claim of human annotation confirmation.',
  cabinet_policy='No allocation until cabinet-run versus individual-unit semantics is defined. No substitution of other COCO categories for absent Desk or Lamp.',
  no_training_until_separate_approval=True))
 print('PROPOSALS: COCO252907541 bytes; OpenImages3273620121 bytes. Both await separate approval; no archive GET.')

if __name__=='__main__':run()
