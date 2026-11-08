"""Offline final integrity, identity and historical-preservation checks."""
import hashlib
import json
from pathlib import Path
import sys
from network import *
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical


def run():
    out=DEST/'verification.json'
    if out.exists():raise FileExistsError('Do not replace a completed verification')
    previous=historical(ROOT/'data/open-images/metadata/phase5-v1')
    meta=ROOT/'data/open-images/metadata/phase5-v1';checked_metadata=0
    for line in (meta/'ARTIFACT-SHA256SUMS').read_text().splitlines():
        expected,rel=line.split('  ',1)
        if sha(meta/rel)!=expected:raise ValueError('Historical metadata changed: '+rel)
        checked_metadata+=1
    for rel,h in json.loads((meta/'implementation-sha256.json').read_text()).items():
        if sha(ROOT/rel)!=h:raise ValueError('Historical metadata tool changed')
    m=load_manifest();approved={r['image_id']:r for r in m['images']};state=json.loads((DEST/'network-ledger.json').read_text());screen=json.loads((DEST/'screening-final-v1/screening.json').read_text())
    if state['actual_body_bytes']!=sum(r['actual_body_bytes'] for r in state['requests']):raise ValueError('Body accounting mismatch')
    if state['charged_bytes']>CAP or allocated(DEST)>DISK:raise ValueError('Resource limit exceeded')
    for rec in state['requests']:
        if not rec['complete'] or sha(DEST/rec['file'])!=rec['sha256']:raise ValueError('Saved response integrity failure')
    for r in screen['images']:
        if r['proposed_split']!=approved[r['image_id']]['proposed_split'] or r['original_annotations']!=approved[r['image_id']]['annotations']:raise ValueError('Source box or membership changed')
    paths=[ROOT/'data/abo/training-prep/phase1d-v1/proposal-v2/eligibility-inventory.json',ROOT/'data/abo/image-pilots/phase3-expansion-v1/audit-v3/screening.json']
    reference=set();reference_images=0
    for path in paths:
        for row in json.loads(path.read_text())['images']:
            reference_images+=1
            for key in ('sha256','decoded_pixel_sha256'):
                if row.get(key):reference.add(row[key])
    old_rooms=json.loads((ROOT/'reports/phase4-room-comparison-v1/results.json').read_text())['images']
    for r in old_rooms:reference.add(r['sha256']);reference_images+=1
    overlap=[]
    for row in screen['images']:
        if row['downloaded'] and any(row[k] in reference for k in ('sha256','decoded_pixel_sha256','raw_rgb_sha256')):overlap.append(row['image_id'])
    if overlap:raise ValueError('Stored historical exact-image/pixel hash overlap: '+str(overlap))
    results=dict(manifest_sha256=PIN,approved_membership_unchanged=150,original_mapped_boxes_unchanged=sum(len(r['original_annotations']) for r in screen['images']),
        historical_preservation=previous,previous_metadata_artifacts_checked=checked_metadata,previous_metadata_changes=0,
        historical_image_metadata_records_compared=reference_images,historical_image_or_pixel_hash_overlaps=overlap,
        cross_partition_exact_or_near_duplicate_pairs=screen['duplicate_pairs'],
        source_images_and_boxes_modified=False,model_inference=False,training=False,official_phase2b_scored=False,
        limitations='Known source families and stored exact hashes checked. Different-resolution near duplicates against historic ABO photos and unknown physical identities are not exhaustively ruled out; no official heldout photos/features opened.')
    atomic(out,results);print(json.dumps(results,indent=2))

if __name__=='__main__':run()
