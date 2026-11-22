"""CPU-only C16 preparation;no inference,training or optimizer launcher."""
import sys,os,json,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c15r-v1'))
from common15r import *
HERE16=Path(__file__).resolve().parent
DATA16=ROOT/'data/open-images/preparation/phase10c16-v1'
REPORT16=ROOT/'reports/phase10c16-v1'
PARENT16='66568c57826619cb36dac1205684968d7350f31e'
BRANCH16='experiment/phase10c16-source-aware-balancing'


def emit16(path,value):
    path=Path(path)
    if not any(path.resolve().is_relative_to(base) for base in (DATA16,REPORT16)):raise PermissionError('C16 output outside isolated roots')
    if path.exists():raise FileExistsError('Immutable C16 record exists')
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
    with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(pending,path)


def verify16_inputs():
    count=verify_seal(REPORTR/'SHA256SUMS');meta=read(REPORTR/'artifact-seal.json');private=ROOT/meta['private_manifest']
    if sha(private)!=meta['private_manifest_sha256']:raise ValueError('C15R private manifest changed')
    for rel,h in read(private)['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C15R artifact changed: '+rel)
    for rel,h in meta['code_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C15R source changed')
    from analyze15r import validated_records
    records,index,cfg,done=validated_records();result=read(REPORTR/'results.json');expected=dict(focal_output=.01750855044370582,BCE_output=1.1325037300382963,focal_projection=.1311844795941962,BCE_projection=9.24055105074548)
    values=dict(focal_output=result['components']['source_focal']['sum_per_photo_output_logit_L1'],BCE_output=result['components']['source_BCE']['sum_per_photo_output_logit_L1'],focal_projection=result['components']['source_focal']['mean_per_photo_projection_L2'],BCE_projection=result['components']['source_BCE']['mean_per_photo_projection_L2'])
    if any(abs(values[k]-v)>1e-10 for k,v in expected.items()) or result['source_group_variability']['BCE_exceeds_positive_projection_groups']!=12 or result['source_group_variability']['classification_direction_reversed_groups']!=10 or result['source_negative_confidence']['below005']!=272 or result['uncovered_regions']!=17:raise ValueError('Historical C15R findings differ')
    return records,index,dict(C15R_public_files=count,C15R_private_files=meta['private_files'],confirmed=values,metadata_and_masks_verified=True,real_model_forwards=0)
