"""CPU image integrity and duplicate heuristics, never labels or object inference."""
import io,warnings
from PIL import Image,ImageOps,ImageStat
from common11 import *
Image.MAX_IMAGE_PIXELS=20_000_000
warnings.simplefilter('error',Image.DecompressionBombWarning)
def hashes(image):
    gray=image.convert('L');small=gray.resize((9,8),Image.Resampling.LANCZOS);pixels=list(small.get_flattened_data());d=0
    for y in range(8):
        for x in range(8):d=(d<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    a=list(gray.resize((8,8),Image.Resampling.LANCZOS).get_flattened_data());mean=sum(a)/64;h=0
    for v in a:h=(h<<1)|int(v>mean)
    return dict(dhash=f'{d:016x}',ahash=f'{h:016x}')
def distance(a,b):return (int(a,16)^int(b,16)).bit_count()
def inspect_bytes(body,row):
    if not 0<len(body)<=IMAGE_CAP:raise ValueError('512000byte JPEG cap')
    with Image.open(io.BytesIO(body)) as im:
        if im.format!='JPEG':raise ValueError('Not JPEG')
        im.verify()
    with Image.open(io.BytesIO(body)) as im:
        im.load();original=im.size;orientation=im.getexif().get(274,1);display=ImageOps.exif_transpose(im);w,h=display.size
        if max(w,h)>1600 or min(w,h)<1:raise ValueError('Unexpected derivative dimensions')
        expected=row['width']/row['height']
        if abs(w/h-expected)>2/min(w,h):raise ValueError('Source/derivative aspect mismatch')
        gray=display.convert('L');stat=ImageStat.Stat(gray);q=dict(encoded_dimensions=original,display_dimensions=[w,h],EXIF_orientation=orientation,
            derivative_from_original=[row['width'],row['height']],mode=display.mode,format='JPEG',grayscale_stddev=stat.stddev[0],
            **hashes(display),flags=[],no_object_labels_generated=True)
        if min(w,h)<320:q['flags'].append('small_dimension_heuristic')
        if stat.stddev[0]<8:q['flags'].append('low_contrast_or_blank_heuristic')
        return q
def historical_duplicates(records):
    from itertools import combinations
    native=read(ROOT/'data/open-images/preparation/phase10e1-v1/training-context.json')
    train=[x['row'] for x in native['entries']];dev=native['development']
    if len(train)!=59 or len(dev)!=19:raise ValueError('Existing59/19 only; no held-out reads')
    reference=[]
    for r in [*train,*dev]:
        # Restrict reference checks to the already exposed internal protocol, not frozen material.
        if r['historical_partition']!='train' or r['internal_partition'] not in ('optimization','inner_development'):raise ValueError('Forbidden reference partition')
        file=ROOT/r['path']
        if sha(file)!=r['sha256']:raise ValueError('Historical source hash changed')
        with Image.open(file) as im:im=ImageOps.exif_transpose(im);p=hashes(im)
        reference.append(dict(image_id=r['image_id'],sha256=r['sha256'],partition=r['internal_partition'],**p))
    exact=[];near=[]
    for a,b in combinations(records,2):
        if a['sha256']==b['sha256']:exact.append([a['image_id'],b['image_id']])
        d=distance(a['quality']['dhash'],b['quality']['dhash']);h=distance(a['quality']['ahash'],b['quality']['ahash'])
        if min(d,h)<=6:near.append(dict(pair=[a['image_id'],b['image_id']],dhash_distance=d,ahash_distance=h,confirmed_duplicate=False))
    cross=[]
    for a in records:
        for b in reference:
            d=distance(a['quality']['dhash'],b['dhash']);h=distance(a['quality']['ahash'],b['ahash'])
            if a['sha256']==b['sha256'] or min(d,h)<=6:cross.append(dict(pilot=a['image_id'],historical=b['image_id'],exact=a['sha256']==b['sha256'],dhash_distance=d,ahash_distance=h))
    return dict(exact_pilot_duplicate_pairs=exact,perceptual_pilot_flags=near,historical_matches_or_flags=cross,reference_count=len(reference),
       frozen_or_reserved_images_read=0,perceptual_hashes_are_heuristics_not_identity_proof=True,scene_or_room_identity_unverified=True)
