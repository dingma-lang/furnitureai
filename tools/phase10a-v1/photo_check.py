"""CPU integrity/coordinate checks on historical TRAIN photos only; no model inference."""
from PIL import Image
from common import ROOT, OUT, READINESS, read, sha, save
from audit import SOURCES

def run():
    caches = {key:{r['image_id']:r for r in read(ROOT/path)[field]}
              for key,(_,path,field) in SOURCES.items()}
    checks = []
    for r in read(READINESS)['source_rows']:
        if r['split'] != 'train': continue
        source = caches[r['source']][r['image_id']]
        p = (ROOT/SOURCES[r['source']][0]/source['file']).resolve()
        if not p.is_relative_to(ROOT/'data') or sha(p)!=r['sha256']: raise ValueError('Source changed')
        with Image.open(p) as im: im.verify()
        with Image.open(p) as im:
            orientation = im.getexif().get(274,1)
            im.load()
            if orientation != 1 or im.size != (source['width'],source['height']):
                raise ValueError('Image orientation/dimensions do not match native frame')
            checks.append(dict(image_id=r['image_id'],size=list(im.size),exif_orientation=orientation,
                               decoding='PASS',coordinate_frame='preserved original pixels',sha256=r['sha256']))
    save(OUT/'train-photo-decode-verification.json',dict(records=checks,status='PASS',training_photos_decoded=len(checks),
         development_or_frozen_photos_decoded=0,source_images_changed=False,model_inference=0))
    print('TRAIN PHOTO INTEGRITY',len(checks),'decoded, original coordinate frames verified; no development/frozen pixels',flush=True)

if __name__=='__main__':run()
