"""CPU-only integrity, orientation and duplicate heuristics; never object labels."""
import io
from PIL import Image,ImageOps,ImageStat
from common16 import *


def perceptual(image):
    gray=image.convert('L');d=list(gray.resize((9,8),Image.Resampling.LANCZOS).get_flattened_data());dh=0
    for y in range(8):
        for x in range(8):dh=(dh<<1)|int(d[y*9+x]>d[y*9+x+1])
    a=list(gray.resize((8,8),Image.Resampling.LANCZOS).get_flattened_data());average=sum(a)/64;ah=0
    for v in a:ah=(ah<<1)|int(v>average)
    return dict(dhash=f'{dh:016x}',ahash=f'{ah:016x}')


def inspect_image(body,row):
    if not 0<len(body)<=IMAGE_CAP or len(body)!=row['expected_bytes']:raise ValueError('Image size mismatch/cap')
    if hashlib.sha1(body).hexdigest()!=row['source_sha1']:raise ValueError('Published source SHA1 mismatch')
    with Image.open(io.BytesIO(body)) as im:
        if im.format!='JPEG' or im.size!=(row['width'],row['height']) or im.width*im.height>30000000:
            raise ValueError('Unexpected image format/dimensions/pixel limit')
        im.verify()
    with Image.open(io.BytesIO(body)) as im:
        im.load();orientation=im.getexif().get(274,1);image=ImageOps.exif_transpose(im).convert('RGB')
        gray=image.convert('L');std=ImageStat.Stat(gray).stddev[0]
        return dict(encoded_dimensions=[row['width'],row['height']],display_dimensions=list(image.size),
                    EXIF_orientation=orientation,format='JPEG',mode='RGB',grayscale_stddev=std,
                    flags=['low_contrast_or_blank_heuristic'] if std<8 else [],**perceptual(image))


def previews():
    closed=read(OUT/'acquisition-closed.json')
    if closed['stop'] or closed['downloaded_count']!=3:raise ValueError('Exact completed pilot required')
    for row in closed['records']:
        p=ROOT/row['file']
        if sha(p)!=row['sha256']:raise ValueError('Acquired source changed')
        target=OUT/'previews'/(row['source_id'].split(':')[1]+'.jpg')
        if target.exists():raise FileExistsError('Preview exists; no overwrite')
        with Image.open(p) as original:
            image=ImageOps.exif_transpose(original).convert('RGB');image.thumbnail((1600,1600),Image.Resampling.LANCZOS)
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=88);save(target,buffer.getvalue())


if __name__=='__main__':previews()
