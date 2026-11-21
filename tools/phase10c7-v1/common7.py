"""Pinned, zero-update query smoke; no training authorization is inferred."""
from pathlib import Path
import sys,os,uuid
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
from common6 import read,save,sha,digest,manifest,optimizer_member,CLASSES,ADMISSION,candidates,registry,overlap
SOURCE=ROOT/'data/open-images/preparation/phase10c6-v1'
RUN=ROOT/'data/open-images/experiments/phase10c7-query-smoke-v1'
REPORT=ROOT/'reports/phase10c7-v1'
BASE=ROOT/'models/owlv2-original-baseline-20261103'
MAX_FORWARDS=24
MAX_SECONDS=600
MAX_VRAM=int(6.5*2**30)
PROMPTS=['a photo of a '+c for c in CLASSES]

def inputs():
    return read(RUN/'binding.json')

def selected_rows(binding=None):
    b=inputs() if binding is None else binding;rows={r['image_id']:r for r in manifest()['rows']};out=[]
    for p in b['photos']:
        r=rows.get(p['image_id'])
        if r is None or not optimizer_member(r) or any(r.get(k)!=v for k,v in p.items()):raise PermissionError('Exact optimization photo binding changed')
        out.append(r)
    if len(out)!=24 or len({r['image_id'] for r in out})!=24 or len({r['identity_group'] for r in out})!=24:raise ValueError('Exact24 source photos/groups required')
    return out

def validate_binding(binding):
    if binding.get('max_forward_passes')!=24 or binding.get('max_seconds')!=600 or binding.get('max_vram_bytes')!=MAX_VRAM or binding.get('training_authorized') is not False or binding.get('GPU_smoke_authorized') is not True:raise PermissionError('Smoke-only authorization/budget mismatch')
    for path,h in binding['files'].items():
        if sha(ROOT/path)!=h:raise ValueError('Pinned input/code changed: '+path)
    rows=selected_rows(binding)
    for r in rows:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Source photo changed')
    return rows

def original_boxes(square_cxcywh,width,height):
    """NumPy inverse of right/bottom square-padding + resize; never clipped."""
    import numpy as np
    q=np.asarray(square_cxcywh,dtype=np.float32);xy=np.concatenate((q[:,:2]-q[:,2:]/2,q[:,:2]+q[:,2:]/2),axis=1)
    pixels=xy*max(width,height);normalized=pixels/np.array([width,height,width,height],dtype=np.float32)
    return pixels.astype(np.float32),normalized.astype(np.float32)

def arrays_valid(arrays):
    import numpy as np
    if set(arrays)!=set(('logits','square_cxcywh','original_xyxy_pixels','original_xyxy_normalized')):raise ValueError('Unexpected raw-query array schema')
    n=len(arrays['logits'])
    if n!=3600 or arrays['logits'].shape!=(n,7) or any(arrays[k].shape!=(n,4) for k in arrays if k!='logits'):raise ValueError('Expected3600 original960px queries and7 prompts')
    if any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite or non-float32 raw query')
    q=arrays['square_cxcywh']
    if (q<0).any() or (q>1).any() or (q[:,2:]<=0).any():raise ValueError('Invalid model box outputs')

def write_arrays(path,arrays):
    import numpy as np
    arrays_valid(arrays);tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        if tmp.stat().st_size>600000:raise ValueError('Raw per-image storage bound')
        os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def load_arrays(path):
    import numpy as np,zipfile
    if path.stat().st_size>600000:raise ValueError('Oversized query file')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())!=4 or sum(i.file_size for i in z.infolist())>600000:raise ValueError('Unsafe query archive')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays_valid(arrays);return arrays
