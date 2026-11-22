"""Immutable exact source/processor/prompt/query contracts for cached teacher outputs."""
from dataclasses import dataclass
import numpy as np
from commonE import *

@dataclass(frozen=True)
class Teacher:
    logits:np.ndarray
    boxes:np.ndarray
    record_sha256:str
    def tensors(self,device='cpu'):
        import torch
        return (torch.tensor(self.logits.copy(),dtype=torch.float32,device=device),torch.tensor(self.boxes.copy(),dtype=torch.float32,device=device))

def verify_alignment(record,row,contract):
    required=dict(image_id=row['image_id'],source_sha256=row['sha256'],source_group=row['identity_group'],width=row['width'],height=row['height'],base_sha256=BASE_SHA,classes=list(CLASSES),prompts=PROMPTS,precision='FP16forward/FP32arrays',query_layout=[60,60],augmentation='none',processor_contract_sha256=contract)
    for key,value in required.items():
        if record.get(key)!=value:raise ValueError('Teacher/student correspondence failure: '+key)
    if record.get('historical_partition')!='train' or record.get('internal_partition')!='optimization':raise PermissionError('Only optimization teacher references')

def load_teacher(record,row,contract):
    verify_alignment(record,row,contract)
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from teacher21 import query_arrays
    path=(ROOT/record['cache']).resolve()
    if not path.is_relative_to(ROOT/'data') or sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Teacher image/path registration changed')
    a=query_arrays(path,record['cache_sha256'])
    if not ((a['square_cxcywh']>=0).all() and (a['square_cxcywh']<=1).all() and (a['square_cxcywh'][:,2:]>0).all()):raise ValueError('Invalid teacher square coordinates')
    a['logits'].setflags(write=False);a['square_cxcywh'].setflags(write=False)
    return Teacher(a['logits'],a['square_cxcywh'],digest(record))
