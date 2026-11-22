"""C21 six-box adapter plus actual OWLv2 derived tensor protection."""
import sys
import torch
from commonD import ROOT
sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
from box21 import PretrainedBoxScope,BOX_KEYS,restore_delta
from telemetry19 import tensor_hash


def unregistered_tensors(model):
    tensors={}
    for prefix,module in model.named_modules():
        for name,value in vars(module).items():
            if isinstance(value,torch.Tensor) and name not in module._parameters and name not in module._buffers:
                tensors[(prefix+'.' if prefix else '')+name]=value
    return tensors


class BoxScopeD(PretrainedBoxScope):
    def __init__(self,model):
        # OWLv2.box_bias is a derived plain Tensor,not a registered buffer.
        values=unregistered_tensors(model)
        if set(values)!={'box_bias'}:raise ValueError('Unexpected actual unregistered tensors')
        method=getattr(model.compute_box_bias,'__wrapped__',None)
        expected=(method(model,model.num_patches_height,model.num_patches_width) if method is not None
                  else model.compute_box_bias(model.num_patches_height,model.num_patches_width))
        if values['box_bias'].requires_grad or not torch.equal(values['box_bias'].cpu(),expected.cpu()):
            raise ValueError('Original computed grid/box bias required')
        self.derived_hashes={n:tensor_hash(v) for n,v in values.items()}
        super().__init__(model)

    def check(self,optimizer=None,require_gradients=False,check_hash=False):
        super().check(optimizer,require_gradients,check_hash)
        current=unregistered_tensors(self.model)
        if set(current)!=set(self.derived_hashes) or any(v.requires_grad or tensor_hash(v)!=self.derived_hashes[n] for n,v in current.items()):
            raise ValueError('Derived original box-bias tensor changed')

    def checkpoint_payload(self,updates,binding_sha256):
        payload=super().checkpoint_payload(updates,binding_sha256)
        payload['unregistered_tensor_sha256']=dict(self.derived_hashes)
        return payload
