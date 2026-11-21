"""C11 full-head snapshot format plus immutable C12 objective/reference binding."""
from common12 import *
from freeze10 import make_payload as legacy_payload,restore_candidate as legacy_restore


def make_payload(scope,optimizer,updates,binding_sha256,*,preserver,**state):
    result=legacy_payload(scope,optimizer,updates,binding_sha256,**state)
    result['projection_preservation']=preserver.state()
    return result


def restore_candidate(scope,payload,optimizer=None,expected_binding=None,*,preserver):
    # Reject changed references/lambda/enable flag BEFORE copying any tensors.
    if expected_binding is None or payload.get('binding_sha256')!=expected_binding or payload.get('projection_preservation')!=preserver.state():raise ValueError('C12 checkpoint configuration/reference mismatch')
    updates=payload.get('updates');epoch=payload.get('epoch');cursor=payload.get('next_group')
    if type(epoch) is not int or type(cursor) is not int or not 0<=epoch<5 or not 0<=cursor<=15 or type(updates) is not int or updates!=15*epoch+cursor:raise ValueError('C12 checkpoint update/epoch/group cursor mismatch')
    legacy_restore(scope,payload,optimizer,expected_binding);preserver.check()


def restore_training_state(scope,payload,optimizer,scaler,expected_binding,*,preserver):
    """Explicit committed-boundary recovery helper, never automatic update replay.

    GPU RNG restoration needs separately authorized execution. CPU tests use the
    same helper with no CUDA RNG records; no historical state is modified.
    """
    import torch,random
    rng=payload.get('rng');scale=payload.get('scaler')
    if not isinstance(rng,dict) or set(rng)!={'python','cpu','cuda'} or not isinstance(scale,dict):raise ValueError('Complete RNG/scaler recovery state required')
    if rng['cuda'] and not torch.cuda.is_initialized():raise ValueError('CUDA RNG restore requires authorized initialized GPU')
    restore_candidate(scope,payload,optimizer,expected_binding,preserver=preserver)
    scaler.load_state_dict(scale);random.setstate(rng['python']);torch.set_rng_state(rng['cpu'])
    if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
    return payload['epoch'],payload['next_group'],payload['updates']
