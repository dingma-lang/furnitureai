"""Restore saved C18 optimization state before any forward; no restart fallback."""
import random
import numpy as np
import torch
from common20 import *
from telemetry19 import tensor_hash


def restore_state(model,optimizer,scaler,payload):
    heads={n:p for n,p in model.named_parameters() if p.requires_grad}
    if list(heads)!=list(payload['weights']) or set(heads)!=HEAD_KEYS:raise ValueError('Recovery parameter order/scope mismatch')
    with torch.no_grad():
        for k,p in heads.items():p.copy_(payload['weights'][k].to(p.device))
    optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
    for index,(name,p) in enumerate(heads.items()):
        if not torch.equal(p.detach().cpu(),payload['weights'][name]):raise ValueError('Restored head differs')
        for field in ('step','exp_avg','exp_avg_sq'):
            if not torch.equal(optimizer.state[p][field].detach().cpu(),payload['optimizer']['state'][index][field]):raise ValueError('Restored AdamW state differs')
    if scaler.state_dict()!=payload['scaler']:raise ValueError('Restored scaler differs')
    optimizer.zero_grad(set_to_none=True)


def restore_recorded_rng(payload,use_cuda):
    random.setstate(payload['rng']['python']);torch.set_rng_state(payload['rng']['cpu'])
    if use_cuda:
        if len(payload['rng']['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(payload['rng']['cuda'])
        if any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),payload['rng']['cuda'])):raise ValueError('CUDA RNG roundtrip mismatch')
    if random.getstate()!=payload['rng']['python'] or not torch.equal(torch.get_rng_state(),payload['rng']['cpu']):raise ValueError('Python/CPU RNG roundtrip mismatch')
    return dict(Python_restored=True,PyTorch_CPU_restored=True,CUDA_restored=use_cuda,CUDA_roundtrip_verified=use_cuda,NumPy_restored=False,NumPy_reason='Absent and unused in original C18 numerical/data path;not fabricated',CPU_RNG_sha256=tensor_hash(payload['rng']['cpu']),CUDA_RNG_sha256=[tensor_hash(v) for v in payload['rng']['cuda']],strict_uninterrupted_bitwise_replay=False)


def numpy_state():
    s=np.random.get_state()
    return dict(generator=s[0],keys=[int(x) for x in s[1]],position=int(s[2]),has_gauss=int(s[3]),cached_gaussian=float(s[4]))
