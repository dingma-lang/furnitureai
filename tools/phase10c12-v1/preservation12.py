"""Optional L2-SP on exactly the verified OWLv2 classification projection."""
import math
import torch
from types import MappingProxyType
from common12 import PROJECTION_KEYS,PROJECTION_SHAPES,CALIBRATION_KEYS,tensor_sha


class ProjectionPreserver:
    def __init__(self,model,reference,*,enabled=False,strength=0.0,expected_hashes):
        if type(enabled) is not bool or type(strength) not in (int,float) or not math.isfinite(strength) or strength<0 or (enabled and strength<=0):raise ValueError('Invalid optional preservation configuration')
        params=dict(model.named_parameters());layer=model.class_head.dense0
        if not isinstance(layer,torch.nn.Linear) or (layer.in_features,layer.out_features)!=(768,512):raise ValueError('Unverified classification architecture')
        if set(expected_hashes)!=set(PROJECTION_KEYS):raise ValueError('Exactly two original projection hashes required')
        self.params={n:params[n] for n in PROJECTION_KEYS};self._reference={};self._device_reference={}
        for name,shape in zip(PROJECTION_KEYS,PROJECTION_SHAPES):
            p=self.params[name];r=reference[name]
            if tuple(p.shape)!=shape or tuple(r.shape)!=shape or p.dtype!=torch.float32 or r.dtype!=torch.float32 or not p.requires_grad or not torch.isfinite(r).all():raise ValueError('Projection shape/dtype/trainability/reference mismatch')
            self._reference[name]=r.detach().cpu().clone()
            if tensor_sha(self._reference[name])!=expected_hashes[name]:raise ValueError('Reference is not the pinned original projection')
        self.hashes=MappingProxyType(dict(expected_hashes));self._enabled=enabled;self._strength=float(strength);self._model=model;self.check()

    @property
    def enabled(self):return self._enabled

    @property
    def strength(self):return self._strength

    def check(self):
        params=dict(self._model.named_parameters())
        for name in PROJECTION_KEYS:
            if params[name] is not self.params[name] or not params[name].requires_grad or params[name].dtype!=torch.float32 or tensor_sha(self._reference[name])!=self.hashes[name] or self._reference[name].requires_grad or self._reference[name].grad is not None:raise ValueError('Projection/reference identity or immutability violation')
            if name in self._device_reference and tensor_sha(self._device_reference[name])!=self.hashes[name]:raise ValueError('Device reference mutated')
        if any(params[n].requires_grad or params[n].grad is not None for n in CALIBRATION_KEYS):raise ValueError('Pretrained calibration must remain frozen')
        return True

    def _ref(self,name):
        p=self.params[name]
        if name not in self._device_reference or self._device_reference[name].device!=p.device:self._device_reference[name]=self._reference[name].to(p.device).clone().detach()
        return self._device_reference[name]

    def penalty(self):
        """Half squared SUM, FP32 outside AMP; independent of annotation counts."""
        self.check();device=self.params[PROJECTION_KEYS[0]].device
        with torch.autocast(device_type=device.type,enabled=False):
            value=.5*torch.stack([(self.params[n].float()-self._ref(n)).square().sum() for n in PROJECTION_KEYS]).sum()
        if not torch.isfinite(value):raise ValueError('Nonfinite projection-preservation penalty')
        return value

    def analytic_gradients(self):
        self.check()
        return {n:(self.strength*(p.detach()-self._ref(n))) if self.enabled else torch.zeros_like(p) for n,p in self.params.items()}

    def gradient_diagnostics(self):
        """After unscale, before clipping; does not modify accumulated gradients."""
        reg=self.analytic_gradients();result={}
        for n,p in self.params.items():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Expected finite accumulated projection gradient')
            result[n]=dict(weighted_regularizer_gradient_norm=float(reg[n].double().norm()),existing_objective_gradient_norm_after_subtraction=float((p.grad.detach()-reg[n]).double().norm()),combined_preclip_gradient_norm=float(p.grad.detach().double().norm()))
        return result

    def state(self):
        self.check()
        return dict(schema='phase10c12-projection-preservation-v1',enabled=self.enabled,lambda_value=self.strength,selected_keys=list(PROJECTION_KEYS),form='half_squared_L2_sum',scalars=393728,reference_hashes=dict(self.hashes),frozen_calibration_keys=sorted(CALIBRATION_KEYS),accumulation='Average existing+weighted_penalty per microbatch; each optimizer group contributes penalty once including final3')
