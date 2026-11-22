"""Failure-safe, bounded receipts; no photograph pixels or optimizer mutation."""
import contextlib
import hashlib
import math
import traceback
import torch
from common19 import DATA19,emit19,sha,time,uuid
from guard19 import CLASS_KEYS,norm64,tolerance_for


def tensor_hash(tensor):
    x=tensor.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def model_fingerprint(model):
    h=hashlib.sha256();count=0
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,tensor in sorted(items):
            h.update(kind.encode());h.update(name.encode());h.update(tensor_hash(tensor).encode());count+=1
    return dict(sha256=h.hexdigest(),tensors=count,scope='all provided model parameters and buffers')


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return 'nonfinite:'+str(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def gradient_snapshot(acc):
    empty=dict(positive_norm=None,negative_pre_norm=None,negative_post_norm=None,cap_coefficient=None,expected_max_negative_norm=None,actual_negative_positive_ratio=None,numerical_tolerance=None,cap_active=None,negative_positive_cosine=None,nonfinite_gradient=None,unavailable_reason='No accumulated components measured yet')
    if acc is None or acc.positive is None:return empty
    try:
        maps={name:{k:v/acc.scale for k,v in getattr(acc,name).items()} for name in ('base','positive','negative')}
        bad={name:[k for k,v in values.items() if not torch.isfinite(v).all()] for name,values in maps.items()}
        result=dict(empty,accumulated_microbatches=acc.count,expected_microbatches=acc.expected,verified_positive_targets=acc.positive_count,amp_scale=acc.scale,nonfinite_gradient=any(bad.values()),nonfinite_parameters=bad,post_measurement='Candidate direct source vector before optimizer;not an applied optimizer update',component_hashes={name:{k:tensor_hash(v) for k,v in values.items()} for name,values in maps.items()})
        if result['nonfinite_gradient']:
            result['unavailable_reason']='Nonfinite components; finite norms unavailable';return result
        p=maps['positive'];n=maps['negative'];pn=norm64(p);nn=norm64(n);budget=.01 if acc.positive_count==0 else .25*pn
        if acc.positive_count and nn and pn<=1e-8:
            return dict(result,positive_norm=pn,negative_pre_norm=nn,expected_max_negative_norm=budget,unavailable_reason='Unusable verified-positive anchor; no cap applied')
        coefficient=min(1.,budget/nn) if nn else 1.;post={k:v*coefficient for k,v in n.items()};used=norm64(post);tol=tolerance_for(budget,sum(v.numel() for v in p.values()),coefficient)
        from measure17 import cos
        result.update(positive_norm=pn,negative_pre_norm=nn,negative_post_norm=used,cap_coefficient=coefficient,expected_max_negative_norm=budget,actual_negative_positive_ratio=used/pn if pn else None,numerical_tolerance=tol,cap_active=coefficient<1,negative_positive_cosine=cos(n,p) if pn and nn else None,unavailable_reason=None)
        return result
    except Exception as exc:
        return dict(empty,unavailable_reason='Gradient diagnostics unavailable: '+type(exc).__name__+': '+str(exc))


class FailureTelemetry:
    """Wrap future startup/forward/group/step operations in failure_scope.

    Update context BEFORE operations; call boundary BEFORE cap composition.
    A failure always names unknown measurements instead of inventing values.
    """
    def __init__(self,context=None,model_provider=None,optimizer_provider=None,resource_provider=None,folder=None):
        self.context=dict(current_photo_id=None,current_source_group_id=None,photo_ids=[],source_group_ids=[],epoch_zero_based=None,accumulation_group_zero_based=None,microbatch_index=None,updates_committed=None,next_update=None)
        self.context.update(context or {});self.model_provider=model_provider;self.optimizer_provider=optimizer_provider;self.resource_provider=resource_provider
        self.folder=DATA19/'telemetry' if folder is None else folder
        self.started=time.monotonic();self.gradient=gradient_snapshot(None);self.boundary_reference=None

    def update(self,**context):self.context.update(context)

    def _provider(self,callback):
        if callback is None:return dict(available=False,reason='Not initialized or not provided')
        try:return dict(available=True,value=callback())
        except BaseException as exc:return dict(available=False,reason=type(exc).__name__+': '+str(exc))

    def _receipt(self,kind):
        return dict(schema='phase10c19-failure-telemetry-v1',kind=kind,context=dict(self.context),elapsed_seconds=time.monotonic()-self.started,gradients=self.gradient,model_state_fingerprint=self._provider(self.model_provider),optimizer_state_availability=self._provider(self.optimizer_provider),resources=self._provider(self.resource_provider),photograph_pixels_saved=False)

    def boundary(self,acc):
        self.gradient=gradient_snapshot(acc)
        record=self._receipt('before-cap-boundary');path=self.folder/('boundary-'+uuid.uuid4().hex+'.json');emit19(path,clean(record))
        self.boundary_reference=dict(file=str(path),sha256=sha(path));return record

    @contextlib.contextmanager
    def failure_scope(self):
        try:yield self
        except BaseException as exc:
            record=self._receipt('failure');record.update(exception_type=type(exc).__name__,exception=str(exc),stack_trace=traceback.format_exc(),last_boundary=self.boundary_reference,automatic_retry=False)
            if hasattr(exc,'diagnostics'):record['guard_exception_diagnostics']=exc.diagnostics
            path=self.folder/('failure-'+uuid.uuid4().hex+'.json')
            try:emit19(path,clean(record))
            except BaseException as write_error:
                # Surface a persistence failure while retaining the original exception.
                exc.add_note('Failure telemetry could not be persisted: '+type(write_error).__name__+': '+str(write_error))
            raise


def compose_with_telemetry(acc,telemetry):
    """Persist scalar/hash evidence BEFORE checking and composing a group."""
    from guard19 import compose_checked
    with telemetry.failure_scope():
        telemetry.boundary(acc)
        return compose_checked(acc)
