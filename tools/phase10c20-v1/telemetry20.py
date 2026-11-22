"""C19 failure fields routed atomically to C20's approved fresh destination."""
import traceback
from common20 import *
from telemetry19 import FailureTelemetry,gradient_snapshot,clean


class RuntimeTelemetry(FailureTelemetry):
    def __init__(self,**kwargs):
        super().__init__(folder=RUN20/'failures',**kwargs)
        self.accumulator=None

    def boundary(self,acc):
        self.accumulator=acc;self.gradient=gradient_snapshot(acc)
        record=self._receipt('before-cap-boundary');path=self.folder/('boundary-'+uuid.uuid4().hex+'.json');emit20(path,clean(record))
        self.boundary_reference=dict(file=str(path.relative_to(ROOT)),sha256=sha(path));return record

    def save_failure(self,exc):
        self.gradient=gradient_snapshot(self.accumulator)
        record=self._receipt('failure');record.update(exception_type=type(exc).__name__,exception=str(exc),stack_trace=traceback.format_exc(),last_boundary=self.boundary_reference,automatic_retry=False)
        if hasattr(exc,'diagnostics'):record['guard_exception_diagnostics']=exc.diagnostics
        emit20(self.folder/('failure-'+uuid.uuid4().hex+'.json'),clean(record))
