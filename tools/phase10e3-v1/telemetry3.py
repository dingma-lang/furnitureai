"""Optional compact, fail-closed gradient tracing. No photograph/tensor payloads."""
import json
import os
from pathlib import Path
import threading
import traceback
import uuid

import torch


def stats(value):
    if value is None:
        return {"present": False}
    t = value.detach()
    finite = torch.isfinite(t)
    # Test in the original dtype FIRST. Widen finite values before any reduction.
    v = t[finite].double()
    all_finite = bool(finite.all())
    return dict(present=True, dtype=str(t.dtype), device=str(t.device),
                shape=list(t.shape), elements=t.numel(),
                nonfinite=int((~finite).sum()), finite=all_finite,
                maximum_absolute=float(v.abs().max()) if v.numel() else None,
                minimum=float(v.min()) if v.numel() else None,
                maximum=float(v.max()) if v.numel() else None,
                L2=(float(v.norm()) if v.numel() else 0.) if all_finite else None,
                finite_subset_L2=float(v.norm()) if v.numel() else None)


def atomic_json(path, value, cap=128 * 2**20):
    """Write-once, durable JSON; retain staging evidence on interruption."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_blocks * 512 for p in path.parent.rglob('*') if p.is_file())
    if used + 2 * ((len(data) + 4095) // 4096) * 4096 + 16384 > cap:
        raise RuntimeError('Diagnostic output limit')
    pending = path.with_name('.' + path.name + '.pending-' + uuid.uuid4().hex)
    with pending.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(pending, path)  # Atomic publication, cannot replace existing records.
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Trace:
    """Disabled by default. Hooks observe gradients without modifying them.

    `first_nonfinite` is the earliest OBSERVED hook/stage, not proof of the
    originating CUDA instruction. The bounded event stream is flushed per event.
    """
    def __init__(self, directory=None, *, enabled=False, context=None, max_events=256):
        self.enabled = enabled
        self.directory = Path(directory) if directory else None
        self.context = dict(context or {})
        self.max_events = max_events
        self.events = []
        self.handles = []
        self.first_nonfinite = None
        self.lock = threading.RLock()

    def record(self, location, tensors=(), **extra):
        if not self.enabled:
            return
        with self.lock:
            if len(self.events) >= self.max_events:
                raise RuntimeError('Bounded gradient telemetry exhausted')
            tensors = list(tensors)
            item = dict(sequence=len(self.events) + 1, location=location,
                        tensors=[stats(t) for t in tensors], context=self.context, **extra)
            if self.first_nonfinite is None and any(t.get('nonfinite', 0) for t in item['tensors']):
                self.first_nonfinite = dict(sequence=item['sequence'], location=location)
            item['earliest_observed_nonfinite'] = self.first_nonfinite
            if self.directory:
                atomic_json(self.directory / ('event-%04d.json' % item['sequence']), item)
            self.events.append(item)

    def tensor(self, name, tensor):
        self.record(name + ':forward', [tensor])
        if self.enabled and tensor.requires_grad:
            def observe(gradient):
                self.record(name + ':backward', [gradient])
                return None
            self.handles.append(tensor.register_hook(observe))

    def graph(self, tensor, max_nodes=64):
        if not self.enabled:
            return
        queue = [tensor.grad_fn]
        seen = set()
        while queue:
            node = queue.pop(0)
            if node is None or node in seen:
                continue
            seen.add(node)
            if len(seen) > max_nodes:
                raise RuntimeError('Unexpected classification graph size')
            name = '%02d:%s' % (len(seen), node.name())
            def observe(inputs, outputs, name=name):
                self.record('node:' + name, [*outputs, *inputs],
                            output_slots=len(outputs), input_slots=len(inputs))
                return None
            self.handles.append(node.register_hook(observe))
            queue.extend(child for child, _ in node.next_functions)

    def failure(self, exc):
        self.record('exception', exception_type=type(exc).__name__, exception=str(exc),
                    stack_trace=''.join(traceback.format_exception(exc)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def persist_component(path, *, component, output_gradient, projection_gradients, context):
    """Publish EACH successful component before attempting another VJP."""
    atomic_json(path, dict(component=component, context=context,
                          output_gradient=stats(output_gradient),
                          projection_gradients=[stats(g) for g in projection_gradients]))
