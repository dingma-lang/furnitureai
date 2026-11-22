"""Component-level, failure-safe integration for a separately approved recovery.

No inference, optimizer or GPU initialization here. Caller supplies one verified
live forward and an authorized resource guard. Historical E1 objective untouched.
"""
from pathlib import Path
import sys
import traceback

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase10e1-v1'))
from objectiveE import objective, selective_mask
from stable3 import observed_logits, measured_vjp
from telemetry3 import Trace, atomic_json, persist_component, stats


def measure_components(head, snapshot, boxes, teacher_logits, teacher_boxes, entry,
                       box_parameters, destination, *, guard, context, enabled=False):
    if not enabled:
        raise PermissionError('Diagnostic recovery is disabled by default')
    row = entry['row']
    if (context.get('internal_partition') != 'optimization' or
            row.get('historical_partition') != 'train' or row.get('internal_partition') != 'optimization' or
            context.get('image_id') != row['image_id'] or
            context.get('source_group') != row['identity_group'] or context.get('source_sha256') != row['sha256']):
        raise PermissionError('Only bound optimization-training photographs')
    directory = Path(destination)
    if directory.exists():
        raise FileExistsError('No overwrite or automatic retry')
    directory.mkdir(parents=True)
    trace = Trace(directory / 'trace', enabled=True, context=dict(context, mode='observed-node-FP32-adjoint',
                  loss_scale=1., optimizer=False, coefficient=None, torch_version=torch.__version__,
                  device=str(snapshot.logits.device), forward_logit_dtype=str(snapshot.logits.dtype),
                  observed_projection_dtype=str(snapshot.raw.dtype), adjoint_dtype='torch.float32',
                  autocast_during_adjoint=False,
                  CUDA_TF32_matmul=(torch.backends.cuda.matmul.allow_tf32 if snapshot.logits.device.type == 'cuda' else None)))
    projection = [head.dense0.weight, head.dense0.bias]
    try:
        guard('before-components')
        z = observed_logits(head, snapshot, enabled=True, trace=trace)[0]
        b = boxes.float()
        # Exact forward means both Hungarian assignment and source masks use
        # the original outputs, not recomputed FP32 logits or new coordinates.
        mask, mask_stats = selective_mask(z, b, teacher_logits, teacher_boxes,
                          entry['row'], entry['distill_regions'], entry['distill_protected'])
        _, parts, objective_stats = objective(z, b, entry['row'], teacher_logits,
                                             teacher_boxes, entry, 1.)
        trace.context['active_distillation_terms'] = int(mask.sum())
        delta = z.detach()[mask] - teacher_logits[mask]
        terminal = 2 * delta / int(mask.sum()) if mask.any() else delta
        atomic_json(directory / 'inputs.json', dict(context=trace.context, masks=mask_stats,
                    student=stats(z), teacher=stats(teacher_logits), differences=stats(delta),
                    unit_distillation_loss=stats(parts['distillation']),
                    terminal_derivative=stats(terminal), base_objective=objective_stats['base']))
        # No autograd fallback, no suppression, no retry. Successful positive
        # measurements are durable BEFORE starting the independent KD VJP.
        guard('positive-VJP-intent')
        pg, pp = measured_vjp(parts['positive'], z, b, projection, box_parameters, trace=trace)
        persist_component(directory / 'positive.json', component='positive', output_gradient=pg,
                          projection_gradients=pp, context=trace.context)
        if not mask.any() or torch.equal(z.detach()[mask], teacher_logits[mask]):
            kg, kp = torch.zeros_like(z), [torch.zeros_like(p) for p in projection]
            atomic_json(directory / 'distillation.json', dict(component='distillation',
                        method='analytic exact zero; no VJP executed', context=trace.context))
        else:
            guard('distillation-VJP-intent')
            kg, kp = measured_vjp(parts['distillation'], z, b, projection, box_parameters, trace=trace)
            persist_component(directory / 'distillation.json', component='distillation', output_gradient=kg,
                              projection_gradients=kp, context=trace.context)
        if (kg[~mask] != 0).any():
            raise ValueError('KD gradient reached ignored/non-Chair output')
        guard('after-components')
        atomic_json(directory / 'complete.json', dict(context=trace.context, status='PASS', coefficient=None))
        # Caller aggregates actual vectors by the unchanged source groups;
        # vectors remain transient, never large/private artifact payloads.
        return dict(positive=pp, distillation=kp, positive_output=pg, distillation_output=kg)
    except BaseException as exc:
        try:
            atomic_json(directory / 'failure.json', dict(context=trace.context,
                        exception_type=type(exc).__name__, exception=str(exc),
                        stack_trace=''.join(traceback.format_exception(exc)),
                        first_observed_nonfinite=trace.first_nonfinite,
                        positive_component_persisted=(directory / 'positive.json').exists()))
        except BaseException as persistence_error:
            exc.add_note('Minimal failure receipt could not be persisted: ' + str(persistence_error))
        raise
    finally:
        trace.close()
