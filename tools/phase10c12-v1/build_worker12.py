"""Derive an isolated future worker with a reviewable C11 delta; never run it."""
from common12 import *


def transformed_source():
    source=(ROOT/'tools/phase10c11-v1/worker11.py').read_text()
    changes={
        'Phase10C.11 authorized execution; prepared algorithm plus detached diagnostics.':'Phase10C.12 future worker. DISABLED; no GPU authorization during preparation.',
        'from common10 import *':'from common12 import *',
        'from config10 import require_training_approval':'from config12 import require_training_approval',
        'from observe11 import objective,verify_initial_outputs11':'from objective12 import objective\n        from observe12 import verify_initial_outputs12\n        from preservation12 import ProjectionPreserver',
        'from freeze10 import FrozenScope,head_weights,make_payload,restore_candidate':'from freeze10 import FrozenScope,head_weights\n        from checkpoint12 import make_payload,restore_candidate',
        'scope=FrozenScope(model,original_heads());optimizer=scope.optimizer();scaler=':'scope=FrozenScope(model,original_heads());optimizer=scope.optimizer();regcfg=cfg[\'projection_preservation\'];preserver=ProjectionPreserver(model,original_heads(),enabled=regcfg[\'enabled\'],strength=regcfg[\'lambda_value\'],expected_hashes=regcfg[\'reference_hashes\']);scaler=',
        'protected[ident],True);scale=':'protected[ident],True,preserver);scale=',
        "protected[row['image_id']],True);scaler.scale":"protected[row['image_id']],True,preserver);scaler.scale",
        'weights_unchanged=True)':'weights_unchanged=True,projection_regularizer_gradient_norms={n:float(t.double().norm()) for n,t in preserver.analytic_gradients().items()})',
        "sha(FUTURE/'authorized-config.json'),epoch=":"sha(FUTURE/'authorized-config.json'),preserver=preserver,epoch=",
        "if state['updates']!=updates or any(":"if state.get('projection_preservation')!=preserver.state() or state['updates']!=updates or any(",
        'verify_initial_outputs11()':'verify_initial_outputs12()',
        'scaler.unscale_(optimizer);norms=scope.gradients(optimizer);':'scaler.unscale_(optimizer);norms=scope.gradients(optimizer);projection_gradient_parts=preserver.gradient_diagnostics();',
        'named_preclip_gradients=norms,images=values':'named_preclip_gradients=norms,projection_gradient_parts=projection_gradient_parts,images=values',
        'final_checkpoint=last,another_experiment_authorized=False':'final_checkpoint=last,projection_preservation=preserver.state(),another_experiment_authorized=False',
    }
    for old,new in changes.items():
        if old not in source:raise ValueError('C11 source no longer matches prepared delta: '+old)
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    source=transformed_source();p=HERE12/'future_worker12.py'
    if p.exists() and p.stat().st_size:raise FileExistsError('Preserve existing prepared worker')
    p.write_text(source)
