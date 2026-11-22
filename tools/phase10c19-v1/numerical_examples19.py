"""Small synthetic CPU witnesses, explicitly not missing GPU buffers."""
import torch
from common19 import *
from guard19 import compose_checked,verify_direct,norm64,tolerance_for,CapViolation
from balanced16 import Accumulator,CLASS_KEYS
from measure17 import finish_accumulator


def run():
    base={k:torch.full((2,),64.) for k in HEAD_KEYS};p={k:torch.full((2,),.001) for k in CLASS_KEYS};n={k:torch.full((2,),-10.) for k in CLASS_KEYS};acc=Accumulator(1,128)
    acc.add({k:v*128 for k,v in base.items()},{k:v*128 for k,v in p.items()},{k:v*128 for k,v in n.items()},1)
    expected,cap=finish_accumulator(acc);out,metrics=compose_checked(acc)
    effective=norm64({k:out[k]-base[k] for k in CLASS_KEYS})
    bad={k:v*cap['scale']*1.001 for k,v in n.items()}
    try:verify_direct(p,n,bad,cap['scale'],1)
    except CapViolation as exc:violation=dict(rejected=True,exception=str(exc),diagnostics=exc.diagnostics)
    else:raise ValueError('Real violation was not rejected')
    elements=sum(v.numel() for k,v in original_heads().items() if k in CLASS_KEYS)
    emit19(REPORT19/'numerical-repair.json',dict(scope='Synthetic CPU witnesses only;no exact failed GPU values recovered',source_norm=metrics['source_post']['norm'],budget=cap['budget'],subtractive_effective_norm=effective,old_guard_would_reject=effective>cap['budget']*(1+1e-6)+1e-8,new_guard_accepts=metrics['direct_guard']['cap_verified'],all12_composed_outputs_bit_identical_to_C16=all(torch.equal(out[k],expected[k]) for k in out),direct_guard=metrics['direct_guard'],real_violation_example=violation,actual_classification_elements=elements,full_scope_tolerance_for_unit_budget=tolerance_for(1,elements,.5),historical_gpu_failed_values_available=False))
    print('Synthetic false positive accepted with identical outputs;real over-cap vector rejected',flush=True)


if __name__=='__main__':run()
