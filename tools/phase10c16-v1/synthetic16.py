"""Fixed synthetic duplicate-count fixtures,not model/data or coefficient sweep."""
import torch
from common16 import *
from balanced16 import *
from test_balanced16 import fixture
from expanded import xyxy_to_cxcywh


def run():
    results=[]
    for copies in (1,10,100):
        _,_,row,e,ctx=fixture();params={k:torch.zeros(2,requires_grad=True) for k in sorted(CLASS_KEYS)};zvalue=sum(p.sum() for p in params.values());z=torch.stack([torch.stack([zvalue if c==1 else zvalue*0 for c in range(7)]) for q in range(copies+1)])
        b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9]]+[[.1,.1,.2,.2]]*copies,dtype=torch.float32)).requires_grad_();r=branches(z,b,row,e,[],ctx,Policy(enabled=True));positive=dict(zip(params,torch.autograd.grad(r['positive_anchor'],tuple(params.values()),retain_graph=True)));outputs={}
        for name,key in [('focal','existing_source_focal'),('BCE','unweighted_source_BCE'),('balanced_raw','source_raw')]:
            ng=dict(zip(params,torch.autograd.grad(r[key],tuple(params.values()),retain_graph=True)));dot=sum(float((positive[k].double()*ng[k].double()).sum()) for k in params);outputs[name]=dict(loss=float(r[key].detach()),negative_parameter_norm=norm(ng.values()),positive_parameter_norm=norm(positive.values()),negative_positive_cosine=dot/(norm(positive.values())*norm(ng.values())),negative_to_positive_ratio=norm(ng.values())/norm(positive.values()))
            if name=='balanced_raw':
                base={k:(positive[k] if k in positive else torch.zeros(2)) for k in HEAD_KEYS};_,d=compose(base,positive,ng,1,Policy(enabled=True));outputs['balanced_capped']=d
        results.append(dict(synthetic=True,exact_negative_query_copies=copies,unique_signatures=r['normalization']['unique_query_signatures'],results=outputs,model_updates=0))
    emit16(REPORT16/'synthetic-comparison.json',dict(scope='Synthetic shared-linear6tensor graph only,not OWLv2 inference',cases=results,fixed_policy=True,coefficient_sweeps=0,classification_projection_measurements_on_real_model=False));print('Synthetic1/10/100duplicate fixtures complete;no model inference',flush=True)


if __name__=='__main__':run()
