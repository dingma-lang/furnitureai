"""Disabled-by-default75-update candidate. Never launched by E4 recovery.

Retains original FP16 forward,uses verified FP32 observed-node classification
adjoint and native box backward at scale1. No GradScaler/manual128 instrumentation.
"""
import argparse
import random
import signal
import traceback
from common4 import ROOT,REPORT,BASE,BASE_SHA,PREP,PROMPTS,read,sha,digest,os,sys,time,uuid,subprocess

FUTURE=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1'


def gate(approval):
    if not isinstance(approval,dict) or approval.get('approved') is not True or approval.get('mode')!='teacher_distillation_training':
        raise PermissionError('Separate explicit training authorization required;E4 approval is zero-update only')
    path=REPORT/'training-disabled.json';plan=read(path)
    if plan['approved'] is not False or plan['enabled'] is not False or approval.get('proposal_sha256')!=sha(path):
        raise PermissionError('Exact disabled proposal must be separately authorized')
    if approval.get('limits')!=plan['limits'] or approval.get('coefficient')!=plan['coefficient']:
        raise PermissionError('Authorized coefficient/resource mismatch')
    for rel,h in plan['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned training input changed: '+rel)
    return plan


def worker(approval):
    plan=gate(approval) # BEFORE torch imports,Cuda,model construction or files.
    launch=read(FUTURE/'launch-once.json')
    if launch['nonce']!=os.environ.get('FURNITURE_E_LAUNCH_NONCE'):raise PermissionError('Bounded future launcher required')
    import torch
    import numpy as np
    import threading
    from PIL import Image
    from transformers import Owlv2ForObjectDetection,Owlv2Processor
    from scopeE import StudentScope,TRAIN_KEYS,HEAD_KEYS
    from teacherE import load_teacher
    from common8 import image_path,grouped_epoch
    from common10 import named_sha
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    sys.path.insert(0,str(ROOT/'tools/phase10e2-v1'))
    from measure2 import replay
    from precision4 import combined
    sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
    from stable3 import Capture
    from expanded import from_square,cxcywh_to_xyxy
    from workerE import Output
    out=Output(FUTURE,plan['limits']['output_bytes']);begin=launch['monotonic_start'];stop=threading.Event()
    state=dict(forwards=0,updates=0,contractions=0,photo=None);scope=None;last=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    previous_TF32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((plan['limits']['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def resources():return dict(state,seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),allocated_output_bytes=out.allocated())
        def guard():
            d=resources()
            if stop.is_set() or (FUTURE/'STOP').exists() or d['seconds']>=590 or d['peak_allocated_bytes']>plan['limits']['VRAM_bytes'] or d['allocated_output_bytes']>plan['limits']['output_bytes'] or state['updates']>75 or state['forwards']>373 or state['contractions']>354:
                raise RuntimeError('Future training stop/resource boundary')
        context=read(PREP/'training-context.json');entries=context['entries'];index=read(PREP/'teacher-index.json')
        byid={e['row']['image_id']:e for e in entries};refs={r['image_id']:r for r in index['records']}
        train=[e['row'] for e in entries];dev=context['development'];prepared=dict(rows=train+dev)
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True);model_state_matches_base(model)
        model=model.to('cuda');scope=StudentScope(model);processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        params=[scope.params[n] for n in scope.names]
        binding=digest(approval)
        def forward(row,optimization=True):
            guard();state['photo']=row['image_id']
            with Image.open(image_path(row,'optimization' if optimization else 'inner_development',prepared)) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image registration mismatch')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if optimization:
                seen={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=__import__('hashlib').sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
                if seen!=refs[row['image_id']]['reconstructed_processor_tensors']:raise ValueError('Input identity mismatch')
            if state['forwards']>=373:raise RuntimeError('Forward cap before call')
            state['forwards']+=1;out.json('telemetry/forward-'+str(state['forwards']).zfill(3)+'.json',dict(state,source_sha256=row['sha256']))
            if optimization:
                if state['contractions']>=354:raise RuntimeError('Head contraction cap before call')
                state['contractions']+=1
                with Capture(model.class_head,enabled=True) as c,torch.autocast('cuda',dtype=torch.float16):pred=model(**{k:v.to('cuda') for k,v in batch.items()})
                return pred.logits[0].float(),pred.pred_boxes[0].float(),c.snapshot
            with torch.inference_mode():pred=model(**{k:v.to('cuda') for k,v in batch.items()})
            return pred.logits[0].float(),pred.pred_boxes[0].float(),None
        # Mandatory original correspondence and a zero-update combined-VJP check
        # BEFORE optimizer construction. Does not require an additional forward.
        before=named_sha(model,TRAIN_KEYS)
        for i,entry in enumerate(entries,1):
            row=entry['row'];teacher=load_teacher(refs[row['image_id']],row,index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda')
            z,b,s=forward(row);r=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),dict(logits=teacher.logits,square_cxcywh=teacher.boxes))
            loss,parts,stats=combined(model.class_head,s,b,tz,tb,entry,plan['coefficient'],enabled=True)
            if i==1:
                gradients=torch.autograd.grad(loss,params)
                if any(not torch.isfinite(g).all() for g in gradients) or any(p.grad is not None for p in model.parameters()):raise ValueError('Combined precision zero-update smoke failed')
                out.json('initial/combined-smoke.json',dict(loss=float(loss.detach()),parameter_gradient_norms=[float(g.double().norm()) for g in gradients],loss_scale=1.,optimizer_updates=0))
                del gradients
            out.json('initial/'+str(i).zfill(2)+'.json',dict(image_id=row['image_id'],replay=r,unit_KD_loss=float(parts['distillation'].detach())))
            del z,b,s,tz,tb,loss,parts
        if named_sha(model,TRAIN_KEYS)!=before:raise ValueError('Initial smoke changed parameters')
        scope.check(hash_frozen=True);optimizer=scope.optimizer() # Reached only with future explicit approval.
        for epoch in range(5):
            for gi,group in enumerate(grouped_epoch(train,epoch)):
                guard();scope.train_mode();optimizer.zero_grad(set_to_none=True);values=[]
                if [r['image_id'] for r in group]!=context['orders'][epoch][gi]:raise ValueError('Data order drift')
                for row in group:
                    entry=byid[row['image_id']];teacher=load_teacher(refs[row['image_id']],row,index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda')
                    z,b,s=forward(row);loss,parts,stats=combined(model.class_head,s,b,tz,tb,entry,plan['coefficient'],enabled=True)
                    (loss/len(group)).backward() # Explicit scale1,FP32 classification adjoint,native box backward.
                    values.append(dict(image_id=row['image_id'],source_group=row['identity_group'],stats=stats));del z,b,s,tz,tb,loss,parts
                norms=scope.gradients(optimizer);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if state['updates']>=75:raise RuntimeError('Update cap before call')
                out.json('updates/'+str(state['updates']+1).zfill(3)+'.intent.json',dict(state,epoch=epoch+1,source_order=[r['image_id'] for r in group]))
                guard();optimizer.step();state['updates']+=1;scope.check(optimizer)
                if any(not torch.isfinite(p).all() for p in params):raise ValueError('Nonfinite updated weight')
                out.json('updates/'+str(state['updates']).zfill(3)+'.json',dict(resources=resources(),epoch=epoch+1,images=values,gradient_norms=norms,loss_scale=1.))
            scope.check(optimizer,hash_frozen=True)
            payload=scope.payload(optimizer,state['updates'],binding,epoch=epoch+1,next_group=0,coefficient=plan['coefficient'],precision=plan['precision'],rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            import io
            raw=io.BytesIO();torch.save(payload,raw);path=out.write('checkpoints/epoch-'+str(epoch+1)+'.pt',raw.getvalue())
            last=dict(file=str(path.relative_to(FUTURE)),sha256=sha(path),updates=state['updates'])
            check=torch.load(path,map_location='cpu',weights_only=True)
            if any(not torch.equal(payload['weights'][n],check['weights'][n]) for n in HEAD_KEYS):raise ValueError('Checkpoint integrity failed')
            out.json('checkpoints/epoch-'+str(epoch+1)+'.json',last)
        scope.eval_mode();optimizer.zero_grad(set_to_none=True);records=[]
        for i,row in enumerate(dev,1):
            z,b,_=forward(row,False);scores,labels=z.sigmoid().max(-1);boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).clamp(0,1)
            p=out.npz('predictions/'+str(i).zfill(2)+'.npz',dict(logits=z.cpu().numpy(),square_cxcywh=b.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),boxes=boxes.cpu().numpy()))
            records.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],file=p.name,sha256=sha(p)));del z,b,scores,labels,boxes
        out.json('predictions/raw-complete.json',dict(FP32=True,rows_sha256=context['development_rows_sha256'],records=records))
        scope.check(optimizer,hash_frozen=True);guard()
        if state['updates']!=75 or state['forwards']!=373 or state['contractions']!=354 or sha(BASE/'model.safetensors')!=BASE_SHA:raise ValueError('Endpoint/preservation mismatch')
        out.json('complete.json',dict(status='PASS',resources=resources(),checkpoint=last,coefficient=plan['coefficient'],precision=plan['precision'],automatic_retry=False))
    except BaseException as exc:
        try:out.json('failure.json',dict(error=type(exc).__name__+': '+str(exc),stack=traceback.format_exc(),state=state,seconds=time.monotonic()-begin,last_checkpoint=last,automatic_retry=False))
        except BaseException as secondary:exc.add_note('Failure persistence: '+str(secondary))
        raise
    finally:torch.backends.cuda.matmul.allow_tf32=previous_TF32


def launch(approval):
    plan=gate(approval)
    if FUTURE.exists():raise FileExistsError('Fresh future training destination required')
    import signal
    FUTURE.mkdir(parents=True);nonce=uuid.uuid4().hex;begin=time.monotonic()
    # WorkerE.Output is reused only for atomic publication,not the legacy loss.
    with (FUTURE/'launch-once.json').open('x') as f:
        __import__('json').dump(dict(monotonic_start=begin,nonce=nonce),f)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONFAULTHANDLER='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',FURNITURE_E_LAUNCH_NONCE=nonce)
    with (FUTURE/'console.log').open('x') as log:
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(__file__),'--worker','--authorization',approval['_path']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=max(.1,595-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=max(.1,600-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=proc.returncode
    if code:raise RuntimeError('Single future training attempt stopped;no automatic retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--authorization');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.authorization is None:raise PermissionError('No training authorization supplied;E4 does not authorize training')
    approval=read(a.authorization);approval['_path']=a.authorization
    if a.worker:worker(approval)
    else:launch(approval)
