"""Prepared single-method worker. Explicit authorization required BEFORE GPU imports."""
import argparse,signal,traceback,random,zipfile
from commonE import *
from configE import require_execution

class Output:
    def __init__(self,root,limit):
        if not root.exists() or {p.name for p in root.iterdir()}!={'launch-once.json','console.log'}:raise FileExistsError('Exactly one fresh supervised launch required;no retry')
        launch=read(root/'launch-once.json')
        if launch['nonce']!=os.environ.get('FURNITURE_E_LAUNCH_NONCE'):raise PermissionError('Use bounded launcher;direct worker execution forbidden')
        self.root=root;self.limit=limit;self.lock=threading.RLock()
        for name in ('predictions','checkpoints','telemetry','updates','initial','probes'):(root/name).mkdir()
    def allocated(self):return sum(p.stat().st_blocks*512 for p in [self.root,*self.root.rglob('*')])
    def write(self,path,data):
        path=self.root/path
        with self.lock:
            if not path.resolve().is_relative_to(self.root) or path.exists():raise PermissionError('Atomic no-replace future outputs only')
            if self.allocated()+len(data)*2+16384>self.limit:raise RuntimeError('Future output budget')
            pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
            with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.link(pending,path)
        return path
    def json(self,path,data):return self.write(path,(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    def npz(self,path,arrays):
        import numpy as np
        if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite predictions')
        b=io.BytesIO();np.savez_compressed(b,**arrays);return self.write(path,b.getvalue())

def run(approval,mode):
    require_execution(approval,mode) # No cache loading,model construction,CUDA,or mkdir before gate.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    import numpy as np
    from PIL import Image
    from transformers import Owlv2ForObjectDetection,Owlv2Processor
    from scopeE import StudentScope,TRAIN_KEYS,HEAD_KEYS,head_weights,restore_candidate,frozen_state
    from teacherE import load_teacher
    from objectiveE import objective,selective_mask,logit_diagnostics
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    from common8 import image_path,grouped_epoch,load_npz
    from expanded import from_square,cxcywh_to_xyxy
    from common10 import named_sha
    from telemetry19 import tensor_hash
    limits=approval['limits'];root=FUTURE if mode=='training' else SMOKE;out=Output(root,limits['output_bytes'])
    began=read(root/'launch-once.json')['monotonic_start'];stop=threading.Event();state=dict(forwards=0,updates=0,VJPs=0,photo=None,phase='initializing');scope=None;model=None;last=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    out.json('approval.json',approval)
    try:
        context=read(PREP/'training-context.json');entries=context['entries'];byid={e['row']['image_id']:e for e in entries};dev=context['development'];prepared=dict(rows=[e['row'] for e in entries]+dev)
        index=read(PREP/'teacher-index.json');refs={r['image_id']:r for r in index['records']};contract=index['processor_contract_sha256']
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry/fallback')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((limits['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def resources():return dict(state,seconds=time.monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),allocated_output_bytes=out.allocated())
        def guard():
            if stop.is_set() or (root/'STOP').exists() or time.monotonic()-began>=limits['seconds']-5 or torch.cuda.max_memory_allocated(0)>limits['VRAM_bytes'] or out.allocated()>limits['output_bytes']:raise RuntimeError('STOP/resource boundary')
            if state['forwards']>limits['forwards'] or state['updates']>limits['updates'] or (mode=='gradient_smoke' and state['VJPs']>48):raise RuntimeError('Execution counter boundary')
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
        model_state_matches_base(model);model=model.to('cuda');scope=StudentScope(model)
        processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        projection=[scope.params['class_head.dense0.'+s] for s in ('weight','bias')]
        out.json('initialization.json',dict(base_sha256=BASE_SHA,immutable_state=scope.immutable_state,trainable_keys=scope.names,teacher_immutable_cache=True,mode=mode))
        def forward(row,training=True):
            guard();state['photo']=row['image_id'];part='optimization' if training else 'inner_development'
            with Image.open(image_path(row,part,prepared)) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image registration changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if training:
                expected=refs[row['image_id']]['reconstructed_processor_tensors']
                actual={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
                if actual!=expected:raise ValueError('Processed pixel/token/query inputs differ from CPU-bound reconstruction')
            if state['forwards']>=limits['forwards']:raise RuntimeError('Forward cap before call')
            state['forwards']+=1;out.json('telemetry/forward-'+str(state['forwards']).zfill(3)+'.json',dict(state,source_sha256=row['sha256']))
            if training:
                with torch.autocast('cuda',dtype=torch.float16):prediction=model(**{k:v.to('cuda') for k,v in batch.items()})
            else:
                with torch.inference_mode():prediction=model(**{k:v.to('cuda') for k,v in batch.items()})
            guard();return prediction.logits[0].float(),prediction.pred_boxes[0].float()
        def gradients(loss,retain=True):
            if mode=='gradient_smoke' and state['VJPs']>=48:raise RuntimeError('48VJPcap')
            state['VJPs']+=1
            g=torch.autograd.grad(loss*128,projection,retain_graph=retain,allow_unused=False)
            if any(not torch.isfinite(v).all() for v in g):raise ValueError('Nonfinite projection gradient')
            return [v.detach().float()/128 for v in g]
        def measured(positive,kd):
            gp=gradients(positive);gk=gradients(kd)
            pn=float(sum(v.double().square().sum() for v in gp).sqrt());kn=float(sum(v.double().square().sum() for v in gk).sqrt())
            dot=float(sum((a.double()*b.double()).sum() for a,b in zip(gp,gk)))
            return dict(positive_projection_norm=pn,unit_distill_projection_norm=kn,projection_cosine=dot/(pn*kn) if pn and kn else None)
        proposal=read(REPORT/('gpu-training-disabled.json' if mode=='training' else 'gradient-smoke-disabled.json'))
        check_ids=[e['row']['image_id'] for e in entries] if mode=='training' else proposal['photo_ids'][:4]
        scope.train_mode();initial_heads=named_sha(model,TRAIN_KEYS)
        for i,ident in enumerate(check_ids,1):
            entry=byid[ident];teacher=load_teacher(refs[ident],entry['row'],contract);z,b=forward(entry['row'])
            exact=np.array_equal(z.detach().cpu().numpy(),teacher.logits) and np.array_equal(b.detach().cpu().numpy(),teacher.boxes)
            out.npz('initial/'+str(i).zfill(2)+'.npz',dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()))
            if not exact:raise ValueError('Live original FP16 output differs from cached teacher;stop,no automatic repair')
            del z,b
        if named_sha(model,TRAIN_KEYS)!=initial_heads:raise ValueError('Initial teacher replay changed student')
        if mode=='gradient_smoke':
            checkpoint=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt'
            if sha(checkpoint)!=proposal['reference_sha256']:raise ValueError('Existing C11 diagnostic checkpoint changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);restore_candidate(scope,saved);before=named_sha(model,TRAIN_KEYS);records=[]
            for i,ident in enumerate(proposal['photo_ids'],1):
                entry=byid[ident];teacher=load_teacher(refs[ident],entry['row'],contract);tz,tb=teacher.tensors('cuda');z,b=forward(entry['row'])
                cached=load_npz(ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/probes/epoch-5'/f'{i:02}.npz')
                if not np.array_equal(z.detach().cpu().numpy(),cached['logits']) or not np.array_equal(b.detach().cpu().numpy(),cached['square_cxcywh']):raise ValueError('Existing C11 gradient-reference replay mismatch;no retry')
                total,parts,stats=objective(z,b,entry['row'],tz,tb,entry,1.)
                record=dict(image_id=ident,source_group=entry['row']['identity_group'],gradients=measured(parts['positive'],parts['distillation']),stats=stats)
                out.npz('probes/'+str(i).zfill(2)+'.npz',dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()))
                out.json('probes/'+str(i).zfill(2)+'.json',record);records.append(record);del total,parts,z,b,tz,tb
            scope.check(hash_frozen=True)
            if named_sha(model,TRAIN_KEYS)!=before or any(p.grad is not None for p in model.parameters()) or state['forwards']!=28 or state['VJPs']!=48:raise ValueError('Zero-update smoke integrity/counters')
            if sha(BASE/'model.safetensors')!=BASE_SHA or sha(checkpoint)!=proposal['reference_sha256']:raise ValueError('Original/existing checkpoint file changed')
            out.json('complete.json',dict(status='PASS',mode=mode,records=records,**resources(),model_updates=0,optimizer_constructed=False,coefficient_selected=None,actual_training_not_authorized=True));return
        optimizer=scope.optimizer();scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2000);hook=[0];optimizer.register_step_post_hook(lambda *_:hook.__setitem__(0,hook[0]+1))
        binding=digest(approval);coefficient=approval['coefficient'];train=[e['row'] for e in entries]
        for epoch in range(5):
            epoch_losses=[]
            for gi,group in enumerate(grouped_epoch(train,epoch)):
                guard();scope.train_mode();optimizer.zero_grad(set_to_none=True);values=[]
                if [r['image_id'] for r in group]!=context['orders'][epoch][gi]:raise ValueError('Deterministic data order changed')
                for row in group:
                    entry=byid[row['image_id']];teacher=load_teacher(refs[row['image_id']],row,contract);tz,tb=teacher.tensors('cuda');z,b=forward(row)
                    loss,parts,stats=objective(z,b,row,tz,tb,entry,coefficient);monitor=measured(parts['positive'],parts['distillation'])
                    scaler.scale(loss/len(group)).backward();values.append(dict(image_id=row['image_id'],source_group=row['identity_group'],stats=stats,gradients=monitor));epoch_losses.append(stats);del loss,parts,z,b,tz,tb
                scaler.unscale_(optimizer);norms=scope.gradients(optimizer);torch.nn.utils.clip_grad_norm_([scope.params[n] for n in scope.names],1.,error_if_nonfinite=True)
                if state['updates']>=75:raise RuntimeError('Update cap before call')
                out.json('updates/'+str(state['updates']+1).zfill(3)+'.intent.json',dict(state,epoch=epoch+1,source_order=[r['image_id'] for r in group]))
                guard();scaler.step(optimizer)
                if hook[0]!=state['updates']+1:raise RuntimeError('AMP skipped/duplicated update,no automatic retry')
                scaler.update();state['updates']+=1;scope.check(optimizer)
                if any(not torch.isfinite(scope.params[n]).all() for n in TRAIN_KEYS):raise ValueError('Nonfinite updated weights')
                out.json('updates/'+str(state['updates']).zfill(3)+'.json',dict(state,epoch=epoch+1,images=values,gradient_norms=norms,resource=resources()))
            scope.check(optimizer,hash_frozen=True)
            payload=scope.payload(optimizer,state['updates'],binding,epoch=epoch+1,next_group=0,coefficient=coefficient,scaler=scaler.state_dict(),rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            raw=io.BytesIO();torch.save(payload,raw);path=out.write('checkpoints/epoch-'+str(epoch+1)+'.pt',raw.getvalue());last=dict(file=str(path.relative_to(root)),sha256=sha(path),updates=state['updates'])
            reloaded=torch.load(path,map_location='cpu',weights_only=True)
            if any(not torch.equal(reloaded['weights'][n],payload['weights'][n]) for n in HEAD_KEYS):raise ValueError('Checkpoint reload mismatch')
            out.json('checkpoints/epoch-'+str(epoch+1)+'.json',dict(checkpoint=last,immutable_state=scope.immutable_state,losses=epoch_losses,resource=resources()))
        scope.eval_mode();optimizer.zero_grad(set_to_none=True);records=[]
        for i,row in enumerate(dev,1):
            z,b=forward(row,False);scores,labels=z.sigmoid().max(-1);boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).clamp(0,1)
            p=out.npz('predictions/'+str(i).zfill(2)+'.npz',dict(logits=z.cpu().numpy(),square_cxcywh=b.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),boxes=boxes.cpu().numpy()))
            records.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],file=p.name,sha256=sha(p)));del z,b,scores,labels,boxes
        out.json('predictions/raw-complete.json',dict(FP32=True,rows_sha256=context['development_rows_sha256'],records=records))
        scope.check(optimizer,hash_frozen=True);guard()
        if state['updates']!=75 or state['forwards']!=373 or sha(BASE/'model.safetensors')!=BASE_SHA:raise ValueError('Fixed endpoint counters/original preservation')
        out.json('complete.json',dict(status='PASS',mode=mode,**resources(),coefficient=coefficient,checkpoint=last,immutable_state=scope.immutable_state,predictions_saved_before_metrics=True,another_run=False))
    except BaseException as exc:
        failure=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),state=state,seconds=time.monotonic()-began,last_checkpoint=last,automatic_retry=False,frozen_state=None,trainable_state=None)
        if torch.cuda.is_initialized():failure.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        if scope is not None and time.monotonic()-began<limits['seconds']-3:
            try:failure.update(frozen_state=frozen_state(model),trainable_state=named_sha(model,TRAIN_KEYS))
            except BaseException as secondary:failure['fingerprint_unavailable']=str(secondary)
        try:out.json('failure.json',failure)
        except BaseException as secondary:exc.add_note('Failure receipt error: '+str(secondary))
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['training','gradient_smoke'],required=True);parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if args.authorization is None:raise PermissionError('GPU execution not authorized;no approval file supplied')
    run(read(args.authorization),args.mode)

if __name__=='__main__':main()
