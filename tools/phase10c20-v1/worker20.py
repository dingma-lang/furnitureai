"""Exactly one checkpoint28 recovery. No CPU metrics or restart fallback."""
import random, resource, traceback, signal
import numpy as np
from common20 import *


def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cfg,prepared,train,dev,registry=inputs20()
    emit20(RUN20/'executed-once.json',dict(binding_sha256=sha(RUN20/'binding.json'),automatic_retry=False))
    begin=read(RUN20/'launch-once.json')['monotonic_start']
    state=dict(updates=28,optimizer_hook_calls=28,forwards=0,training_VJPs=0,probe_VJPs=0,peak_device_bytes=0,phase='initializing')
    requested=threading.Event();stop=threading.Event();violations=[];thread=None;torch=None;model=None;initial_frozen=None;last=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:requested.set())
    def telemetry(kind):
        d=dict(state,kind=kind,seconds=time.monotonic()-begin)
        if torch is not None and torch.cuda.is_initialized():
            d.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        emit20(RUN20/'telemetry'/('event-'+uuid.uuid4().hex+'.json'),d)
        return d
    from telemetry20 import RuntimeTelemetry
    from telemetry19 import model_fingerprint
    runtime=RuntimeTelemetry(context=dict(experiment='C20-recovery',updates_committed=28,next_update=29),model_provider=lambda:model_fingerprint(model) if model is not None else (_ for _ in ()).throw(RuntimeError('Model not initialized')),optimizer_provider=lambda:dict(updates=state['updates'],optimizer_hook_calls=state['optimizer_hook_calls'],last_checkpoint=last),resource_provider=lambda:dict(state,elapsed_seconds=time.monotonic()-begin))
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from expanded import configure_heads,HEAD_PARAMETERS,from_square,cxcywh_to_xyxy
        from common8 import image_path,PROMPTS,parameter_hash,grouped_epoch
        from balanced16 import branches,Accumulator,Policy,CLASS_KEYS
        from measure17 import scaled_gradients,info,l2
        from update18 import assign_scaled
        from guard19 import compose_checked
        from restore20 import restore_state,restore_recorded_rng,numpy_state
        from masks6 import validate_bundle,authorized
        from common6 import candidates
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required;no retry/fallback')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((MAX_VRAM17-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def guard():
            if time.monotonic()-begin>=510 or requested.is_set() or violations or (RUN20/'STOP').exists():raise RuntimeError('Time/STOP/resource boundary: '+str(violations))
            if state['updates']>75 or state['forwards']>270 or state['training_VJPs']>552 or state['probe_VJPs']>144 or state['updates']-28>47:raise RuntimeError('Update/forward/VJP boundary')
            if max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM17 or allocation20()>MAX_OUTPUT20-64*2**20 or cfg['prior_allocated_bytes']+allocation20()>MAX_OUTPUT18:raise RuntimeError('VRAM/storage boundary')
        def monitor():
            last=0
            while not stop.wait(.1):
                try:
                    free,total=torch.cuda.mem_get_info(0);state['peak_device_bytes']=max(state['peak_device_bytes'],total-free)
                    if total-free>MAX_VRAM17:violations.append('Whole-device VRAM exceeded6.5GiB')
                    if time.monotonic()-last>=2:telemetry('heartbeat');last=time.monotonic()
                except Exception as exc:violations.append('Monitor failure: '+str(exc));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start();guard()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda')
        processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False);configure_heads(model)
        params=dict(model.named_parameters());heads={n:p for n,p in params.items() if p.requires_grad};names=list(heads);tensors=list(heads.values());classes=sorted(CLASS_KEYS)
        original=original_heads()
        if set(heads)!=HEAD_KEYS or sum(p.numel() for p in heads.values())!=HEAD_PARAMETERS or any(not torch.equal(heads[n].detach().cpu(),original[n]) for n in heads):raise ValueError('Exact original12-head initialization required')
        initial_frozen=parameter_hash(model,True);initial_heads=parameter_hash(model,False);c8init=read(C8/'B-corrected/initialization.json')
        if initial_heads!=c8init['initial_heads_sha256'] or initial_frozen!=c8init['frozen_sha256']:raise ValueError('C8 initialization/frozen fingerprint mismatch')
        optimizer=torch.optim.AdamW(tensors,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2000)
        # Three manually scaled VJPs still need GradScaler's ordinary lazy state.
        scaler.scale(torch.zeros((),device='cuda'))
        def stepped(*_):state['optimizer_hook_calls']+=1
        optimizer.register_step_post_hook(stepped)
        def scope():
            if {n for n,p in model.named_parameters() if p.requires_grad}!=HEAD_KEYS or {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in heads.values()} or any(p.grad is not None for n,p in model.named_parameters() if n not in HEAD_KEYS):raise ValueError('Trainable/optimizer/frozen scope drift')
        bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};validate_bundle(bundle,prepared['rows'],frozen,observations);authorized(cfg['unchanged_C8_settings']['source_configuration'],bundle,prepared)
        context=read(DATA16/'catalog.json');selected={r['image_id']:[e for e in bundle['records'] if e['image_id']==r['image_id']] for r in train}
        protected={r['image_id']:[p['box'] for p in registry[r['image_id']]['protected']]+[p['annotation']['box'] for p in frozen if p['image_id']==r['image_id'] and observations[p['candidate_id']]['visual_outcome']=='mixed_or_uncertain'] for r in train}
        probe_ids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']];byid={r['image_id']:r for r in train}
        def counter(*_):
            guard()
            if state['forwards']>=270:raise RuntimeError('270-forward recovery boundary before execution')
            state['forwards']+=1
        model.register_forward_pre_hook(counter)
        def forward(row,optimization):
            runtime.update(current_photo_id=row['image_id'],current_source_group_id=row['identity_group'],operation='training_or_probe_forward' if optimization else 'internal_diagnostic_forward',updates_committed=state['updates'],next_update=state['updates']+1)
            path=image_path(row,'optimization' if optimization else 'inner_development',prepared)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image coordinate frame/orientation changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if optimization:
                with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
            else:
                with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
            guard();return out
        def differentiate(z,b,r,M,mode):
            vectors={};scale=scaler.get_scale();losses=[('positive',r['positive_anchor']),('base',r['base_total']),('source',r['source_raw'])]
            for index,(name,loss) in enumerate(losses):
                guard();key='training_VJPs' if mode=='training' else 'probe_VJPs';limit=552 if mode=='training' else 144
                if state[key]>=limit:raise RuntimeError('VJP bound before execution')
                state[key]+=1;g=scaled_gradients(loss,z,b,tensors,M,scale,retain_graph=index!=2);gz,gb=g[:2];gp=dict(zip(names,g[2:]))
                if name!='base' and (gb is not None or any(gp[k] is not None for k in names if k.startswith('box_head.'))):raise ValueError('Class/source reached box gradients')
                if any(gp[k] is None for k in classes) or (name=='base' and any(gp[k] is None for k in names)):raise ValueError('Missing required head gradient')
                if name=='source' and (gz is None or (gz[~r['masks']['added']]!=0).any()):raise ValueError('Negative gradient reached positive/ignored class-query terms')
                vectors[name]={k:gp[k] for k in (names if name=='base' else classes)}
                del g,gz,gb,gp
            return vectors
        def observations_for(row,z,r,v,M):
            scores=z.detach()[r['masks']['added'][:,1],1].double().sigmoid().cpu().tolist()
            return dict(image_id=row['image_id'],source_group=row['identity_group'],source_sha256=row['sha256'],positive_boxes=r['stats']['positive_boxes'],positive_loss=float(r['positive_anchor'].detach()),native_negative_loss=float(r['native_negative'].detach()),base_loss=float(r['base_total'].detach()),box_loss=float(r['box'].detach()),source_loss=float(r['source_raw'].detach()),legacy_source_focal_loss=float(r['existing_source_focal'].detach()),source_scores=scores,source_terms=len(scores),region_query_counts=r['stats']['region_query_counts'],protected_violations=r['stats']['protected_violations'],ignored_terms=int((~r['masks']['mask']).sum()),normalization=r['normalization'],positive_gradient=info({k:g/scaler.get_scale() for k,g in v['positive'].items()}),source_gradient=info({k:g/scaler.get_scale() for k,g in v['source'].items()}),accumulation_group_size=M,mean_logits_by_category=z.detach().double().mean(0).cpu().tolist())
        def parameter_diagnostics(epoch):
            rec=dict(epoch=epoch,updates=state['updates'],head_sha256=parameter_hash(model,False),frozen_sha256=parameter_hash(model,True),parameters={})
            for k in classes:
                cur=heads[k].detach().cpu();delta=(cur-original[k]).double();rec['parameters'][k]=dict(delta_L2=float(delta.norm()),delta_max_abs=float(delta.abs().max()),current_L2=float(cur.double().norm()))
                if 'dense0' not in k:rec['parameters'][k]['values']=cur.tolist()
            if rec['frozen_sha256']!=initial_frozen:raise ValueError('Frozen parameters changed')
            emit20(RUN20/('parameters-epoch-'+str(epoch)+'.json'),rec)
        def snapshot(epoch,next_group):
            payload=dict(binding_sha256=sha(RUN20/'binding.json'),arm='C18-source-balanced',updates=state['updates'],epoch=epoch,next_group=next_group,frozen_sha256=initial_frozen,weights={n:p.detach().cpu().clone() for n,p in heads.items()},optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=numpy_state()))
            raw=io.BytesIO();torch.save(payload,raw);p=RUN20/'checkpoints'/('update-'+str(state['updates']).zfill(3)+'-'+uuid.uuid4().hex[:8]+'.pt');
            if len(raw.getvalue())>20000000:raise RuntimeError('20MB checkpoint size limit')
            if len(list((RUN20/'checkpoints').glob('*.pt')))>=47:raise RuntimeError('47 new-checkpoint limit')
            atomic20(p,raw.getvalue());saved=torch.load(p,map_location='cpu',weights_only=True)
            if saved['updates']!=state['updates'] or saved['binding_sha256']!=payload['binding_sha256'] or set(saved['weights'])!=HEAD_KEYS or any(not torch.equal(saved['weights'][k],payload['weights'][k]) for k in HEAD_KEYS) or any(int(v['step'])!=state['updates'] for v in saved['optimizer']['state'].values()):raise ValueError('Checkpoint reload integrity failed')
            rec=dict(file=str(p.relative_to(RUN20)),sha256=sha(p),bytes=p.stat().st_size,updates=state['updates'],epoch=epoch)
            emit20(p.with_suffix('.receipt.json'),rec);return rec
        def probe(epoch):
            runtime.update(operation='training_only_probe',epoch_zero_based=epoch-1,accumulation_group_zero_based=None,photo_ids=probe_ids,source_group_ids=[byid[i]['identity_group'] for i in probe_ids])
            configure_heads(model);scope();optimizer.zero_grad(set_to_none=True);before=parameter_hash(model,False);folder=RUN20/'probes'/('epoch-'+str(epoch));folder.mkdir();records=[]
            for index,ident in enumerate(probe_ids,1):
                state['phase']='probe-'+str(epoch)+'-'+str(index);row=byid[ident];out=forward(row,True);z=out.logits[0].float();b=out.pred_boxes[0].float();r=branches(z,b,row,selected[ident],protected[ident],context,Policy(enabled=True));v=differentiate(z,b,r,1,'probe');rec=observations_for(row,z,r,v,1)
                arrays=dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy());p=folder/(str(index).zfill(2)+'.npz')
                if epoch==0:
                    a=load_npz(C8/'B-corrected/probes/epoch-0'/p.name)
                    if any(not np.array_equal(a[k],arrays[k]) for k in arrays):raise ValueError('Original C8 probe equality failed')
                write_npz20(p,arrays);rec.update(file=p.name,sha256=sha(p),epoch=epoch);emit20(p.with_suffix('.json'),rec);records.append(rec)
                del out,z,b,r,v,rec,arrays
            if parameter_hash(model,False)!=before or any(p.grad is not None for p in heads.values()):raise ValueError('Probe changed weights or accumulated .grad')
            emit20(folder/'complete.json',dict(epoch=epoch,updates=state['updates'],records=records,weights_unchanged=True));print('PROBE',epoch,'source terms',sum(r['source_terms'] for r in records),flush=True)
        def evaluate(epoch):
            runtime.update(operation='internal_diagnostic_evaluation',epoch_zero_based=epoch-1,accumulation_group_zero_based=None,photo_ids=[r['image_id'] for r in dev],source_group_ids=[r['identity_group'] for r in dev])
            model.eval();scope();folder=RUN20/'evaluation'/('epoch-'+str(epoch));folder.mkdir();records=[]
            for index,row in enumerate(dev,1):
                state['phase']='evaluation-'+str(epoch)+'-'+str(index);out=forward(row,False);z=out.logits[0].float();b=out.pred_boxes[0].float();scores,labels=z.sigmoid().max(-1);boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).clamp(0,1)
                arrays=dict(logits=z.cpu().numpy(),square_cxcywh=b.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),boxes=boxes.cpu().numpy());p=folder/(str(index).zfill(2)+'.npz');write_npz20(p,arrays);rec=dict(index=index,image_id=row['image_id'],source_sha256=row['sha256'],file=p.name,sha256=sha(p));emit20(p.with_suffix('.receipt.json'),rec);records.append(rec);del out,z,b,scores,labels,boxes,arrays
            emit20(folder/'raw-complete.json',dict(epoch=epoch,updates=state['updates'],rows_sha256=digest(dev),records=records,FP32=True));configure_heads(model);scope()
        emit20(RUN20/'initialization.json',dict(original_checkpoint_sha256=cfg['initialization']['SHA256'],initial_heads_sha256=initial_heads,frozen_sha256=initial_frozen,trainable_keys=names,parameters=HEAD_PARAMETERS,calibration_trainable=True,seed=6106,device=torch.cuda.get_device_name(0),torch_version=torch.__version__))
        payload=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True)
        if sha(CHECKPOINT28)!=cfg['source_checkpoint_sha256']:raise ValueError('Checkpoint28 changed before restore')
        restore_state(model,optimizer,scaler,payload);scope()
        if parameter_hash(model,True)!=payload['frozen_sha256']:raise ValueError('Restored frozen backbone differs')
        restored_heads=parameter_hash(model,False)
        first=dict(file=cfg['source_checkpoint'],sha256=cfg['source_checkpoint_sha256'],updates=28);last=first
        parameter_diagnostics('28-restored')
        # No initial probe or fresh optimizer step. Restore recorded RNG LAST.
        rng_receipt=restore_recorded_rng(payload,use_cuda=True)
        emit20(RUN20/'restoration.json',dict(checkpoint_sha256=cfg['source_checkpoint_sha256'],updates=28,epoch_zero_based=payload['epoch'],next_group_zero_based=payload['next_group'],restored_heads_sha256=restored_heads,frozen_sha256=initial_frozen,scaler=scaler.state_dict(),rng=rng_receipt,parameter_order=names,optimizer_states=len(optimizer.state),no_model_forward_before_restore=state['forwards']==0,bitwise_uninterrupted_equivalence_not_claimed=True))
        for epoch in range(payload['epoch'],5):
            creator_vectors={};groups=grouped_epoch(train,epoch)
            for group_index,group in enumerate(groups):
                if epoch==payload['epoch'] and group_index<payload['next_group']:continue
                runtime.update(epoch_zero_based=epoch,accumulation_group_zero_based=group_index,photo_ids=[r['image_id'] for r in group],source_group_ids=[r['identity_group'] for r in group],updates_committed=state['updates'],next_update=state['updates']+1)
                guard();configure_heads(model);optimizer.zero_grad(set_to_none=True);scope();M=len(group);scale=scaler.get_scale();acc=Accumulator(M,scale);values=[];pending=[]
                runtime.accumulator=acc
                if [r['image_id'] for r in group]!=cfg['orders'][epoch][group_index]:raise ValueError('Bound data order changed')
                for row in group:
                    state['phase']='training-'+str(epoch+1)+'-'+str(group_index+1);out=forward(row,True);z=out.logits[0].float();b=out.pred_boxes[0].float();r=branches(z,b,row,selected[row['image_id']],protected[row['image_id']],context,Policy(enabled=True));v=differentiate(z,b,r,M,'training');acc.add(v['base'],v['positive'],v['source'],r['stats']['positive_boxes']);values.append(observations_for(row,z,r,v,M));pending.append((row['identity_group'],{k:g.detach().cpu()/scale for k,g in v['source'].items()}));del out,z,b,r,v
                scope();runtime.boundary(acc);composed,metric=compose_checked(acc)
                for ident,source in pending:
                    if ident not in creator_vectors:creator_vectors[ident]={k:torch.zeros_like(v) for k,v in source.items()}
                    for k,v in source.items():creator_vectors[ident][k].add_(v*metric['cap']['scale'])
                assign_scaled(heads,composed,scale);scaler.unscale_(optimizer);scope()
                if any(not torch.equal(heads[k].grad,composed[k]) for k in heads):raise ValueError('AMP compose/assign/unscale changed gradients')
                preclip_norm=l2(p.grad for p in heads.values());torch.nn.utils.clip_grad_norm_(tensors,1.,error_if_nonfinite=True)
                if state['updates']>=75:raise RuntimeError('75-update cap before step')
                guard();next_update=state['updates']+1;emit20(RUN20/'updates'/('update-'+str(next_update).zfill(3)+'.intent.json'),dict(update=next_update,epoch=epoch+1,image_ids=[r['image_id'] for r in group],binding_sha256=sha(RUN20/'binding.json')))
                scaler.step(optimizer)
                if state['optimizer_hook_calls']!=next_update:raise RuntimeError('Optimizer skipped/duplicated;no automatic replay')
                scaler.update();state['updates']=next_update
                if any(not torch.isfinite(p).all() for p in heads.values()) or any(int(v['step'])!=state['updates'] for v in optimizer.state.values()):raise ValueError('Updated parameter/state integrity failure')
                last=snapshot(epoch,group_index+1);emit20(RUN20/'updates'/('update-'+str(next_update).zfill(3)+'.json'),dict(update=next_update,epoch=epoch+1,source_order=[r['image_id'] for r in group],images=values,metrics=metric,preclip_total_norm=preclip_norm,scaler_scale=scaler.get_scale(),checkpoint=last,resource=telemetry('update_committed')))
                optimizer.zero_grad(set_to_none=True);scope();runtime.accumulator=None;del acc,composed,metric,pending,values
                print('UPDATE',next_update,'/75 epoch',epoch+1,'elapsed',round(time.monotonic()-begin,2),flush=True)
            parameter_diagnostics(epoch+1)
            emit20(RUN20/('sources-epoch-'+str(epoch+1)+'.json'),{k:info(v) for k,v in sorted(creator_vectors.items())});del creator_vectors
            if epoch+1 in (3,5):probe(epoch+1);evaluate(epoch+1)
        scope();guard()
        if state['updates']!=75 or state['optimizer_hook_calls']!=75 or state['forwards']!=270 or state['training_VJPs']!=552 or state['probe_VJPs']!=144 or parameter_hash(model,True)!=initial_frozen:raise ValueError('Completion counts or frozen scope differ')
        if sha(BASE/'model.safetensors')!=cfg['initialization']['SHA256']:raise ValueError('Original file changed')
        state['phase']='complete';stop.set();thread.join(timeout=2);r=telemetry('complete')
        emit20(RUN20/'complete.json',dict(status='PASS',**r,additional_updates=state['updates']-28,cumulative_forwards=state['forwards']+182,epochs=5,initial_heads_sha256=initial_heads,final_heads_sha256=parameter_hash(model,False),frozen_sha256=initial_frozen,initial_checkpoint=first,final_checkpoint=last,nonfinite_events=0,original_model_unchanged=True,calibration_trainable=True,predictions_saved_before_metrics=True,another_arm_executed=False))
        print('C20 RECOVERY COMPLETE75cumulativeupdates;no follow-on GPU experiment',flush=True)
    except BaseException as exc:
        stop.set()
        try:runtime.save_failure(exc)
        except BaseException as telemetry_error:exc.add_note('Failure detail persistence error: '+str(telemetry_error))
        try:emit20(RUN20/'failure.json',dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),state=telemetry('failure'),automatic_retry=False,original_frozen_fingerprint=initial_frozen,checkpoint_recovery_requires_review=True))
        except BaseException:pass
        raise
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=2)


if __name__=='__main__':run()
