"""One original-initialized C8-scope smoke;fiveVJPs/photo,no optimizer."""
import io,random,signal,threading,traceback
import numpy as np
from common17 import *
from measure17 import info,l2,cos,scaled_gradients,group_metrics,finish_accumulator
from gpu15 import state_fingerprint


def run():
    cfg,prepared,groups,registry=inputs17();emit17(RUN17/'executed-once.json',dict(config_sha256=sha(RUN17/'authorized-config.json'),automatic_retry=False))
    begin=read(RUN17/'launch-once.json')['monotonic_start'];state=dict(forwards=0,gradient_calculations=0,completed_photos=0,completed_groups=0,last_verified_model_sha256=None,last_verified_at_seconds=None,peak_observed_device_bytes=0,phase='initializing')
    stop=threading.Event();requested=threading.Event();violations=[];lock=threading.Lock();model=None;torch=None;before=None;thread=None;receipts=[];group_receipts=[];creators={}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:requested.set())
    def pulse(kind,terminal=False):
        with lock:
            rec=dict(state,kind=kind,elapsed_seconds=time.monotonic()-begin,optimizer_steps=0,model_updates=0,terminal=terminal)
            if torch is not None and torch.cuda.is_initialized():
                try:rec.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                except Exception as exc:rec['memory_read_error']=str(exc)
            emit17(RUN17/'telemetry'/('event-'+uuid.uuid4().hex+'.json'),rec);return rec
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from expanded import configure_heads,HEAD_PARAMETERS
        from common8 import image_path,PROMPTS,parameter_hash
        from balanced16 import branches,Accumulator,Policy,CLASS_KEYS
        from common6 import candidates
        from masks6 import validate_bundle
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required;no fallback or retry')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((MAX_VRAM17-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def guard():
            if time.monotonic()-begin>=175 or requested.is_set() or violations or (RUN17/'STOP').exists():raise RuntimeError('Time/STOP/device boundary: '+str(violations))
            if state['forwards']>59 or state['gradient_calculations']>295 or max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM17 or allocation17()>MAX_OUTPUT17-4*2**20:raise RuntimeError('Forward/gradient/VRAM/output boundary')
        def monitor():
            last=0.
            while not stop.wait(.1):
                try:
                    free,total=torch.cuda.mem_get_info(0);state['peak_observed_device_bytes']=max(state['peak_observed_device_bytes'],total-free)
                    if total-free>MAX_VRAM17:violations.append('Observed whole-device VRAM boundary')
                    if time.monotonic()-last>=1:pulse('heartbeat');last=time.monotonic()
                except Exception as exc:violations.append('Resource monitor: '+str(exc));return
        pulse('GPU_initialized');thread=threading.Thread(target=monitor,daemon=True);thread.start();guard()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False);configure_heads(model)
        params=dict(model.named_parameters());names=[n for n in params if n in HEAD_KEYS];tensors=[params[n] for n in names];original=original_heads()
        if any(not torch.equal(params[n].detach().cpu(),original[n]) for n in HEAD_KEYS):raise ValueError('Not original pretrained initialization')
        initial_head_hash=parameter_hash(model,False);old_init=read(C8/'B-corrected/initialization.json')
        if initial_head_hash!=old_init['initial_heads_sha256']:raise ValueError('C8 original initialization fingerprint differs')
        def scope():
            if {n for n,p in model.named_parameters() if p.requires_grad}!=HEAD_KEYS or sum(p.numel() for p in model.parameters() if p.requires_grad)!=HEAD_PARAMETERS or any(p.grad is not None for p in model.parameters()):raise ValueError('Trainable/frozen gradient scope changed')
        scope();before=state_fingerprint(model);state.update(last_verified_model_sha256=before['sha256'],last_verified_at_seconds=time.monotonic()-begin)
        emit17(RUN17/'initialization.json',dict(model_state=before,initial_heads_sha256=initial_head_hash,original_checkpoint_sha256=cfg['original_checkpoint_sha256'],trainable_keys=names,calibration_trainable=True,trainable_parameters=HEAD_PARAMETERS,optimizer_constructed=False,adapted_checkpoint_loaded=False));pulse('original_model_verified')
        frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,obs);context=read(DATA16/'catalog.json');scale=128.;clskeys=sorted(CLASS_KEYS);branch_names=['positive','base','A_focal','B_BCE','C_pre_cap'];probe_ids=[r['image_id'] for r in read(SOURCE/'proposed-smoke-photos.json')['photos']]
        def count(*_):
            guard()
            if state['forwards']>=59:raise RuntimeError('59forward boundary before execution')
            state['forwards']+=1
        hook=model.register_forward_pre_hook(count)
        def cpu_add(creator,method,mapping):
            if creator not in creators:creators[creator]={}
            if method not in creators[creator]:creators[creator][method]={k:torch.zeros_like(mapping[k],device='cpu') for k in clskeys}
            for k in clskeys:creators[creator][method][k].add_(mapping[k].detach().cpu())
        emit17(RUN17/'resources-start.json',dict(device=torch.cuda.get_device_name(0),device_bytes=torch.cuda.get_device_properties(0).total_memory,limits=cfg['limits'],scale=128.,precision='C8 FP16forward,FP32loss;branch loss divided by actual accumulation size before scale128;accumulate scaled buffers then unscale once'))
        photo_index=0
        for gi,group in enumerate(groups,1):
            guard();scope();M=len(group);acc=Accumulator(M,scale);extras={n:{k:torch.zeros_like(params[k]) for k in clskeys} for n in ('A_focal','B_BCE')};pending=[];positive_boxes=chair_positive=source_terms=0
            for row in group:
                photo_index+=1;state['phase']='photo-'+str(photo_index);ident=row['image_id'];selected=[e for e in bundle['records'] if e['image_id']==ident];protected=[p['box'] for p in registry[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain'];guard();scope()
                emit17(RUN17/'photos'/(str(photo_index).zfill(3)+'.intent.json'),dict(index=photo_index,accumulation_group=gi,image_id=ident,next_forward=state['forwards']+1))
                with Image.open(image_path(row,'optimization',prepared)) as im:
                    if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image frame/orientation changed')
                    batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
                with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
                del batch;z=out.logits[0].float();b=out.pred_boxes[0].float();guard()
                if z.shape!=(3600,7) or b.shape!=(3600,4):raise ValueError('Actual query shape changed')
                cached_match=None
                if ident in probe_ids:
                    a=load_npz(C8/'B-corrected/probes/epoch-0'/(str(probe_ids.index(ident)+1).zfill(2)+'.npz'));cached_match=bool(np.array_equal(a['logits'],z.detach().cpu().numpy()) and np.array_equal(a['square_cxcywh'],b.detach().cpu().numpy()))
                    if not cached_match:raise ValueError('Original-init output differs from preserved C8 probe')
                r=branches(z,b,row,selected,protected,context,Policy(enabled=True));losses=dict(positive=r['positive_anchor'],base=r['base_total'],A_focal=r['existing_source_focal'],B_BCE=r['unweighted_source_BCE'],C_pre_cap=r['source_raw']);arrays=dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy(),target=r['masks']['target'].cpu().numpy().astype(np.uint8),legacy=r['masks']['legacy'].cpu().numpy().astype(np.uint8),mask=r['masks']['mask'].cpu().numpy().astype(np.uint8),added=r['masks']['added'].cpu().numpy().astype(np.uint8),weights=r['weights'].cpu().numpy());vectors={};stats={}
                for bi,name in enumerate(branch_names):
                    guard()
                    if state['gradient_calculations']>=295:raise RuntimeError('295gradient boundary before calculation')
                    state['gradient_calculations']+=1;g=scaled_gradients(losses[name],z,b,tensors,M,scale,retain_graph=bi!=4);gz,gb=g[:2];gp=dict(zip(names,g[2:]))
                    if name!='base' and (gb is not None or any(gp[k] is not None for k in names if k.startswith('box_head.'))):raise ValueError('Classification/source gradient reaches box head')
                    if any(gp[k] is None for k in clskeys) or (name=='base' and any(gp[k] is None for k in names)):raise ValueError('Missing permitted head gradient')
                    if name in ('A_focal','B_BCE','C_pre_cap') and (gz is None or (gz[~r['masks']['added']]!=0).any()):raise ValueError('Source gradient reached ignored/positive output')
                    actual={k:gp[k]/scale for k in clskeys};vectors[name]={k:gp[k] for k in (names if name=='base' else clskeys)};stats[name]=dict(loss=float(losses[name].detach()),loss_divided_by_group_size=float((losses[name].detach()/M)),output_logit_L1=float((gz.double()/scale).abs().sum()) if gz is not None else 0.,parameter_classification=info(actual),box_norm=l2(gp[k]/scale for k in names if k.startswith('box_head.') and gp[k] is not None))
                    arrays['output_gradient_'+name]=(gz/scale).cpu().numpy() if gz is not None else np.zeros((3600,7),np.float32);cpu_add(row['identity_group'],name,actual)
                    del g,gz,gb,gp,actual
                acc.add(vectors['base'],vectors['positive'],vectors['C_pre_cap'],r['stats']['positive_boxes'])
                for name in extras:
                    for k in clskeys:extras[name][k].add_(vectors[name][k])
                scope()
                if parameter_hash(model,False)!=initial_head_hash:raise ValueError('Head parameters changed during gradient diagnostic')
                positive_boxes+=r['stats']['positive_boxes'];chair_positive+=sum(not a['ignore'] and a['label']=='chair' for a in row['annotations']);source_terms+=r['stats']['added_active_Chair_terms'];source_scores=z.detach()[r['masks']['added'][:,1],1].double().sigmoid().cpu().tolist();rec=dict(index=photo_index,image_id=ident,identity_group=row['identity_group'],source_sha256=row['sha256'],group=gi,group_size=M,heads_unchanged=True,all_parameter_grads_none=True,original_C8_cached_outputs_bit_identical=cached_match,counts=dict(positive=r['stats']['positive_boxes'],Chair_positive=sum(not a['ignore'] and a['label']=='chair' for a in row['annotations']),source_terms=r['stats']['added_active_Chair_terms'],ignored=int((~r['masks']['mask']).sum()),protected_violations=r['stats']['protected_violations']),losses_and_gradients=stats,source_scores=source_scores,region_query_counts=r['stats']['region_query_counts'],normalization=r['normalization'],same_targets_masks_for_all_methods=True,box_loss=r['stats']['box_loss'],normalization_note='Parameter/output derivatives include1/actual accumulation group size;loss values retained before group normalization')
                raw=io.BytesIO();np.savez_compressed(raw,**arrays)
                if len(raw.getvalue())>1_000_000:raise ValueError('Per-photo diagnostic size bound')
                fp=RUN17/'photos'/(str(photo_index).zfill(3)+'.npz');atomic17(fp,raw.getvalue());rec.update(raw_file=str(fp.relative_to(RUN17)),raw_sha256=sha(fp));jp=fp.with_suffix('.json');emit17(jp,rec);receipts.append(dict(index=photo_index,image_id=ident,file=str(jp.relative_to(RUN17)),sha256=sha(jp)))
                pending.append(dict(index=photo_index,identity_group=row['identity_group'],source_scaled={k:vectors['C_pre_cap'][k] for k in clskeys}));state['completed_photos']+=1
                del out,z,b,r,losses,arrays,vectors,stats,raw,source_scores,rec;guard()
            p_before={k:v.clone() for k,v in acc.positive.items()};boxes_before={k:v.clone() for k,v in acc.base.items() if k.startswith('box_head.')};composed,cap=finish_accumulator(acc)
            if any(not torch.equal(acc.positive[k],v) for k,v in p_before.items()) or any(not torch.equal(acc.base[k],v) for k,v in boxes_before.items()):raise ValueError('Cap altered positive or box accumulation buffers')
            base={k:v/scale for k,v in acc.base.items()};p={k:v/scale for k,v in acc.positive.items()};n={k:v/scale for k,v in acc.negative.items()};f={k:v/scale for k,v in extras['A_focal'].items()};be={k:v/scale for k,v in extras['B_BCE'].items()};metrics=group_metrics(base,p,f,be,n,composed,cap)
            photo_contributions=[]
            for item in pending:
                post={k:item['source_scaled'][k]/scale*cap['scale'] for k in clskeys};cpu_add(item['identity_group'],'C_post_cap',post);photo_contributions.append(dict(index=item['index'],identity_group=item['identity_group'],post_cap_negative=info(post),scale=cap['scale']))
            scope();fingerprint=state_fingerprint(model)
            if fingerprint!=before:raise ValueError('Full parameter/buffer state changed')
            state.update(completed_groups=gi,last_verified_model_sha256=fingerprint['sha256'],last_verified_at_seconds=time.monotonic()-begin,phase='group-'+str(gi)+'-verified');resource=pulse('group_complete')
            rec=dict(group=gi,photo_indices=[x['index'] for x in pending],image_ids=[r['image_id'] for r in group],source_groups=[r['identity_group'] for r in group],size=M,positive_boxes=positive_boxes,Chair_positive_boxes=chair_positive,source_terms=source_terms,metrics=metrics,photo_post_cap_contributions=photo_contributions,model_state_sha256=fingerprint['sha256'],parameters_and_buffers_unchanged=True,resource=resource);gp=RUN17/'groups'/(str(gi).zfill(2)+'.json');emit17(gp,rec);group_receipts.append(dict(group=gi,file=str(gp.relative_to(RUN17)),sha256=sha(gp)));print('GROUP',gi,'/15;photos',state['completed_photos'],'/59;forwards',state['forwards'],';VJPs',state['gradient_calculations'],';cap',round(cap['scale'],6),flush=True)
            del acc,extras,pending,p_before,boxes_before,composed,base,p,n,f,be,metrics,fingerprint,photo_contributions,rec;guard()
        hook.remove();torch.cuda.synchronize();scope();after=state_fingerprint(model)
        if after!=before or state['forwards']!=59 or state['gradient_calculations']!=295 or state['completed_groups']!=15:raise ValueError('Final counter/state mismatch')
        creator_reports=[];global_vectors={name:{k:torch.zeros_like(params[k],device='cpu') for k in clskeys} for name in ('positive','base','A_focal','B_BCE','C_pre_cap','C_post_cap')}
        for creator,methods in sorted(creators.items()):
            for name,values in methods.items():
                for k in clskeys:global_vectors[name][k].add_(values[k]/15)
            creator_reports.append(dict(identity_group=creator,positive=info(methods['positive']),base=info(methods['base']),negatives={name:info(methods[name],methods['positive']) for name in ('A_focal','B_BCE','C_pre_cap','C_post_cap')},aggregation='Actual sum of matching creator VJP buffers with C8 per-group normalization;not reconstructed from norms'))
        global_report=dict(positive=info(global_vectors['positive']),negative={name:info(global_vectors[name],global_vectors['positive']) for name in ('A_focal','B_BCE','C_pre_cap','C_post_cap')},combined={name:dict(info({k:global_vectors['base'][k]+global_vectors[name][k] for k in clskeys}),cosine_positive=cos({k:global_vectors['base'][k]+global_vectors[name][k] for k in clskeys},global_vectors['positive'])) for name in ('A_focal','B_BCE','C_pre_cap','C_post_cap')},scope='Arithmetic mean of15 fixed-weight,update-normalized group vectors;cap contract is per group,not bound relative to this potentially cancelling global mean')
        guard();emit17(RUN17/'source-groups.json',dict(groups=creator_reports,global_mean=global_report));state.update(last_verified_model_sha256=after['sha256'],last_verified_at_seconds=time.monotonic()-begin,phase='complete');stop.set();thread.join(timeout=1);resource=pulse('complete',terminal=True)
        emit17(RUN17/'complete.json',dict(status='PASS',forwards=59,gradient_calculations=295,photos=59,known_source_groups=len(creators),accumulation_groups=15,records=receipts,groups=group_receipts,resource=resource,model_state_before=before['sha256'],model_state_after=after['sha256'],all_parameters_and_buffers_unchanged=True,optimizer_constructed=False,optimizer_steps=0,model_updates=0,automatic_retry=False,training_authorized=False));print('C17 SINGLE ZERO-UPDATE SMOKE COMPLETE',flush=True)
    except BaseException as exc:
        error=str(exc);trace=traceback.format_exc();terminal={}
        try:
            if model is not None and before is not None:
                after=state_fingerprint(model);state.update(last_verified_model_sha256=after['sha256'],last_verified_at_seconds=time.monotonic()-begin);terminal=dict(model_state_before=before['sha256'],model_state_after=after['sha256'],unchanged=before==after)
        except BaseException as secondary:terminal['fingerprint_error']=str(secondary)
        try:terminal['resource']=pulse('failure',terminal=True)
        except BaseException as secondary:terminal['telemetry_error']=str(secondary)
        emit17(RUN17/'failure.json',dict(error=error,traceback=trace,**state,terminal=terminal,optimizer_steps=0,automatic_retry=False));raise
    finally:stop.set()


if __name__=='__main__':run()
