"""Prepared two-fit CUDA queue. Explicit future authorization is required before any runtime setup."""
import argparse
import importlib.metadata
import random
import signal
import subprocess
import time
import warnings
from p10 import *
from state import Ledger,StopRun,lease
from protocol import FIT_IDS,check_bounds
from tracking import record,score

STOP=False
def grouped_epoch(rows,seed,epoch):
    order=list(rows);random.Random(seed+epoch).shuffle(order)
    return [order[i:i+4] for i in range(0,len(order),4)]
def request_stop(*_):
    global STOP
    STOP=True

def resource_limits(allocated,peak,free_volumes,stored):
    if max(allocated,peak)>int(6.5*2**30):raise StopRun('6.5GiB allocated VRAM target exceeded')
    if any(n<5*2**30 for n in free_volumes):raise StopRun('Maintain5GiB free on WSL and Windows volume')
    if stored+64*2**20>8*2**30:raise StopRun('Queue snapshot-storage budget')

def validate(authorization):
    # This executes before CUDA/model imports, creation of RUN, or any image access.
    a=read(authorization)
    if a.get('approved') is not True or a.get('model_updates_authorized') is not True or not a.get('approved_by') or not a.get('approval_reference'):
        raise PermissionError('Separate explicit GPU-training approval required')
    manifest=read(RAW/'eligibility-manifest.json');digest=sha(RAW/'eligibility-manifest.json')
    configs={n:read(RAW/'configs'/(n+'.json')) for n in FIT_IDS}
    for c in configs.values():check_bounds(c)
    expected={n:sha(RAW/'configs'/(n+'.json')) for n in FIT_IDS}
    if a.get('manifest_sha256')!=digest or a.get('config_sha256')!=expected or a.get('code_hashes')!=code_hashes():raise ValueError('Authorization data/config/code binding mismatch')
    if a.get('max_updates_per_fit')!=150 or a.get('max_updates_total')!=300 or a.get('max_seconds')!=3600 or a.get('max_epochs')!=5:
        raise ValueError('Approval budget mismatch')
    if (len(manifest['rows']),manifest['eligible_boxes'])!=(78,243):raise ValueError('Approved diagnostic data drift')
    contract=read(OLD/'input-contract.json')
    for name,h in contract['model_files'].items():
        if sha(ROOT/'models/owlv2-original-baseline-20261103'/name)!=h:raise ValueError('Pinned base/processor changed')
    for name,h in expected.items():
        if configs[name]['manifest_sha256']!=digest:raise ValueError('Config data binding changed')
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit source changes before approved training')
    return manifest,configs,dict(authorization_sha256=sha(authorization),dataset_sha256=digest,configs=expected,code=code_hashes(),git_commit=git,
        environment={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow','scipy')})

def parameters_hash(model,frozen=True):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if (not p.requires_grad)==frozen:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def scope(model,optimizer):
    from expanded import HEAD_KEYS,HEAD_PARAMETERS
    heads={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(heads)!=HEAD_KEYS or sum(p.numel() for p in heads.values())!=HEAD_PARAMETERS:raise ValueError('Head scope violation')
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in heads.values()}:raise ValueError('Optimizer includes frozen parameters')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Backbone gradient detected')

def run(authorization,resume=False,smoke_only=False):
    manifest,configs,binding=validate(authorization)
    smoke_path=RUN/'smoke-pass.json'
    if smoke_path.exists():
        smoke=read(smoke_path)
        if smoke['binding']!=binding or sha(RUN/smoke['receipt']['path'])!=smoke['receipt']['sha256']:raise ValueError('GPU smoke binding/receipt changed')
        if smoke_only:print('SKIP COMPLETE GPU SMOKE',flush=True);return
    elif not smoke_only:
        raise PermissionError('A separately approved --smoke-only run must pass before optimizer calls')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection,Owlv2Processor
    from expanded import configure_heads,load_delta,detection_loss
    from internal import predict,evaluate
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if RUN.is_symlink() or RUN.resolve()!=RUN:raise ValueError('Refuse redirected experiment destination')
    if RUN.exists() and not resume:raise FileExistsError('Use explicit --resume; never remove artifacts')
    RUN.mkdir(parents=True,exist_ok=True)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    with lease(RUN/'queue.lock'):
        for fit in FIT_IDS:
            folder=RUN/fit
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json')
                if done['binding']!=binding or sha(RUN/done['receipt']['path'])!=done['receipt']['sha256']:raise ValueError('Completed fit binding/hash changed')
                if sha(RUN/done['best']['receipt']['path'])!=done['best']['receipt']['sha256']:raise ValueError('Completed best checkpoint changed')
                print('SKIP COMPLETE',fit,flush=True);continue
            ledger=Ledger(RUN,binding);ledger.start(fit);source=ledger.resume()
            config=configs[fit];folder.mkdir(exist_ok=True);torch.manual_seed(config['seed']);random.seed(config['seed'])
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            torch.cuda.set_per_process_memory_fraction(.8,0)
            try:torch.cuda.reset_peak_memory_stats(0)
            except RuntimeError as exc:warnings.warn('Optional peak-stat reset unavailable: '+str(exc))
            def guard():
                ledger.tick()
                if STOP or (RUN/'STOP').exists():raise StopRun('Graceful stop requested')
                free=[]
                for mount in (ROOT,Path('/mnt/c')):
                    stat=os.statvfs(mount)
                    free.append(stat.f_bavail*stat.f_frsize)
                resource_limits(torch.cuda.memory_allocated(0),torch.cuda.max_memory_allocated(0),free,
                    sum(p.stat().st_blocks*512 for p in RUN.rglob('*') if p.is_file()))
            guard();model=Owlv2ForObjectDetection.from_pretrained(str(ROOT/'models/owlv2-original-baseline-20261103'),local_files_only=True,use_safetensors=True).to('cuda')
            processor=Owlv2Processor.from_pretrained(str(ROOT/'models/owlv2-original-baseline-20261103'),local_files_only=True)
            configure_heads(model);frozen=parameters_hash(model);initial_heads=parameters_hash(model,frozen=False);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=128);scope(model,optimizer)
            train=[r for r in manifest['rows'] if r['internal_partition']=='optimization'];dev=[r for r in manifest['rows'] if r['internal_partition']=='inner_development']
            refs=RUN/'internal-reference.json'
            def validation():
                preds={};times=[]
                for row in dev:
                    guard();t=time.perf_counter();preds[row['image_id']]=predict(model,processor,row,manifest);torch.cuda.synchronize(0);times.append(time.perf_counter()-t);guard()
                result=evaluate(dev,preds);result.update(seconds_per_photo=sum(times)/len(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                return result
            if not refs.exists():
                originals={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
                base=validation();registry=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json')
                if sha(registry['checkpoint'])!=registry['sha256']:raise ValueError('Phase5C comparator changed')
                payload=torch.load(registry['checkpoint'],map_location='cpu',weights_only=True);load_delta(model,payload);adapted=validation();load_delta(model,dict(weights=originals))
                save(refs,dict(binding=binding,original=base,phase5c=adapted,original_sha256=config['base']['sha256'],adapted_sha256=registry['sha256']))
            reference=read(refs)
            if reference['binding']!=binding:raise ValueError('Reference binding mismatch')
            cursor=epoch=next_group=0;best=None;last_receipt=None;hook_calls=[0]
            optimizer.register_step_post_hook(lambda *_:hook_calls.__setitem__(0,hook_calls[0]+1))
            if source:
                state=torch.load(source,map_location='cpu',weights_only=True)
                if state['binding']!=binding or state['frozen_sha256']!=frozen:raise ValueError('Snapshot binding/frozen mismatch')
                load_delta(model,state);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
                scope(model,optimizer)
                if any(g['lr']!=config['learning_rate'] or g['weight_decay']!=.01 for g in optimizer.param_groups):raise ValueError('Restored optimizer settings changed')
                random.setstate(state['rng']['python']);torch.set_rng_state(state['rng']['cpu']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
                cursor,epoch,next_group,best=state['cursor'],state['epoch'],state['next_group'],state['best'];hook_calls[0]=cursor
                if cursor!=ledger.data['fits'][fit]['applied']:raise ValueError('Snapshot cursor mismatch')
            def persist(ep,group):
                payload=dict(binding=binding,weights={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    cursor=cursor,epoch=ep,next_group=group,best=best,frozen_sha256=frozen,
                    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
                rec=ledger.checkpoint(payload);ledger.committed(rec);return rec
            try:
                for ep in range(epoch,5):
                    groups=grouped_epoch(train,config['seed'],ep)
                    for group in range(next_group,len(groups)):
                        guard();configure_heads(model);optimizer.zero_grad(set_to_none=True);loss_values=[]
                        for row in groups[group]:
                            guard();p=image_path(row,'optimization',manifest)
                            with Image.open(p) as im:batch=processor(text=[config['prompts']],images=im.convert('RGB'),return_tensors='pt')
                            with torch.autocast('cuda',dtype=torch.float16):outputs=model(**{k:v.to('cuda') for k,v in batch.items()})
                            guard();loss,parts=detection_loss(outputs.logits[0],outputs.pred_boxes[0],row)
                            scaler.scale(loss/len(groups[group])).backward();loss_values.append(float(loss.detach()));del outputs,loss,batch;guard()
                        scaler.unscale_(optimizer);scope(model,optimizer)
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError('Missing/nonfinite head gradient; optimizer not called')
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                        if smoke_only:
                            if cursor or parameters_hash(model)!=frozen or parameters_hash(model,frozen=False)!=initial_heads:raise ValueError('Smoke must preserve all weights and apply zero updates')
                            optimizer.zero_grad(set_to_none=True);scaler.update(new_scale=scaler.get_scale());guard()
                            smoke_receipt=persist(0,0)
                            save(smoke_path,dict(binding=binding,receipt=smoke_receipt,actual_optimizer_calls=0,
                                frozen_and_head_weights_unchanged=True,finite_gradients=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(0)))
                            print('GPU SMOKE PASS; zero optimizer calls; separate execution resumes committed cursor0',flush=True);return
                        guard();ledger.reserve(cursor);scaler.step(optimizer)
                        if hook_calls[0]!=cursor+1:raise RuntimeError('Optimizer call was skipped or duplicated; reconciliation required')
                        scaler.update();cursor+=1;last_receipt=persist(ep,group+1)
                        save(folder/('update-'+str(cursor).zfill(4)+'.json'),dict(cursor=cursor,epoch=ep+1,samples=len(groups[group]),loss=sum(loss_values)/len(loss_values),receipt=last_receipt))
                        print('UPDATE',fit,cursor,'epoch',ep+1,flush=True)
                    result=validation();ranking=min(score(result,reference[n]) for n in ('original','phase5c'))
                    save(folder/('validation-'+str(ep+1)+'-'+uuid.uuid4().hex+'.json'),dict(epoch=ep+1,metrics=result,rank=ranking,scope='provisional_weak_label_internal'))
                    if best is None or tuple(ranking)>tuple(best['rank']):best=dict(rank=ranking,epoch=ep+1,receipt=last_receipt or ledger.data['fits'][fit]['receipt'],metrics=result)
                    last_receipt=persist(ep+1,0);next_group=0
                    if parameters_hash(model)!=frozen:raise ValueError('Frozen backbone changed')
                scope(model,optimizer)
                if hook_calls[0]!=cursor or cursor!=75:raise ValueError('Unexpected completed update count')
                if sha(ROOT/'models/owlv2-original-baseline-20261103/model.safetensors')!=config['base']['sha256']:raise ValueError('Original model file changed')
                save(folder/'complete.json',dict(binding=binding,updates=cursor,epochs=5,best=best,receipt=last_receipt,peak_allocated_bytes=torch.cuda.max_memory_allocated(0)))
                record(RUN/'history',dict(experiment_id=fit,dataset_version='phase10b-v1',dataset_sha256=binding['dataset_sha256'],git_commit=binding['git_commit'],
                    checkpoint_sha256=best['receipt']['sha256'],settings=config,metrics=best['metrics'],resources=dict(allocated_vram_bytes=torch.cuda.max_memory_allocated(0),charged_seconds=ledger.data['fits'][fit]['charged_seconds']),
                    decision='investigate' if best['rank'][0] else 'reject',next_experiment='Review per-category matched-recall/localization and label coverage; no automatic deployment',evaluation_scope='provisional_weak_label_internal'))
            except StopRun as exc:
                optimizer.zero_grad(set_to_none=True)
                if parameters_hash(model)!=frozen:raise ValueError('Frozen weights changed during stop')
                save(folder/('stopped-'+uuid.uuid4().hex+'.json'),dict(reason=str(exc),cursor=cursor,previous_committed_checkpoint=ledger.data['fits'][fit]['receipt']))
                print('STOPPED; resume uses last committed cursor only',exc,flush=True);return
            except Exception as exc:
                save(folder/('failure-'+uuid.uuid4().hex+'.json'),dict(error_type=type(exc).__name__,message=str(exc),
                    cursor=cursor,previous_committed_checkpoint=ledger.data['fits'][fit]['receipt'],automatic_retry=False))
                raise
            finally:
                del optimizer,scaler,model;torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();run(args.authorization,args.resume,args.smoke_only)
