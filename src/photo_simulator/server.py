"""Local photo replacement simulator. Isolated projects, offline workers, no production writes."""
from __future__ import annotations
import argparse,base64,copy,hashlib,io,json,mimetypes,os,re,subprocess,sys,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from .compositor import make_mask,extract_asset,composite

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=Path(__file__).resolve().parent/'web'
MAX_BYTES=24*1024*1024
Image.MAX_IMAGE_PIXELS=24_000_000

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def decode_image(value):
    if not isinstance(value,str) or not re.match(r'^data:image/(jpeg|png|webp);base64,',value):raise ValueError('Use JPG, PNG or WebP image data')
    raw=base64.b64decode(value.split(',',1)[1],validate=True)
    if len(raw)>18*1024*1024:raise ValueError('Image exceeds 18 MB limit')
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in ('JPEG','PNG','WEBP'):raise ValueError('Unsupported image content')
        if source.width*source.height>24_000_000:raise ValueError('Image exceeds 24 megapixels')
        source.load();image=ImageOps.exif_transpose(source).convert('RGBA' if 'A' in source.getbands() else 'RGB')
    image.thumbnail((1600,1600),Image.Resampling.LANCZOS)
    if min(image.size)<32:raise ValueError('Image must be at least 32 pixels in each dimension')
    return image,raw

def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

class Store:
    def __init__(self,path,device='auto'):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.pool=ThreadPoolExecutor(max_workers=1);self.jobs={};self.active=set();self.device=device
        self.asset_file=ROOT/'simulator-assets/manifest.json'
    def folder(self,id):
        if not re.fullmatch(r'[a-f0-9]{32}',id):raise ValueError('Invalid simulation ID')
        folder=self.path/id
        if not folder.is_dir():raise FileNotFoundError('Simulation not found')
        return folder
    def read(self,id):return json.loads((self.folder(id)/'project.json').read_text())
    def write(self,p):p['updated_at']=now();atomic(self.folder(p['id'])/'project.json',p);return p
    def listing(self):
        projects=[]
        for p in self.path.glob('*/project.json'):
            try:
                v=json.loads(p.read_text());projects.append({k:v.get(k) for k in ('id','name','updated_at','original_url','latest_result_url')})
            except (ValueError,OSError):continue
        return sorted(projects,key=lambda p:p.get('updated_at',''),reverse=True)
    def assets(self):return json.loads(self.asset_file.read_text())['assets'] if self.asset_file.is_file() else []
    def create(self,payload):
        image,raw=decode_image(payload['image']);id=uuid.uuid4().hex;folder=self.path/id;folder.mkdir()
        (folder/'original-upload.bin').write_bytes(raw);image.convert('RGB').save(folder/'original.png')
        return self.write({'schema_version':1,'id':id,'name':str(payload.get('name') or 'Room simulation')[:160],'created_at':now(),'image_width':image.width,'image_height':image.height,'original_sha256':hashlib.sha256(raw).hexdigest(),'original_url':f'/media/{id}/original.png','detections':[],'analysis':{},'asset':None,'settings':{},'outputs':[],'latest_result_url':None,'mask_url':None,'physical_fit_verified':False})
    def check_idle(self,id):
        if id in self.active:raise ValueError('This simulation has an active job; wait for it to complete')
    def set_asset(self,id,payload):
        self.check_idle(id);p=self.read(id);folder=self.folder(id);tag=uuid.uuid4().hex[:12]
        if payload.get('asset_id'):
            selected=next((x for x in self.assets() if x['id']==payload['asset_id']),None)
            if not selected:raise ValueError('Unknown authorized asset')
            path=(ROOT/'simulator-assets'/selected['file']).resolve()
            if (ROOT/'simulator-assets').resolve() not in path.parents:raise ValueError('Invalid asset path')
            asset=Image.open(path).convert('RGBA');details={'method':'existing authorized alpha asset','provenance':selected}
        else:
            image,raw=decode_image(payload['image']);(folder/f'asset-upload-{tag}.bin').write_bytes(raw);asset,details=extract_asset(image)
        filename=f'asset-{tag}.png';asset.save(folder/filename)
        p['asset']={'url':f'/media/{id}/{filename}','file':filename,'name':str(payload.get('name') or payload.get('asset_id') or 'Uploaded furniture')[:160],'width':asset.width,'height':asset.height,'extraction':details}
        return self.write(p)
    def set_background(self,id,payload):
        self.check_idle(id);p=self.read(id);folder=self.folder(id)
        if payload.get('clear'):
            p['background_override']=None
        else:
            image,raw=decode_image(payload['image'])
            with Image.open(io.BytesIO(raw)) as source:
                exact_size=ImageOps.exif_transpose(source).size
            if image.size!=(p['image_width'],p['image_height']) or exact_size!=image.size:raise ValueError('Corrected background must have the original pixel dimensions; rescaling is not allowed')
            tag=uuid.uuid4().hex[:12];name=f'background-upload-{tag}.png'
            image.convert('RGB').save(folder/name);(folder/f'background-upload-{tag}.bin').write_bytes(raw)
            p['background_override']={'file':name,'url':f'/media/{id}/{name}','provenance':'Manually supplied appearance correction; not verified hidden geometry'}
            scope=p.get('layers',{}).get('restoration-mask') if p.get('settings',{}).get('options',{}).get('restoration_scope')=='visible' else None
            if scope:p['background_override'].update(scope_mask_file=scope.rsplit('/',1)[-1],scope='visible_when_uploaded')
        return self.write(p)
    def history(self,id,payload):
        self.check_idle(id);p=self.read(id);direction=payload.get('direction')
        if direction not in ('undo','redo'):raise ValueError('History direction must be undo or redo')
        outputs=p['outputs'];current=next((i for i,o in enumerate(outputs) if o['url']==p.get('latest_result_url')),len(outputs)-1)
        target=current+(-1 if direction=='undo' else 1)
        if not 0<=target<len(outputs):raise ValueError('No earlier or later render version')
        o=outputs[target];p.update(latest_result_url=o['url'],settings=copy.deepcopy(o['settings']),mask_url=o['mask_url'],last_diagnostics=o['diagnostics'],layers=o.get('layers',{}),background_override=o.get('background_override'))
        if o.get('asset'):p['asset']=copy.deepcopy(o['asset'])
        return self.write(p)
    def settings(self,id,payload):
        self.check_idle(id);p=self.read(id)
        # Settings are observation/edit parameters, never physical calibration.
        json.dumps(payload,allow_nan=False)
        if len(json.dumps(payload))>200_000:raise ValueError('Too many editing points')
        p['name']=str(payload.get('name') or p['name'])[:160]
        p['settings']={k:copy.deepcopy(payload[k]) for k in ['box','mask_edits','protected_edits','placement','options','reference','selected_detection_id'] if k in payload};p['physical_fit_verified']=False;return self.write(p)
    def submit(self,id,action,payload):
        self.check_idle(id);self.read(id);jobid=uuid.uuid4().hex;self.active.add(id)
        self.jobs[jobid]={'id':jobid,'project_id':id,'status':'queued','progress':'Waiting for local worker','started_at':now()}
        self.pool.submit(self.execute,jobid,id,action,copy.deepcopy(payload));return {'job_id':jobid}
    def execute(self,jobid,id,action,payload):
        job=self.jobs[jobid];job.update(status='running',progress='Preparing '+action);started=time.perf_counter()
        try:
            with self.lock:p=self.read(id)
            folder=self.folder(id)
            if action=='analyze':
                if (folder/'analysis.json').exists() and p.get('analysis',{}).get('complete'):
                    job['progress']='Reusing completed inference for this unchanged image'
                else:
                    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                    command=[sys.executable,'-B','-m','photo_simulator.worker','--folder',str(folder),'--device',self.device]
                    with (folder/'inference.log').open('a') as log:
                        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                        try:
                            while proc.poll() is None:
                                if time.perf_counter()-started>240:raise TimeoutError('Inference exceeded 240 seconds; use manual selection')
                                progress=folder/'analysis-progress.json'
                                if progress.exists():
                                    try:job['progress']='Running '+json.loads(progress.read_text()).get('stage','models')+' locally'
                                    except ValueError:pass
                                time.sleep(.3)
                        finally:
                            if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
                        if proc.returncode:raise RuntimeError(f'Inference worker exited {proc.returncode}; use manual selection. See saved inference.log')
                    analysis=json.loads((folder/'analysis.json').read_text());analysis['complete']=True
                    p['analysis']=analysis;p['detections']=analysis.get('detections',[])
                    for filename in ['depth.png','semantic.png','detection.png']:
                        if (folder/filename).exists():p[filename.split('.')[0]+'_url']=f'/media/{id}/{filename}'
            else:
                image=Image.open(folder/'original.png').convert('RGB');box=payload.get('box')
                if not box:raise ValueError('Select a detected object or draw a manual box first')
                job['progress']='Segmenting selection and applying your mask corrections'
                semantic=None;semantic_diagnostic='Unavailable; classical mask fallback'
                if (folder/'segmentation.npz').exists():
                    dets=p.get('detections',[]);match=max(dets,key=lambda d:box_iou(d['box'],box)) if dets else None
                    category=match['label'] if match and box_iou(match['box'],box)>.15 else None
                    codes={'sofa':[23],'chair':[19,30],'table':[15,64],'bed':[7],'cabinet':[10],'desk':[33],'lamp':[36]}.get(category,[])
                    values=np.load(folder/'segmentation.npz',allow_pickle=False);labels=values['labels'];confidence=values['confidence']
                    region=np.zeros(labels.shape,bool)
                    try:x1,y1,x2,y2=[int(float(v)) for v in box]
                    except (ValueError,TypeError,OverflowError):raise ValueError('Invalid selection box')
                    region[max(0,y1):min(image.height,y2),max(0,x1):min(image.width,x2)]=True
                    chosen=region&np.isin(labels,codes)&(confidence>=.25)
                    if chosen.sum()>max(20,region.sum()*.03):semantic=Image.fromarray(np.uint8(chosen)*255);semantic_diagnostic='UPerNet semantic category inside selected box; not an instance-specific SAM mask'
                mask,maskinfo=make_mask(image,box,payload.get('mask_edits',[]),initial_mask=semantic)
                maskinfo['semantic_support']=semantic_diagnostic
                tag=uuid.uuid4().hex[:12];maskname=f'mask-{tag}.png';mask.save(folder/maskname);p['mask_url']=f'/media/{id}/{maskname}';p['mask_diagnostics']=maskinfo
                if action=='render':
                    if not p.get('asset'):raise ValueError('Choose or upload a replacement image')
                    job['progress']='Filling background and compositing replacement'
                    asset=Image.open(folder/p['asset']['file']).convert('RGBA');depth=None
                    if (folder/'depth.npz').exists():depth=np.load(folder/'depth.npz',allow_pickle=False)['relative_depth']
                    protected=Image.new('L',image.size);draw=ImageDraw.Draw(protected)
                    for edit in payload.get('protected_edits',[]):
                        points=edit.get('points',[])
                        if len(points)>1000 or len(points)<3:raise ValueError('Protected polygon needs 3–1000 points')
                        pts=np.asarray(points,dtype=float)
                        if pts.shape!=(len(points),2) or not np.isfinite(pts).all():raise ValueError('Invalid protected polygon')
                        draw.polygon([tuple(p) for p in pts],fill=255)
                    # Semantic table pixels are preserved only where inferred to overlap the selected furniture.
                    if payload.get('options',{}).get('depth_assist') and (folder/'segmentation.npz').exists():
                        v=np.load(folder/'segmentation.npz',allow_pickle=False);table=np.isin(v['labels'],[15,64])&(v['confidence']>.4)
                        protected=Image.fromarray(np.maximum(np.array(protected),table.astype(np.uint8)*255))
                    placement=payload.get('placement') or {'x':(box[0]+box[2])/2,'y':(box[1]+box[3])/2,'width':box[2]-box[0],'height':box[3]-box[1],'rotation':0,'skew':0,'flip':False}
                    excluded=Image.new('L',image.size);edraw=ImageDraw.Draw(excluded)
                    for edit in payload.get('mask_edits',[]):edraw.polygon([tuple(point) for point in edit['points']],fill=255 if edit['mode']=='subtract' else 0)
                    override=Image.open(folder/p['background_override']['file']).convert('RGB') if p.get('background_override') else None
                    override_scope=Image.open(folder/p['background_override']['scope_mask_file']) if p.get('background_override',{} ) and p['background_override'].get('scope_mask_file') else None
                    layers={}
                    output,diagnostics=composite(image,mask,asset,placement,payload.get('options',{}),depth,protected,layers=layers,background_override=override,cache_dir=folder/'background-cache-v2',expansion_exclusion=excluded,background_override_mask=override_scope)
                    layer_urls={}
                    for layer,im in layers.items():
                        name=f'{layer}-{tag}.png';im.save(folder/name);layer_urls[layer]=f'/media/{id}/{name}'
                    p['layers']=layer_urls
                    filename=f'result-{tag}.png';output.save(folder/filename)
                    diagnostics.update(total_job_seconds=time.perf_counter()-started,method='rough mask-based composite; classical background fill',physical_fit_verified=False)
                    p['outputs'].append({'url':f'/media/{id}/{filename}','mask_url':p['mask_url'],'created_at':now(),'settings':payload,'diagnostics':diagnostics,'layers':layer_urls,'asset':copy.deepcopy(p['asset']),'background_override':copy.deepcopy(p.get('background_override'))})
                    p['latest_result_url']=f'/media/{id}/{filename}';p['last_diagnostics']=diagnostics
                p['settings']={k:payload[k] for k in ['box','mask_edits','protected_edits','placement','options','reference','selected_detection_id'] if k in payload}
            with self.lock:self.write(p)
            job.update(status='complete',progress='Complete',project=p,elapsed_seconds=time.perf_counter()-started)
        except Exception as exc:
            job.update(status='failed',progress='Stopped; original photograph and previous outputs retained',error=str(exc),elapsed_seconds=time.perf_counter()-started)
        finally:
            with self.lock:self.active.discard(id)
            atomic(self.folder(id)/('job-'+jobid+'.json'),job)

class Handler(BaseHTTPRequestHandler):
    server_version='FurniturePhotoSimulator/1.0'
    def send(self,status,body,mime='application/json'):
        if isinstance(body,(dict,list)):body=json.dumps(body,allow_nan=False).encode()
        if isinstance(body,str):body=body.encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'");self.end_headers();self.wfile.write(body)
    def valid_host(self):return self.headers.get('Host','').split(':')[0] in ('localhost','127.0.0.1')
    def do_GET(self):
        try:
            if not self.valid_host():return self.send(403,{'error':'Localhost access only'})
            parts=urlsplit(self.path).path.strip('/').split('/');store=self.server.store
            if parts==['api','projects']:return self.send(200,{'projects':store.listing()})
            if parts==['api','assets']:return self.send(200,{'assets':[{**a,'url':'/assets/'+a['id']} for a in store.assets()]})
            if len(parts)==3 and parts[:2]==['api','projects']:return self.send(200,store.read(parts[2]))
            if len(parts)==3 and parts[:2]==['api','jobs']:
                value=store.jobs.get(parts[2]);return self.send(200 if value else 404,value or {'error':'Job unavailable after server restart; reopen saved simulation or retry explicitly'})
            if len(parts)==3 and parts[0]=='media':
                if not re.fullmatch(r'[a-zA-Z0-9_-]+\.png',parts[2]):raise ValueError('Invalid media filename')
                path=store.folder(parts[1])/parts[2]
            elif len(parts)==2 and parts[0]=='assets':
                a=next((a for a in store.assets() if a['id']==parts[1]),None)
                if not a:raise FileNotFoundError('Asset missing')
                path=ROOT/'simulator-assets'/a['file']
            else:
                name={'':'index.html','index.html':'index.html','app.js':'app.js','style.css':'style.css'}.get('/'.join(parts))
                if not name:raise FileNotFoundError('Not found')
                path=PUBLIC/name
            return self.send(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        except FileNotFoundError as exc:self.send(404,{'error':str(exc)})
        except (ValueError,KeyError) as exc:self.send(400,{'error':str(exc)})
    def do_POST(self):
        try:
            if not self.valid_host():return self.send(403,{'error':'Localhost access only'})
            origin=self.headers.get('Origin')
            if origin and origin not in ('http://'+self.headers.get('Host',''),):return self.send(403,{'error':'Cross-origin writes rejected'})
            length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>MAX_BYTES:return self.send(413,{'error':'Request must be under 24 MB'})
            payload=json.loads(self.rfile.read(length),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON number')))
            if not isinstance(payload,dict):raise ValueError('Object payload required')
            parts=urlsplit(self.path).path.strip('/').split('/');store=self.server.store
            with store.lock:
                if parts==['api','projects']:return self.send(201,store.create(payload))
                if len(parts)!=4 or parts[:2]!=['api','projects']:raise FileNotFoundError('Unknown API endpoint')
                id,action=parts[2:]
                if action=='asset':result=store.set_asset(id,payload)
                elif action=='background':result=store.set_background(id,payload)
                elif action=='history':result=store.history(id,payload)
                elif action=='settings':result=store.settings(id,payload)
                elif action=='restore':
                    store.check_idle(id);p=store.read(id);p['latest_result_url']=None;result=store.write(p)
                elif action in ('analyze','mask','render'):result=store.submit(id,action,payload)
                else:raise FileNotFoundError('Unknown operation')
            self.send(200,result)
        except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':str(exc)})
        except FileNotFoundError as exc:self.send(404,{'error':str(exc)})
        except Exception as exc:self.send(500,{'error':str(exc)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8787);p.add_argument('--store',default=str(ROOT/'.local-simulations'));p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.store=Store(args.store,args.device)
    print(f'Photo simulator: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    finally:server.server_close();server.store.pool.shutdown(wait=False,cancel_futures=True)
if __name__=='__main__':main()
