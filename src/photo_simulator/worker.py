"""Offline inference using existing recognition/layout classes, isolated from the HTTP process."""
from __future__ import annotations
import argparse,gc,json,os,time,resource
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from PIL import Image,ImageDraw
from furniture_bench.detection.owlv2 import Owlv2Detector
from room_layout.models import LocalModel

ROOT=Path(__file__).resolve().parents[2]

class DeviceLayout(LocalModel):
    """Reuse checkpoint loading; move tensors explicitly without changing original CPU adapter."""
    def __init__(self,path,stage,device):
        super().__init__(path,stage);self.device=device
    def load(self):
        super().load();self.model.to(self.device)
    def predict(self,image):
        import torch
        if self.model is None:self.load()
        if self.device=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.inference_mode():
            inputs={k:v.to(self.device) for k,v in self.processor(images=image,return_tensors='pt').items()}
            output=self.model(**inputs)
            if self.stage=='depth':
                value=torch.nn.functional.interpolate(output.predicted_depth[:,None],size=(image.height,image.width),mode='bicubic',align_corners=False)[0,0].float().cpu().numpy()
                result={'relative_depth':value}
            else:
                confidence,labels=output.logits.softmax(dim=1).max(dim=1)
                labels=np.asarray(Image.fromarray(labels[0].cpu().numpy().astype(np.uint8)).resize(image.size,Image.Resampling.NEAREST))
                confidence=np.asarray(Image.fromarray(confidence[0].float().cpu().numpy()).resize(image.size,Image.Resampling.BILINEAR))
                result={'labels':labels,'confidence':confidence.astype(np.float32)}
        if self.device=='cuda':torch.cuda.synchronize()
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite model output')
        return result,time.perf_counter()-started

def run(folder,device):
    import torch
    folder=Path(folder);image=Image.open(folder/'original.png').convert('RGB');torch.set_num_threads(4)
    device='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    result={'device':device,'stages':{},'warnings':[],'physical_fit_verified':False,'geometry':{'image_coordinates':'pixels','physical_scale':'unknown','camera_calibration':'unknown','hidden_surfaces':'unknown','relative_depth':'estimated only'}}
    started=time.perf_counter()
    stages=[('detection',ROOT/'models/owlv2-original-baseline-20261103'),('segmentation',ROOT/'models/photo-simulator-existing/segmentation-876ffc56'),('depth',ROOT/'models/photo-simulator-existing/depth-5426e4f0')]
    for stage,path in stages:
        model=None;t=time.perf_counter()
        try:
            (folder/'analysis-progress.json').write_text(json.dumps({'stage':stage,'status':'running'}))
            if not path.is_dir():raise FileNotFoundError('Local checkpoint absent; downloads disabled')
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            if stage=='detection':
                model=Owlv2Detector(str(path),device=device,allow_download=False,threshold=.18,nms_iou=.5)
                detections,seconds=model.detect(image)
                result['detections']=[dict(d,id='det-'+str(i)) for i,d in enumerate(detections)]
                annotated=image.copy();draw=ImageDraw.Draw(annotated)
                for d in detections:
                    draw.rectangle(d['box'],outline='#00dbad',width=4);draw.text(d['box'][:2],f"{d['label']} {d['score']:.2f}",fill='black',stroke_width=2,stroke_fill='white')
                annotated.save(folder/'detection.png')
            else:
                model=DeviceLayout(path,stage,device);arrays,seconds=model.predict(image)
                np.savez_compressed(folder/(stage+'.npz'),**arrays)
                if stage=='depth':
                    d=arrays['relative_depth'];lo,hi=np.percentile(d,[2,98]);preview=np.uint8(np.clip((d-lo)/max(float(hi-lo),1e-6),0,1)*255);Image.fromarray(preview).save(folder/'depth.png')
                else:
                    result['id2label']={str(k):v for k,v in model.model.config.id2label.items()}
                    labels=arrays['labels'];rgb=np.array(image).copy();colors={0:(130,180,255),3:(240,160,60),23:(100,235,135),15:(220,100,190),64:(220,100,190)}
                    for code,color in colors.items():rgb[labels==code]=(rgb[labels==code]*.5+np.array(color)*.5).astype(np.uint8)
                    Image.fromarray(rgb).save(folder/'semantic.png')
            result['stages'][stage]={'status':'complete','inference_seconds':seconds,'total_stage_seconds':time.perf_counter()-t,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,'checkpoint':str(path.relative_to(ROOT))}
        except Exception as exc:
            result['stages'][stage]={'status':'failed','error':str(exc),'total_stage_seconds':time.perf_counter()-t};result['warnings'].append(stage+': '+str(exc))
        finally:
            del model;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            (folder/'analysis.json').write_text(json.dumps(result,indent=2))
    result['total_seconds']=time.perf_counter()-started;result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result['geometry']['relative_depth']='estimated ordinal depth' if result['stages']['depth']['status']=='complete' else 'unavailable'
    (folder/'analysis.json').write_text(json.dumps(result,indent=2));(folder/'analysis-progress.json').write_text(json.dumps({'status':'complete'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);args=parser.parse_args();run(args.folder,args.device)
