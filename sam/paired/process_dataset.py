#!/usr/bin/env python3
"""Independent offline foreground-preserving LeRobot v2.1 dataset processor.

Automatic grounding + SAM2 image segmentation on every augmented frame.
No dependence on hand-written episode timings, fixed wrist ROIs, or external APIs.
Unsafe frames remain original and are logged. Videos are never processed in place.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from pair_common import fingerprint

def atomic_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(p)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def relative(info,ep,key=None):
    args=dict(episode_index=ep,episode_chunk=ep//info['chunks_size'])
    if key is not None:args['video_key']=key
    return info['video_path' if key is not None else 'data_path'].format(**args)

class ImageStats:
    def __init__(self):
        self.frames=0;self.pixels=0
        self.sum=np.zeros(3,np.float64);self.sum2=np.zeros(3,np.float64)
        self.min=np.ones(3,np.float64);self.max=np.zeros(3,np.float64)
        self.histogram=np.zeros((3,256),np.int64)
    def add(self,bgr):
        a=bgr[:,:,::-1].astype(np.float64)/255
        self.frames+=1;self.pixels+=a.shape[0]*a.shape[1]
        self.sum+=a.sum((0,1));self.sum2+=(a*a).sum((0,1))
        self.min=np.minimum(self.min,a.min((0,1)));self.max=np.maximum(self.max,a.max((0,1)))
        for c in range(3):self.histogram[c]+=np.bincount(bgr[:,:,2-c].ravel(),minlength=256)
    def finish(self):
        mean=self.sum/self.pixels
        def shape(x):return np.asarray(x).reshape(3,1,1).tolist()
        result={'mean':shape(mean),'std':shape(np.sqrt(np.maximum(0,self.sum2/self.pixels-mean**2))),'min':shape(self.min),'max':shape(self.max),'count':[self.frames]}
        for q in [.01,.10,.50,.90,.99]:result[f'q{int(q*100):02d}']=shape([np.searchsorted(np.cumsum(v),max(1,int(np.ceil(q*self.pixels))))/255 for v in self.histogram])
        return result
    def packed(self):
        return dict(frames=self.frames,pixels=self.pixels,sum=self.sum.tolist(),sum2=self.sum2.tolist(),min=self.min.tolist(),max=self.max.tolist(),histogram=self.histogram.tolist())

class Foreground:
    def __init__(self,c):
        self.c=c;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.processor=AutoProcessor.from_pretrained(c['detector'],local_files_only=True)
        self.detector=AutoModelForZeroShotObjectDetection.from_pretrained(c['detector'],local_files_only=True).to(self.device).eval()
        self.sam=SAM2ImagePredictor(build_sam2(c['sam_config'],c['sam_checkpoint'],device=self.device,apply_postprocessing=False))
        self.post_parameters=inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self.previous_area=None
    def reset(self):self.previous_area=None
    def clean(self,mask):
        mask=np.asarray(mask,dtype=np.uint8)
        n,ids,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<self.c['small_island_area']:mask[ids==i]=0
        n,ids,stats,_=cv2.connectedComponentsWithStats(1-mask,8)
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if x>0 and y>0 and x+w<mask.shape[1] and y+h<mask.shape[0] and area<=self.c['small_hole_area']:mask[ids==i]=1
        return mask
    @torch.inference_mode()
    def mask(self,bgr,key):
        h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        inputs=self.processor(images=Image.fromarray(rgb),text=self.c['prompt'],return_tensors='pt').to(self.device)
        outputs=self.detector(**inputs)
        kw=dict(input_ids=inputs.input_ids,target_sizes=[(h,w)],text_threshold=self.c['text_threshold'])
        kw['box_threshold' if 'box_threshold' in self.post_parameters else 'threshold']=self.c['box_threshold']
        detections=self.processor.post_process_grounded_object_detection(outputs,**kw)[0]
        boxes=detections['boxes'].detach().float().cpu().numpy()
        scores=detections['scores'].detach().float().cpu().numpy()
        labels=detections.get('text_labels',detections.get('labels',[]))
        labels=[str(x).lower() for x in labels]
        # Do not silently accept a whole scene or person as a task object.
        accepted=[]
        for i,(box,label) in enumerate(zip(boxes,labels)):
            bw,bh=box[2]-box[0],box[3]-box[1]
            if bw<4 or bh<4 or bw*bh>w*h*.98:continue
            if 'person' in label or 'human' in label:continue
            accepted.append(i)
        accepted=sorted(accepted,key=lambda i:float(scores[i]),reverse=True)[:self.c['max_boxes']]
        if not accepted:return None,{'reason':'no_task_detection','labels':labels}
        boxes=boxes[accepted];labels=[labels[i] for i in accepted]
        if 'wrist' in key and not any('hand' in l or 'arm' in l or 'robot' in l for l in labels):
            return None,{'reason':'robot_not_detected','labels':labels}
        if ('cam_top' in key or '.goal' in key) and not any('table' in l for l in labels):
            return None,{'reason':'table_reference_not_detected','labels':labels}
        self.sam.set_image(rgb)
        ctx=torch.autocast('cuda',dtype=torch.bfloat16) if self.device=='cuda' else contextlib.nullcontext()
        with ctx:
            masks,quality,_=self.sam.predict(box=boxes,multimask_output=False)
        masks=np.asarray(masks).reshape(-1,h,w)
        quality=np.asarray(quality).reshape(-1)
        union=np.zeros((h,w),np.uint8)
        for mask,q in zip(masks,quality):
            if q>=.35:union|=self.clean(mask)
        raw_area=float(union.mean())
        if raw_area<self.c['min_foreground_fraction'] or raw_area>self.c['max_foreground_fraction']:
            return None,{'reason':'foreground_area_invalid','area':raw_area,'labels':labels}
        previous=self.previous_area;self.previous_area=raw_area
        if previous is not None and abs(raw_area-previous)>self.c['max_area_jump']:
            return None,{'reason':'foreground_area_jump','area':raw_area,'previous_area':previous,'labels':labels}
        margin=self.c['safety_margin_px']
        keep=cv2.dilate(union,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1)))>0
        # Cup/blue robot safeguard. A missing purple cup must never be erased.
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        purple=((hsv[:,:,0]>=95)&(hsv[:,:,0]<=155)&(hsv[:,:,1]>=50)&(hsv[:,:,2]>=35)).astype(np.uint8)
        n,ids,stats,_=cv2.connectedComponentsWithStats(purple,8)
        meaningful=np.zeros_like(purple,dtype=bool)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]>=100:meaningful|=ids==i
        missing=float((meaningful&~keep).sum())/max(1,int(meaningful.sum()))
        if missing>self.c['max_purple_unprotected_fraction']:
            return None,{'reason':'purple_cup_or_hand_unprotected','missing_fraction':missing,'labels':labels}
        d=cv2.distanceTransform((~keep).astype(np.uint8),cv2.DIST_L2,5)
        alpha=np.clip(d/self.c['outside_feather_px'],0,1)
        alpha=alpha*alpha*(3-2*alpha)
        return alpha,{'reason':'ok','area':raw_area,'labels':labels,'sam_min_quality':float(quality.min())}

def background(h,w,seed):
    rng=np.random.default_rng(seed)
    base=rng.uniform(85,175,3)
    y,x=np.mgrid[:h,:w]
    wave=9*np.sin(x/160+rng.uniform(0,6.28))+7*np.cos(y/130+rng.uniform(0,6.28))+4*np.sin((x+y)/210)
    return np.clip(base[None,None,:]+wave[:,:,None],0,255).astype(np.float32)

def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate','-of','json',str(path)]))
    return data['streams'][0]

def process_video(engine,c,source,dest,ep,key,expected,augment,report_dir,max_frames=None):
    started=time.time();engine.reset()
    dest.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise RuntimeError(f'Cannot decode {source}')
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stats=ImageStats();unsafe={};changed=0;max_error=0;count=0
    preview_frames={int((expected-1)*f) for f in np.linspace(0,1,c['previews_per_video'])}
    if max_frames:preview_frames={0}
    tmp=dest.with_name(dest.stem+'.partial.mp4')
    encoder=None
    if augment:
        encoder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset',c['video_preset'],'-crf',str(c['video_crf']),'-pix_fmt','yuv420p','-g',str(c['video_gop']),'-keyint_min',str(c['video_gop']),'-sc_threshold','0','-threads','2',str(tmp)],stdin=subprocess.PIPE)
    bg=background(h,w,c['seed']+ep)
    goal_alpha=None;goal_meta=None;goal_frame=None
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if max_frames and count>=max_frames:break
            output=bgr
            diagnostic={'reason':'original_episode'}
            alpha=None
            if augment:
                if '.goal' in key and goal_frame is not None and np.array_equal(bgr,goal_frame):
                    alpha,diagnostic=goal_alpha,goal_meta
                else:
                    alpha,diagnostic=engine.mask(bgr,key)
                    if '.goal' in key:goal_alpha,goal_meta,goal_frame=alpha,diagnostic,bgr.copy()
                if alpha is None:
                    reason=diagnostic['reason'];unsafe[reason]=unsafe.get(reason,0)+1
                    if unsafe[reason]<=3:
                        with open(report_dir/'unsafe_frames.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(dict(episode=ep,camera=key,frame=count,**diagnostic))+'\n')
                else:
                    output=np.rint(bgr.astype(np.float32)*(1-alpha[:,:,None])+bg*alpha[:,:,None]).clip(0,255).astype(np.uint8)
                    protected=alpha==0
                    error=int(np.abs(output[protected].astype(np.int16)-bgr[protected].astype(np.int16)).max(initial=0))
                    max_error=max(max_error,error)
                    if error:raise RuntimeError('Protected pixels changed before video encoding')
                    changed+=1
                encoder.stdin.write(output.tobytes())
            stats.add(output)
            if count in preview_frames:
                preview=report_dir/'previews'/f'episode_{ep:06d}'/key
                preview.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(preview/f'{count:06d}_before.jpg'),bgr)
                cv2.imwrite(str(preview/f'{count:06d}_after.jpg'),output)
                if alpha is not None:cv2.imwrite(str(preview/f'{count:06d}_keep.png'),(alpha==0).astype(np.uint8)*255)
                atomic_json(preview/f'{count:06d}.json',diagnostic)
            count+=1
            if count%100==0:print(f'episode={ep} camera={key} frames={count}/{expected} augmented={changed} unsafe={sum(unsafe.values())}',flush=True)
        if not max_frames and count!=expected:raise RuntimeError(f'{source}: decoded {count}, expected {expected}')
        if augment:
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('ffmpeg encoding failed')
            observed=probe(tmp)
            if int(observed['nb_read_frames'])!=count:raise RuntimeError('Encoded frame count mismatch')
            tmp.replace(dest)
            # Metadata describes decoded output, not pre-encoding RGB.
            stats=ImageStats()
            check=cv2.VideoCapture(str(dest))
            while True:
                ok,frame=check.read()
                if not ok:break
                stats.add(frame)
            check.release()
            if stats.frames!=count:raise RuntimeError('Final decode frame count mismatch')
        else:shutil.copy2(source,dest)
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            with contextlib.suppress(Exception):encoder.stdin.close()
            encoder.terminate();encoder.wait()
    return dict(episode=ep,camera=key,frames=count,augmented_frames=changed,unsafe_frames=sum(unsafe.values()),unsafe_reasons=unsafe,protected_rgb_max_difference_before_encoding=max_error,seconds=round(time.time()-started,2),image_stats=stats.finish(),accumulator=stats.packed(),video_sha256=sha256(dest))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-frames',type=int,default=8);p.add_argument('--complement-of')
    a=p.parse_args();c=json.loads(Path(a.config).read_text());torch.set_num_threads(4)
    source=Path(c['source']);dest=Path(c['destination']+'.processing')
    info=json.loads((source/'meta/info.json').read_text())
    episodes=[json.loads(s) for s in (source/'meta/episodes.jsonl').read_text().splitlines()]
    keys=[k for k,v in info['features'].items() if v['dtype']=='video']
    if set(keys)!={'observation.images.cam_top','observation.images.cam_left_wrist','observation.images.cam_right_wrist','observation.images.goal'}:raise RuntimeError('Unexpected video keys')
    order=np.random.default_rng(c['seed']).permutation(len(episodes))
    augmented=set(int(x) for x in order[:round(len(episodes)*c['augment_episode_fraction'])])
    excluded=set()
    if a.complement_of:
        previous=json.loads(Path(a.complement_of).read_text())
        if fingerprint(previous)!=fingerprint(c):raise RuntimeError('Previous rendering config differs')
        order_old=np.random.default_rng(previous['seed']).permutation(len(episodes))
        excluded=set(int(x) for x in order_old[:round(len(episodes)*previous['augment_episode_fraction'])])
    reports=dest/'processing_reports';reports.mkdir(parents=True,exist_ok=True)
    engine=Foreground(c)
    for record in episodes:
        ep=record['episode_index']
        if a.smoke and ep!=0:continue
        if not a.smoke and ep in excluded:continue
        if ep%a.workers!=a.worker:continue
        report=reports/f'episode_{ep:06d}.json'
        if report.exists() and not a.smoke:
            saved=json.loads(report.read_text())
            if saved.get('augment')==(ep in augmented) and saved.get('render_fingerprint')==fingerprint(c) and set(saved['videos'])==set(keys) and all((dest/relative(info,ep,k)).exists() and sha256(dest/relative(info,ep,k))==v['video_sha256'] for k,v in saved['videos'].items()):
                print('Resume: verified completed episode',ep,flush=True);continue
        all_videos={}
        for key in keys:
            target=dest/relative(info,ep,key)
            if a.smoke:target=reports/'smoke_videos'/key/f'episode_{ep:06d}.mp4'
            result=process_video(engine,c,source/relative(info,ep,key),target,ep,key,record['length'],a.smoke or ep in augmented,reports,a.smoke_frames if a.smoke else None)
            all_videos[key]=result
            print(json.dumps({k:v for k,v in result.items() if k not in ('image_stats','accumulator')},ensure_ascii=False),flush=True)
        saved=dict(episode=ep,augment=a.smoke or ep in augmented,render_fingerprint=fingerprint(c),videos=all_videos)
        atomic_json(reports/('smoke.json' if a.smoke else report.name),saved)
        print('Episode complete',ep,flush=True)
    if a.smoke:
        saved=json.loads((reports/'smoke.json').read_text())
        failed=[k for k,v in saved['videos'].items() if v['augmented_frames']==0 or v['unsafe_frames']/v['frames']>c['max_unsafe_fraction']]
        if failed:raise RuntimeError('Smoke QA failed: '+', '.join(failed))
        print('SMOKE PASSED: all four views processed and frame/pixel checks passed',flush=True)

if __name__=='__main__':main()
