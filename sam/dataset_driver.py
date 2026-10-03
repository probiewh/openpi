#!/usr/bin/env python3
"""Start workers, validate a complete new dataset, then mirror with rsync."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rel(info,ep,key=None):
    d=dict(episode_index=ep,episode_chunk=ep//info['chunks_size'])
    if key is not None:d['video_key']=key
    return info['video_path' if key is not None else 'data_path'].format(**d)
def state(**kw):write(ROOT/'status.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))
def path_checks(c):
    source=Path(c['source']).resolve();destination=Path(c['destination']).resolve();mirror=Path(c['mirror']).resolve()
    if source!=Path('/root/data/xxy/openpi/datasets/competition/cup_four_cameras'):raise RuntimeError('Unexpected source')
    if destination.parent!=source.parent or destination.name!='cup_four_cameras_bgaug32_sam2':raise RuntimeError('Unsafe destination')
    if mirror.parent!=Path('/root/data1/xxy/huggingface/lerobot/competition') or mirror.name!=destination.name:raise RuntimeError('Unsafe mirror')
    if source in (destination,mirror):raise RuntimeError('Refusing original dataset overwrite')
    return source,destination,mirror
def prepare(c):
    source,dest,mirror=path_checks(c);stage=Path(str(dest)+'.processing')
    if dest.exists():
        if (dest/'PROCESSING_COMPLETE.json').is_file():return source,dest,mirror,stage,True
        raise RuntimeError('Destination exists without completion marker; refusing overwrite')
    stage.mkdir(parents=True,exist_ok=True)
    owner=stage/'PROCESSING_OWNER.json'
    if owner.exists():
        if json.loads(owner.read_text()).get('source')!=str(source):raise RuntimeError('Staging owner mismatch')
    else:
        unexpected=[p for p in stage.iterdir() if p.name!='processing_reports']
        if unexpected:raise RuntimeError('Nonempty unknown staging directory')
        write(owner,dict(source=str(source),repo_id=c['repo_id'],settings=c))
    subprocess.run(['rsync','-a','--exclude=videos/',str(source)+'/',str(stage)+'/'],check=True)
    return source,dest,mirror,stage,False
def finalize(c,source,dest,stage):
    info=json.loads((source/'meta/info.json').read_text());eps=[json.loads(s) for s in (source/'meta/episodes.jsonl').read_text().splitlines()]
    keys=[k for k,v in info['features'].items() if v['dtype']=='video'];reports=[]
    image_acc={k:dict(frames=0,pixels=0,sum=np.zeros(3),sum2=np.zeros(3),min=np.ones(3),max=np.zeros(3),histogram=np.zeros((3,256),np.int64)) for k in keys}
    verified=[];unsafe=0;attempted=0;changed=0;aug_eps=[];quality_warnings=[]
    for ep in eps:
        i=ep['episode_index'];r=json.loads((stage/f'processing_reports/episode_{i:06d}.json').read_text());reports.append(r)
        a,b=source/rel(info,i),stage/rel(info,i)
        if digest(a)!=digest(b):raise RuntimeError(f'Original state/actions changed: episode {i}')
        verified.append(dict(path=str(b.relative_to(stage)),sha256=digest(b)))
        if r['augment']:aug_eps.append(i)
        for k in keys:
            v=r['videos'][k]
            if v['frames']!=ep['length'] or v['protected_rgb_max_difference_before_encoding']!=0:raise RuntimeError('Frame/pixel integrity failure')
            video=stage/rel(info,i,k)
            if digest(video)!=v['video_sha256']:raise RuntimeError('Video integrity failure')
            verified.append(dict(path=str(video.relative_to(stage)),sha256=v['video_sha256']))
            if r['augment']:
                attempted+=v['frames'];unsafe+=v['unsafe_frames'];changed+=v['augmented_frames']
                if v['unsafe_frames']/v['frames']>c['max_unsafe_fraction']:quality_warnings.append(dict(episode=i,camera=k,unsafe_fraction=v['unsafe_frames']/v['frames']))
            a=v['accumulator'];acc=image_acc[k]
            acc['frames']+=a['frames'];acc['pixels']+=a['pixels'];acc['sum']+=a['sum'];acc['sum2']+=a['sum2'];acc['min']=np.minimum(acc['min'],a['min']);acc['max']=np.maximum(acc['max'],a['max'])
            acc['histogram']+=np.asarray(a['histogram'],np.int64)
    if unsafe/max(1,attempted)>c['max_unsafe_fraction']:
        write(stage/'QUALITY_REVIEW_REQUIRED.json',dict(unsafe_fraction=unsafe/attempted,warnings=quality_warnings))
        raise RuntimeError('Too many unsafe frames; kept originals, but refusing training-ready finalization')
    stats=json.loads((source/'meta/stats.json').read_text())
    image_stats={}
    for k,a in image_acc.items():
        mean=a['sum']/a['pixels'];std=np.sqrt(np.maximum(0,a['sum2']/a['pixels']-mean**2))
        shape=lambda x:np.asarray(x).reshape(3,1,1).tolist()
        image_stats[k]=dict(mean=shape(mean),std=shape(std),min=shape(a['min']),max=shape(a['max']),count=[a['frames']])
        for q in [.01,.10,.50,.90,.99]:image_stats[k][f'q{int(q*100):02d}']=shape([np.searchsorted(np.cumsum(v),max(1,int(np.ceil(q*a['pixels']))))/255 for v in a['histogram']])
        stats[k]=image_stats[k]
    write(stage/'meta/stats.json',stats)
    original_epstats=source/'meta/episodes_stats.jsonl'
    by_ep={r['episode']:r for r in reports}
    lines=[]
    for s in original_epstats.read_text().splitlines():
        e=json.loads(s);r=by_ep[e['episode_index']]
        for k in keys:e['stats'][k]=r['videos'][k]['image_stats']
        lines.append(json.dumps(e,ensure_ascii=False))
    (stage/'meta/episodes_stats.jsonl').write_text('\n'.join(lines)+'\n')
    for k in keys:
        # Half the videos are bit-exact original copies; codec layout/FPS remain compatible.
        info['features'][k]['info'].pop('video.crf',None)
        info['features'][k]['info'].pop('video.preset',None)
    info['repo_id']=c['repo_id'];write(stage/'meta/info.json',info)
    # Non-image statistics must remain identical, including action normalization.
    if any(stats[k]!=v for k,v in json.loads((source/'meta/stats.json').read_text()).items() if k not in keys):raise RuntimeError('Non-image statistics changed')
    manifest=dict(source=str(source),destination=str(dest),repo_id=c['repo_id'],total_episodes=len(eps),total_frames=info['total_frames'],total_videos=len(eps)*len(keys),augmented_episodes=aug_eps,attempted_augmented_frames=attempted,augmented_frames=changed,unsafe_frames_kept_original=unsafe,quality_warnings=quality_warnings,settings=c,verified_files=verified,notes=['RGB invariance is verified before lossy H264 encoding.','100 episodes preserve source video bytes; others use foreground-only background replacement.','No extra color/crop/rotation augmentation was baked into this dataset.'])
    write(stage/'PROCESSING_COMPLETE.json',manifest)
    if dest.exists():raise RuntimeError('Destination appeared during processing; refusing overwrite')
    stage.rename(dest)
    return manifest
def mirror_dataset(c,dest,mirror):
    if mirror.exists() and any(mirror.iterdir()) and not (mirror/'PROCESSING_OWNER.json').exists():raise RuntimeError('Mirror exists and is not owned by this pipeline')
    mirror.mkdir(parents=True,exist_ok=True)
    state(phase='syncing',destination=str(dest),mirror=str(mirror))
    subprocess.run(['rsync','-a','--checksum','--partial','--delay-updates',str(dest)+'/',str(mirror)+'/'],check=True)
    manifest=json.loads((dest/'PROCESSING_COMPLETE.json').read_text())
    for item in manifest['verified_files']:
        if digest(mirror/item['path'])!=item['sha256']:raise RuntimeError('Mirror checksum mismatch: '+item['path'])
    for name in ['info.json','stats.json','episodes.jsonl','episodes_stats.jsonl','tasks.jsonl']:
        if digest(dest/'meta'/name)!=digest(mirror/'meta'/name):raise RuntimeError('Mirror metadata mismatch')
    write(ROOT/'status.json',dict(phase='complete',destination=str(dest),mirror=str(mirror),total_episodes=manifest['total_episodes'],total_frames=manifest['total_frames'],unsafe_frames_kept_original=manifest['unsafe_frames_kept_original'],finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'settings.json'));p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--prepare-only',action='store_true');p.add_argument('--skip-smoke',action='store_true');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());source,dest,mirror,stage,already=prepare(c)
    if a.prepare_only:print('Staging prepared',stage,flush=True);return
    if already:mirror_dataset(c,dest,mirror);return
    gpu_ids=a.gpus.split(',');logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    if not a.skip_smoke:
        state(phase='smoke_check')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_ids[0],OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        subprocess.run([sys.executable,'-u',str(ROOT/'process_dataset.py'),'--config',a.config,'--smoke'],env=env,check=True)
    state(phase='processing',gpus=gpu_ids,staging=str(stage),total_episodes=200)
    workers=[]
    for i,gpu in enumerate(gpu_ids):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        log=open(logs/f'worker_{i}.log','a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'process_dataset.py'),'--config',a.config,'--worker',str(i),'--workers',str(len(gpu_ids))],env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((proc,log))
    write(ROOT/'workers.json',[dict(worker=i,pid=proc.pid,gpu=gpu_ids[i]) for i,(proc,_) in enumerate(workers)])
    failed=[]
    while workers:
        remaining=[]
        for proc,log in workers:
            code=proc.poll()
            if code is None:remaining.append((proc,log))
            else:
                log.close()
                if code:failed.append(dict(pid=proc.pid,exit_code=code))
        workers=remaining
        if failed:
            for proc,log in workers:proc.terminate();proc.wait();log.close()
            raise RuntimeError('Workers failed: '+json.dumps(failed))
        if workers:time.sleep(10)
    state(phase='validating');manifest=finalize(c,source,dest,stage)
    print('Dataset validated',manifest['total_episodes'],manifest['total_frames'],flush=True)
    mirror_dataset(c,dest,mirror)
    print('PROCESSING AND RSYNC COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        state(phase='failed',error=str(e));raise
