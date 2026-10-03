#!/usr/bin/env python3
"""Build one LeRobot v2.1 dataset with matched original/augmented episodes."""
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pair_common import SOURCE, FULL, PAIRS, MIRROR, read, write, relative, sha256

def empty_acc():
    return dict(frames=0,pixels=0,sum=np.zeros(3),sum2=np.zeros(3),
                min=np.ones(3),max=np.zeros(3),histogram=np.zeros((3,256),np.int64))

def add_acc(total, a):
    total['frames']+=a['frames'];total['pixels']+=a['pixels']
    for k in ('sum','sum2','histogram'):total[k]+=np.asarray(a[k])
    total['min']=np.minimum(total['min'],a['min'])
    total['max']=np.maximum(total['max'],a['max'])

def image_stats(a):
    mean=a['sum']/a['pixels']
    shape=lambda x:np.asarray(x).reshape(3,1,1).tolist()
    r=dict(mean=shape(mean),std=shape(np.sqrt(np.maximum(0,a['sum2']/a['pixels']-mean**2))),
           min=shape(a['min']),max=shape(a['max']),count=[a['frames']])
    for q in (.01,.1,.5,.9,.99):
        r[f'q{int(q*100):02d}']=shape([np.searchsorted(np.cumsum(h),max(1,math.ceil(q*a['pixels'])))/255 for h in a['histogram']])
    return r

def numeric_stats(values):
    a=np.asarray(values).reshape(len(values),-1)
    r=dict(min=a.min(0).tolist(),max=a.max(0).tolist(),mean=a.mean(0).tolist(),std=a.std(0).tolist(),count=[len(a)])
    for q in (.01,.1,.5,.9,.99):r[f'q{int(q*100):02d}']=np.quantile(a,q,axis=0).tolist()
    return r

def decode_stats(path, expected, shape):
    # No float64 RGB image allocation: C++ moments and uint8 histograms.
    a=empty_acc();cap=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if list(bgr.shape)!=shape:raise RuntimeError(f'Video shape mismatch: {path}')
            pixels=bgr.shape[0]*bgr.shape[1]
            mean,std=cv2.meanStdDev(bgr)
            mean=mean.ravel()[::-1]/255;std=std.ravel()[::-1]/255
            h=np.array([cv2.calcHist([bgr],[ch],None,[256],[0,256]).ravel().astype(np.int64) for ch in (2,1,0)])
            a['sum']+=mean*pixels;a['sum2']+=(std**2+mean**2)*pixels
            a['histogram']+=h;a['frames']+=1;a['pixels']+=pixels
    finally:cap.release()
    if a['frames']!=expected:raise RuntimeError(f'Frame count mismatch: {path}: {a["frames"]}/{expected}')
    a['min']=np.array([np.flatnonzero(h)[0]/255 for h in a['histogram']])
    a['max']=np.array([np.flatnonzero(h)[-1]/255 for h in a['histogram']])
    serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in a.items()}
    return dict(frames=expected,video_sha256=sha256(path),image_stats=image_stats(a),accumulator=serial)

def copy_checked(source,target,expected_hash):
    if sha256(source)!=expected_hash:raise RuntimeError(f'Source hash changed: {source}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or sha256(target)!=expected_hash:
        tmp=target.with_suffix(target.suffix+'.tmp');shutil.copy2(source,tmp);tmp.replace(target)
    if sha256(target)!=expected_hash:raise RuntimeError(f'Copy checksum mismatch: {target}')

def reindex(table, ep, start):
    n=len(table)
    for key,values in [('episode_index',np.full(n,ep)),('index',np.arange(start,start+n))]:
        field=table.schema.field(key)
        table=table.set_column(table.schema.get_field_index(key),field,pa.array(values,type=field.type))
    return table

def jsonlines(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in items));tmp.replace(path)

def build(source=SOURCE,full=FULL,dest=PAIRS,mirror=MIRROR,sync=True):
    source=Path(source);full=Path(full);dest=Path(dest);mirror=Path(mirror)
    manifest_aug=read(full/'PROCESSING_COMPLETE.json')
    info=read(source/'meta/info.json')
    eps=[json.loads(s) for s in (source/'meta/episodes.jsonl').read_text().splitlines()]
    epstats={e['episode_index']:e for e in [json.loads(s) for s in (source/'meta/episodes_stats.jsonl').read_text().splitlines()]}
    n=len(eps);frames=info['total_frames']
    if [e['episode_index'] for e in eps]!=list(range(n)):raise RuntimeError('Source episode IDs not contiguous')
    if manifest_aug['total_episodes']!=n or manifest_aug['total_frames']!=frames or sorted(manifest_aug['augmented_episodes'])!=list(range(n)):
        raise RuntimeError('Augmented dataset is incomplete')
    stage=Path(str(dest)+'.processing')
    if dest.exists():
        if not (dest/'PROCESSING_COMPLETE.json').exists():raise RuntimeError('Unknown destination; refusing overwrite')
    else:
        stage.mkdir(parents=True,exist_ok=True)
        owner=stage/'PROCESSING_OWNER.json'
        identity=dict(source=str(source),augmented_source=str(full),destination=str(dest),pair_count=n)
        if owner.exists():
            if read(owner)!=identity:raise RuntimeError('Merge staging provenance mismatch')
        elif any(stage.iterdir()):raise RuntimeError('Unknown nonempty merge staging directory')
        else:write(owner,identity)
        keys=[k for k,v in info['features'].items() if v['dtype']=='video']
        totals={k:empty_acc() for k in keys};all_epstats=[];new_eps=[];verified=[];mapping=[]
        index_values=[];episode_values=[];offset=0
        for variant in ('original','augmented'):
            for e in eps:
                i=e['episode_index'];new_id=i+(n if variant=='augmented' else 0);length=e['length']
                src_table=pq.read_table(source/relative(info,i))
                if len(src_table)!=length:raise RuntimeError('Source Parquet length mismatch')
                if not np.array_equal(src_table['frame_index'].to_numpy(),np.arange(length)):
                    raise RuntimeError('Source frame_index is not contiguous')
                if not np.array_equal(src_table['index'].to_numpy(),np.arange(offset%frames,offset%frames+length)):
                    raise RuntimeError('Source global index mismatch')
                if not np.all(src_table['episode_index'].to_numpy()==i):raise RuntimeError('Source episode_index mismatch')
                if variant=='augmented':
                    aug_table=pq.read_table(full/relative(info,i))
                    if not src_table.equals(aug_table,check_metadata=False):raise RuntimeError('Augmented state/action/timestamps differ')
                out_table=reindex(src_table,new_id,offset)
                out_path=stage/relative(info,new_id);out_path.parent.mkdir(parents=True,exist_ok=True)
                tmp=out_path.with_suffix('.parquet.tmp');pq.write_table(out_table,tmp);tmp.replace(out_path)
                loaded=pq.read_table(out_path)
                unchanged=[k for k in src_table.column_names if k not in ('episode_index','index')]
                if not loaded.select(unchanged).equals(src_table.select(unchanged),check_metadata=False):raise RuntimeError('Labels changed in merge')
                if not np.array_equal(loaded['index'].to_numpy(),np.arange(offset,offset+length)):raise RuntimeError('Output global index mismatch')
                if not np.all(loaded['episode_index'].to_numpy()==new_id):raise RuntimeError('Output episode mismatch')
                verified.append(dict(path=str(out_path.relative_to(stage)),sha256=sha256(out_path)))
                r=copy.deepcopy(epstats[i]);r['episode_index']=new_id
                r['stats']['episode_index']=numeric_stats(np.full(length,new_id))
                r['stats']['index']=numeric_stats(np.arange(offset,offset+length))
                if variant=='original':
                    cache=stage/f'processing_reports/original_{i:06d}.json'
                    cached=read(cache) if cache.exists() else {}
                    videos={}
                    for k in keys:
                        video=source/relative(info,i,k);h=sha256(video)
                        if cached.get(k,{}).get('video_sha256')==h and cached[k].get('frames')==length:
                            videos[k]=cached[k]
                        else:videos[k]=decode_stats(video,length,info['features'][k]['shape'])
                    write(cache,videos)
                else:
                    report=read(full/f'processing_reports/episode_{i:06d}.json')
                    if not report['augment']:raise RuntimeError('Unaugmented counterpart encountered')
                    videos=report['videos']
                origin=source if variant=='original' else full
                for k in keys:
                    v=videos[k]
                    if v['frames']!=length:raise RuntimeError('Video/action alignment mismatch')
                    target=stage/relative(info,new_id,k)
                    copy_checked(origin/relative(info,i,k),target,v['video_sha256'])
                    verified.append(dict(path=str(target.relative_to(stage)),sha256=v['video_sha256']))
                    add_acc(totals[k],v['accumulator']);r['stats'][k]=v['image_stats']
                record=copy.deepcopy(e);record['episode_index']=new_id
                new_eps.append(record);all_epstats.append(r)
                mapping.append(dict(episode_index=new_id,source_episode_index=i,pair_id=i,variant=variant))
                index_values.append(np.arange(offset,offset+length));episode_values.append(np.full(length,new_id));offset+=length
                print(f'Merged {variant} episode {i} -> {new_id}: {offset}/{2*frames} frames',flush=True)
        if offset!=frames*2:raise RuntimeError('Merged frame total mismatch')
        stats=copy.deepcopy(read(source/'meta/stats.json'))
        for k,v in stats.items():
            if k not in keys and 'count' in v:v['count']=[frames*2]
        stats['index']=numeric_stats(np.concatenate(index_values))
        stats['episode_index']=numeric_stats(np.concatenate(episode_values))
        for k in keys:stats[k]=image_stats(totals[k])
        merged_info=copy.deepcopy(info)
        merged_info.update(repo_id='competition/'+dest.name,total_episodes=n*2,total_frames=frames*2,
                           total_videos=n*2*len(keys),total_chunks=math.ceil(n*2/info['chunks_size']),splits={'train':f'0:{n*2}'})
        for k in keys:
            for setting in ('video.crf','video.preset'):merged_info['features'][k]['info'].pop(setting,None)
        write(stage/'meta/info.json',merged_info);write(stage/'meta/stats.json',stats)
        jsonlines(stage/'meta/episodes.jsonl',new_eps);jsonlines(stage/'meta/episodes_stats.jsonl',all_epstats)
        jsonlines(stage/'meta/augmentation_pairs.jsonl',mapping)
        shutil.copy2(source/'meta/tasks.jsonl',stage/'meta/tasks.jsonl')
        # These ancillary images are not the four training video features.
        if (source/'images').exists():shutil.copytree(source/'images',stage/'images',dirs_exist_ok=True)
        for p in sorted((stage/'meta').iterdir()):
            verified.append(dict(path=str(p.relative_to(stage)),sha256=sha256(p)))
        write(stage/'PROCESSING_COMPLETE.json',dict(source=str(source),augmented_source=str(full),
              repo_id=merged_info['repo_id'],total_episodes=n*2,total_frames=frames*2,total_videos=n*2*len(keys),
              independent_demonstrations=n,original_episodes=n,augmented_episodes=n,
              original_frame_fraction=0.5,augmented_variant_frame_fraction=0.5,
              actual_changed_frames=manifest_aug['augmented_frames'],
              unsafe_frames_kept_original=manifest_aug['unsafe_frames_kept_original'],
              verified_files=verified,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              notes=['Original IDs 0..N-1; augmented IDs N..2N-1.',
                     'Pair IDs must stay together for any train/validation split.',
                     'Actions, states, timestamps, task IDs and within-episode frame IDs are unchanged.',
                     'Only episode_index and global index are reindexed; 400 variants are not 400 independent demonstrations.']))
        stage.rename(dest)
    complete=read(dest/'PROCESSING_COMPLETE.json')
    if complete.get('source')!=str(source) or complete.get('augmented_source')!=str(full) or complete.get('total_episodes')!=n*2 or complete.get('total_frames')!=frames*2:
        raise RuntimeError('Completed paired dataset provenance/count mismatch')
    if sync:
        if mirror.exists() and any(mirror.iterdir()) and not (mirror/'PROCESSING_OWNER.json').exists():raise RuntimeError('Unknown nonempty mirror')
        mirror.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','--checksum','--partial','--delay-updates',str(dest)+'/',str(mirror)+'/'],check=True)
        for entry in complete['verified_files']:
            if sha256(mirror/entry['path'])!=entry['sha256']:raise RuntimeError('Mirror integrity failure: '+entry['path'])
        for name in ('PROCESSING_COMPLETE.json','PROCESSING_OWNER.json'):
            if sha256(mirror/name)!=sha256(dest/name):raise RuntimeError('Mirror manifest mismatch')
    return complete

if __name__=='__main__':
    cv2.setNumThreads(1)
    build()
