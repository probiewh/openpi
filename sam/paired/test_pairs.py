#!/usr/bin/env python3
"""Synthetic end-to-end merge test: labels, frame IDs, pairing and video stats."""
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pair_common import write,read,relative,fingerprint
from merge_pairs import build,decode_stats,numeric_stats,jsonlines,reindex

def main():
    assert fingerprint({'seed':1,'augment_episode_fraction':0.5,'destination':'a'})==fingerprint({'seed':1,'augment_episode_fraction':1.0,'destination':'b'})
    assert fingerprint({'seed':1,'safety_margin_px':12})!=fingerprint({'seed':1,'safety_margin_px':13})
    with tempfile.TemporaryDirectory(prefix='sam-pairs-test-') as td:
        root=Path(td);source=root/'source';full=root/'augmented';dest=root/'pairs'
        features={k:dict(dtype='video',shape=[16,24,3],info={'video.crf':30,'video.preset':12}) for k in ['top','left','right','goal']}
        features.update({k:dict(dtype='float32' if k in ('action','observation.state','timestamp') else 'int64',shape=[26] if k in ('action','observation.state') else [1]) for k in ['action','observation.state','timestamp','episode_index','frame_index','index','task_index']})
        info=dict(features=features,fps=30,chunks_size=1000,total_episodes=2,total_frames=8,total_videos=8,total_chunks=1,
                  splits={'train':'0:2'},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                  video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
        eps=[dict(episode_index=i,length=n,tasks=['pick']) for i,n in enumerate((3,5))]
        epstats=[];offset=0
        for e in eps:
            i=e['episode_index'];n=e['length'];table=pa.table({
                'action':pa.array(np.arange(n*26,dtype=np.float32).reshape(n,26).tolist(),type=pa.list_(pa.float32(),26)),
                'observation.state':pa.array(np.ones((n,26),np.float32).tolist(),type=pa.list_(pa.float32(),26)),
                'timestamp':pa.array(np.arange(n,dtype=np.float32)/30),
                'episode_index':pa.array(np.full(n,i,dtype=np.int64)),
                'frame_index':pa.array(np.arange(n,dtype=np.int64)),
                'index':pa.array(np.arange(offset,offset+n,dtype=np.int64)),
                'task_index':pa.array(np.zeros(n,dtype=np.int64))})
            for base in (source,full):
                path=base/relative(info,i);path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,path)
            stats={k:numeric_stats(table[k].to_pylist()) for k in table.column_names};videos={}
            for key in ('top','left','right','goal'):
                for base,color in ((source,40),(full,180)):
                    path=base/relative(info,i,key);path.parent.mkdir(parents=True,exist_ok=True)
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(24,16))
                    if not writer.isOpened():raise RuntimeError('Test video encoder unavailable')
                    for j in range(n):writer.write(np.full((16,24,3),color+j,np.uint8))
                    writer.release()
                videos[key]=decode_stats(full/relative(info,i,key),n,[16,24,3])
                stats[key]=decode_stats(source/relative(info,i,key),n,[16,24,3])['image_stats']
            write(full/f'processing_reports/episode_{i:06d}.json',dict(episode=i,augment=True,videos=videos))
            epstats.append(dict(episode_index=i,stats=stats));offset+=n
        write(source/'meta/info.json',info);write(source/'meta/stats.json',epstats[0]['stats'])
        jsonlines(source/'meta/episodes.jsonl',eps);jsonlines(source/'meta/episodes_stats.jsonl',epstats)
        jsonlines(source/'meta/tasks.jsonl',[dict(task_index=0,task='pick')])
        write(full/'PROCESSING_COMPLETE.json',dict(total_episodes=2,total_frames=8,augmented_episodes=[0,1],augmented_frames=32,unsafe_frames_kept_original=0))
        manifest=build(source,full,dest,root/'mirror',sync=False)
        assert manifest['total_episodes']==4 and manifest['total_frames']==16
        out=read(dest/'meta/info.json');assert out['splits']=={'train':'0:4'} and out['total_videos']==16
        mapping=[json.loads(s) for s in (dest/'meta/augmentation_pairs.jsonl').read_text().splitlines()]
        assert [m['pair_id'] for m in mapping]==[0,1,0,1]
        all_index=[]
        for i in range(4):
            t=pq.read_table(dest/relative(info,i));original=pq.read_table(source/relative(info,i%2))
            assert np.all(t['episode_index'].to_numpy()==i)
            for key in ('action','observation.state','timestamp','frame_index','task_index'):assert t[key].equals(original[key])
            all_index.extend(t['index'].to_pylist())
        assert all_index==list(range(16))
        stats=read(dest/'meta/stats.json')
        assert stats['index']['min']==[0] and stats['index']['max']==[15]
        assert stats['index']['count']==[16] and stats['top']['count']==[16]
        assert 0.3<stats['top']['mean'][0][0][0]<0.6
        # Second run must be resumable without recomputing or changing the final data.
        assert build(source,full,dest,root/'mirror',sync=False)==manifest
        print('PASS: paired IDs, action/state/time preservation, 4 cameras, frame totals, stats and resume')

if __name__=='__main__':main()
