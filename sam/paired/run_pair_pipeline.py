#!/usr/bin/env python3
"""Run the missing half on idle GPUs, reuse the old half, then merge and sync."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from pair_common import SOURCE, OLD, FULL, PAIRS, MIRROR, OLD_ROOT, fingerprint, read, write, relative, sha256

ROOT=Path(__file__).resolve().parent

def status(phase,**kw):
    write(ROOT/'pipeline_status.json',dict(phase=phase,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))

def driver(gpus,*extra):
    subprocess.run([sys.executable,'-u',str(ROOT/'dataset_driver.py'),'--config',str(ROOT/'settings.json'),
                    '--gpus',gpus,'--skip-smoke','--skip-sync',*extra],check=True)

def reuse_old(c,old_config):
    if fingerprint(c)!=fingerprint(old_config):raise RuntimeError('Old/new renderer settings differ')
    old=OLD if (OLD/'PROCESSING_COMPLETE.json').exists() else Path(str(OLD)+'.processing')
    stage=Path(str(FULL)+'.processing')
    info=read(SOURCE/'meta/info.json');keys=[k for k,v in info['features'].items() if v['dtype']=='video']
    reused=0
    for report in sorted((old/'processing_reports').glob('episode_*.json')):
        r=read(report);i=r['episode']
        if not r.get('augment'):continue
        if set(r['videos'])!=set(keys):raise RuntimeError('Old report video keys mismatch')
        for k in keys:
            src=old/relative(info,i,k);expected=r['videos'][k]['video_sha256']
            if not src.exists() or sha256(src)!=expected:raise RuntimeError('Old augmented result checksum failed')
            dst=stage/relative(info,i,k);dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() and sha256(dst)!=expected:raise RuntimeError('Conflicting augmented outputs')
            if not dst.exists():shutil.copy2(src,dst)
        r['render_fingerprint']=fingerprint(c)
        r['reused_from']=str(old)
        write(stage/'processing_reports'/report.name,r)
        previews=old/'processing_reports/previews'/f'episode_{i:06d}'
        if previews.exists():shutil.copytree(previews,stage/'processing_reports/previews'/f'episode_{i:06d}',dirs_exist_ok=True)
        reused+=1
    write(ROOT/'reuse_report.json',dict(reused_augmented_episodes=reused,old_dataset=str(old),render_fingerprint=fingerprint(c)))
    print('Reused completed augmented episodes:',reused,flush=True)
    return reused

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='4,5,6,7');a=p.parse_args()
    c=read(ROOT/'settings.json');old_config=read(OLD_ROOT/'settings.json')
    if Path(c['destination'])!=FULL or Path(c['source'])!=SOURCE or c['augment_episode_fraction']!=1:
        raise RuntimeError('Unexpected paired configuration')
    if fingerprint(c)!=fingerprint(old_config):raise RuntimeError('Previous renderer config differs')
    if not FULL.exists():
        # The old worker processes retain their loaded config; record that config
        # independently so no later edit changes which complement we select.
        snapshot=ROOT/'old_settings_snapshot.json'
        if snapshot.exists() and read(snapshot)!=old_config:raise RuntimeError('Previous config changed since launch')
        write(snapshot,old_config)
        status('processing_missing_half',gpus=a.gpus,total_new_augmented_episodes=100)
        driver(a.gpus,'--workers-only','--complement-of',str(snapshot))
        status('waiting_for_original_job',lock=str(OLD_ROOT/'processing.lock'))
        with open(OLD_ROOT/'processing.lock','a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    time.sleep(30)
            # Original job has finished or stopped. Reuse only completed,
            # checksum-verified augmented reports; fill any missing ones below.
            if read(OLD_ROOT/'settings.json')!=old_config:raise RuntimeError('Old job settings changed; refusing reuse')
            reused=reuse_old(c,old_config)
        status('filling_gaps_and_validating',reused_augmented_episodes=reused)
        driver(a.gpus)
    status('merging_pairs',destination=str(PAIRS),independent_demonstrations=200,total_variants=400)
    from merge_pairs import build
    complete=build()
    status('complete',destination=str(PAIRS),mirror=str(MIRROR),
           total_episodes=complete['total_episodes'],total_frames=complete['total_frames'],
           original_episodes=complete['original_episodes'],augmented_episodes=complete['augmented_episodes'],
           actual_changed_frames=complete['actual_changed_frames'],
           unsafe_frames_kept_original=complete['unsafe_frames_kept_original'])
    print('PAIRED DATASET VALIDATED AND RSYNC VERIFIED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=str(e));raise
