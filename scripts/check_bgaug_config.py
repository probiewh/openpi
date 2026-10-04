"""Read-only checks for the paired background-augmentation training config."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openpi.models import gemma
from openpi.training import config as config_lib

NAME="first_task_4cam_bgaug_lora32"
REPO="competition/cup_four_cameras_bgaug32_sam2_pairs400"

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_config():
    c=config_lib.get_config(NAME);baseline=config_lib.get_config('first_task_4cam_lora32')
    assert c.model==baseline.model
    assert c.num_train_steps==baseline.num_train_steps==50_000
    assert c.batch_size==baseline.batch_size==32
    assert c.model.action_horizon==50 and c.model.action_dim==32
    assert c.ema_decay is None and repr(c.freeze_filter)==repr(baseline.freeze_filter)
    assert c.data.repo_id==REPO
    assert dataclasses.replace(c.data,repo_id=baseline.data.repo_id,assets=baseline.data.assets)==baseline.data
    for variant in [c.model.paligemma_variant,c.model.action_expert_variant]:
        assert all(x.rank==32 for x in gemma.get_config(variant).lora_configs.values())
    expected=c.assets_dirs/REPO/'norm_stats.json'
    lookup=Path(c.data.assets.assets_dir)/c.data.assets.asset_id/'norm_stats.json'
    assert expected==lookup and expected.is_file()
    original=Path('/root/data1/xxy/openpi/assets/first_task_4cam/competition/cup_four_cameras/norm_stats.json')
    assert digest(expected)==digest(original), 'Normalization differs from the baseline'
    stats=json.loads(expected.read_text())
    payload=stats.get('norm_stats',stats)
    for key in ('state','actions'):
        assert key in payload
        for field in ('mean','std','q01','q99'):
            a=np.asarray(payload[key][field]);assert np.all(np.isfinite(a)) and a.size in (26,32)
    root=Path(os.environ.get('HF_LEROBOT_HOME','/root/data/xxy/openpi/datasets'))/REPO
    manifest=json.loads((root/'PROCESSING_COMPLETE.json').read_text())
    info=json.loads((root/'meta/info.json').read_text())
    assert info['repo_id']==REPO and info['total_episodes']==400 and info['total_frames']==291956
    assert manifest['independent_demonstrations']==200
    assert manifest['original_episodes']==manifest['augmented_episodes']==200
    print('PASS: 4 cameras, LoRA 32/32, 50-step chunks, batch 32, 50k steps, complete pairs400 data and baseline-identical norm stats',flush=True)
    return c

def check_samples(c):
    from openpi.training import data_loader
    data=c.data.create(c.assets_dirs,c.model)
    assert data.norm_stats is not None and data.use_quantile_norm
    raw=data_loader.create_torch_dataset(data,c.model.action_horizon,c.model)
    transformed=data_loader.transform_dataset(raw,data)
    frames=145978;frame=330
    original=transformed[frame];augmented=transformed[frames+frame]
    assert np.array_equal(original['state'],augmented['state'])
    assert np.array_equal(original['actions'],augmented['actions'])
    assert original['actions'].shape==(50,32) and original['state'].shape==(32,)
    assert set(original['image'])==set(c.model.image_keys)
    for key in c.model.image_keys:
        assert original['image'][key].shape==augmented['image'][key].shape==(224,224,3)
        assert original['image_mask'][key] and augmented['image_mask'][key]
    print('PASS: actual dataset loads original/augmented pair through the OpenPI transforms; normalized state/actions match and four images have shape 224x224x3',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--samples',action='store_true');a=p.parse_args()
    c=check_config()
    if a.samples:check_samples(c)
