"""Shared provenance and safe paths for the paired-data pipeline."""
import hashlib
import json
from pathlib import Path

SOURCE = Path('/root/data/xxy/openpi/datasets/competition/cup_four_cameras')
OLD = SOURCE.parent / 'cup_four_cameras_bgaug32_sam2'
FULL = SOURCE.parent / 'cup_four_cameras_bgaug32_sam2_all200'
PAIRS = SOURCE.parent / 'cup_four_cameras_bgaug32_sam2_pairs400'
MIRROR = Path('/root/data1/xxy/huggingface/lerobot/competition') / PAIRS.name
OLD_ROOT = Path('/root/data1/xxy/openpi/sam')

def fingerprint(c):
    # Routing and selection do not change a given episode's rendered pixels.
    exclude = {'source', 'destination', 'mirror', 'repo_id', 'augment_episode_fraction'}
    payload = {k:v for k,v in c.items() if k not in exclude}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def relative(info, ep, key=None):
    args = dict(episode_index=ep, episode_chunk=ep//info['chunks_size'])
    if key is not None: args['video_key'] = key
    return info['video_path' if key is not None else 'data_path'].format(**args)
