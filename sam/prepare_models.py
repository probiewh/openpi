"""Prepare downloaded, local-only model artifacts. Never calls a paid API."""
import os
from pathlib import Path
os.environ['HF_HOME']='/root/data/xxy/openpi/sam_runtime/hf'
os.environ.setdefault('HF_ENDPOINT','https://huggingface.co')
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
target=Path('/root/data/xxy/openpi/sam_runtime/checkpoints/grounding-dino-tiny')
target.mkdir(parents=True,exist_ok=True)
if not (target/'config.json').exists() or not list(target.glob('*.safetensors')):
    print('Downloading official Grounding DINO tiny artifacts',flush=True)
    processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    model=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
    processor.save_pretrained(target)
    model.save_pretrained(target,safe_serialization=True)
print('Local detection model ready:',target,flush=True)
