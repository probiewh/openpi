#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/data1/xxy/openpi/sam
BASE=/root/data1/xxy/openpi/.venv
PYBASE=/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
mkdir -p "$ROOT" /root/data/xxy/openpi/sam_runtime
if [[ ! -x /root/data/xxy/openpi/sam_runtime/env/bin/python ]]; then
  "$PYBASE" -m venv /root/data/xxy/openpi/sam_runtime/env
fi
if [[ ! -e "$ROOT/.venv" ]]; then
  ln -s /root/data/xxy/openpi/sam_runtime/env "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
SITE=$($PY -c 'import site; print(site.getsitepackages()[0])')
# Reuse installed CUDA torch without upgrading or writing to the training environment.
printf '%s\n' "$BASE/lib/python3.11/site-packages" > "$SITE/openpi_readonly_dependencies.pth"
if [[ ! -d "$ROOT/vendor/sam2/.git" ]]; then
  mkdir -p "$ROOT/vendor"
  git clone --depth 1 https://github.com/facebookresearch/sam2.git "$ROOT/vendor/sam2"
fi
"$PY" -m pip install --disable-pip-version-check hydra-core==1.3.2 iopath==0.1.10
SAM2_BUILD_CUDA=0 "$PY" -m pip install --disable-pip-version-check --no-deps --no-build-isolation -e "$ROOT/vendor/sam2"
mkdir -p /root/data/xxy/openpi/sam_runtime/checkpoints
if [[ ! -s /root/data/xxy/openpi/sam_runtime/checkpoints/sam2.1_hiera_tiny.pt ]]; then
  "$PY" -u - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt','/root/data/xxy/openpi/sam_runtime/checkpoints/sam2.1_hiera_tiny.pt')
PY
fi
"$PY" -u - <<'PY'
import torch,cv2,hydra,sam2,pyarrow
print('Isolated environment ready:',torch.__version__,'CUDA:',torch.cuda.is_available())
PY
