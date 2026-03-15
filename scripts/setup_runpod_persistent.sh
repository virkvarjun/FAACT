#!/bin/bash
# RunPod setup with PERSISTENT storage. Clones to /workspace.
set -e

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
RESEARCH_DIR="${WORKSPACE_ROOT}/Research"

echo "=== [1/5] Cloning Research repo to /workspace (PERSISTENT) ==="
cd "${WORKSPACE_ROOT}"
if [ -d "Research" ]; then
    echo "Research dir exists, pulling latest..."
    cd Research
    git pull --recurse-submodules
    cd ..
else
    git clone --recurse-submodules https://github.com/virkvarjun/Research.git
fi

echo "=== [2/5] Installing Miniforge ==="
if [ -f "/root/miniforge3/bin/conda" ]; then
    echo "Miniforge already installed, skipping..."
else
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /root/miniforge3
    rm /tmp/miniforge.sh
fi

eval "$(/root/miniforge3/bin/conda shell.bash hook)"

echo "=== [3/5] Creating conda environment ==="
if conda env list | grep -q "lerobot"; then
    echo "lerobot env already exists, skipping..."
else
    conda create -y -n lerobot python=3.12
fi

conda activate lerobot

echo "=== [3.5/5] Installing system deps ==="
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev xvfb ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev tmux > /dev/null 2>&1

grep -q "MUJOCO_GL" ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl

echo "=== [4/5] Installing lerobot with aloha ==="
cd "${RESEARCH_DIR}/lerobot"
pip install -e ".[aloha]"
pip install "shimmy[gym-v26]"

echo "=== [5/5] Verifying ==="
python -c "
import torch
import gymnasium
print('OK: Python', torch.__version__, 'CUDA:', torch.cuda.is_available())
"

echo ""
echo "=== SETUP COMPLETE ==="
echo "  Research:  ${RESEARCH_DIR}"
echo "  Outputs:   ${RESEARCH_DIR}/outputs  (persistent on /workspace)"
echo ""
echo "Next: tmux new -s train"
echo "Then: cd ${RESEARCH_DIR} && bash scripts/train_act_persistent.sh transfer_cube 100000 8"
