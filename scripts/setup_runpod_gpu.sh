#!/bin/bash
# RunPod GPU — full setup for lerobot + failure prediction + Karpathy experiments
# Run this after SSH: ssh r6egq6hjpr6aol-64410ccc@ssh.runpod.io -i ~/.ssh/id_ed25519
#
# Usage: bash setup_runpod_gpu.sh
#
set -e

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/virkvarjun/lerobot-test.git}"
REPO_DIR="${WORKSPACE}/lerobot-test"
RESEARCH_DIR="${REPO_DIR}/Research"

echo "=== [1/6] Clone / update repo → ${REPO_DIR} ==="
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"
if [ -d "lerobot-test" ]; then
    echo "lerobot-test exists, pulling latest..."
    cd lerobot-test
    git pull --recurse-submodules
else
    git clone --recurse-submodules "${REPO_URL}" lerobot-test
    cd lerobot-test
fi
# Ensure submodules are populated (lerobot at repo root)
git submodule update --init --recursive

echo "=== [2/6] System deps (FFmpeg, EGL for headless MuJoCo) ==="
apt-get update -qq && apt-get install -y -qq \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    xvfb ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev \
    tmux \
    > /dev/null 2>&1

grep -q "MUJOCO_GL" ~/.bashrc 2>/dev/null || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl

echo "=== [3/6] Python environment (Miniforge + conda) ==="
if [ -f "/root/miniforge3/bin/conda" ]; then
    echo "Miniforge already installed"
else
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /root/miniforge3
    rm /tmp/miniforge.sh
fi
eval "$(/root/miniforge3/bin/conda shell.bash hook)"

if conda env list | grep -q "lerobot"; then
    echo "Conda env lerobot exists"
else
    conda create -y -n lerobot python=3.12
fi
conda activate lerobot

echo "=== [4/6] Install lerobot (aloha for sim) ==="
# lerobot submodule is at repo root, not inside Research/
cd "${REPO_DIR}/lerobot"
pip install -e ".[aloha]"
pip install "shimmy[gym-v26]"

echo "=== [5/6] Install failure_prediction + extras ==="
pip install matplotlib umap-learn scikit-learn  # for embedding viz, project figures

echo "=== [6/6] Verify ==="
python -c "
import torch
import gymnasium
import lerobot
print('OK: Python', torch.__version__)
print('OK: CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('OK: LeRobot')
"

echo ""
echo "=== SETUP COMPLETE ==="
echo "  Project:  ${RESEARCH_DIR}"
echo "  Train:    cd ${RESEARCH_DIR} && bash experiments/karpathy_autoresearch/scripts/train_faact_karpathy.sh runpod"
echo ""
echo "  Tip: tmux new -s train   (to persist if SSH drops)"
echo ""
