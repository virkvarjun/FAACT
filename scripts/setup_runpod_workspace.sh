#!/bin/bash
# Setup for RunPod when Research is already in /workspace (skip broken submodule clone).
# Run from: /workspace/Research (with lerobot already cloned at /workspace/Research/lerobot)
set -e

WORKSPACE="${WORKSPACE:-/workspace}"
RESEARCH_DIR="${WORKSPACE}/Research"

echo "=== [1/4] Installing Miniforge ==="
if [ -f "/root/miniforge3/bin/conda" ]; then
    echo "Miniforge already installed, skipping..."
else
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /root/miniforge3
    rm /tmp/miniforge.sh
fi

eval "$(/root/miniforge3/bin/conda shell.bash hook)"

echo "=== [2/4] Creating conda environment ==="
if conda env list | grep -q "lerobot"; then
    echo "lerobot env already exists, skipping..."
else
    conda create -y -n lerobot python=3.12
fi

conda activate lerobot

echo "=== [3/4] Installing system deps for headless MuJoCo ==="
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev xvfb > /dev/null 2>&1
apt-get install -y -qq ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev > /dev/null 2>&1

grep -q "MUJOCO_GL" ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
export MUJOCO_GL=egl

echo "=== [4/4] Installing lerobot with hilserl + aloha ==="
cd "${RESEARCH_DIR}/lerobot"
pip install -e ".[hilserl,aloha]"
pip install "shimmy[gym-v26]"

echo "=== SETUP COMPLETE ==="
echo "Run: cd ${RESEARCH_DIR} && bash scripts/train_act_persistent.sh"
